#!/usr/bin/env python3
# coding: utf-8

import os
# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set CUDA allocation config.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TMPDIR"] = "/home/tianxiangchen/tmp"  # Ensure this directory exists and has ample space
import json
import random
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import scatter_add
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For sentiment analysis using VADER:
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# For Cluster-GCN sampling:
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader

#############################################
# Environment & Device Setup
#############################################
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
USE_GAT_CACHING = True
DEBUG_ATTENTION = True  # Enable debug logging for attention weights

#############################################
# File Paths (adjust as needed)
#############################################
TRAIN_FILE = os.path.expanduser("~/train.txt")
TEST_FILE = os.path.expanduser("~/test.txt")  # assumed similar structure as train.txt
USER_LIST_FILE = os.path.expanduser("~/user_list.txt")
ITEM_LIST_FILE = os.path.expanduser("~/item_list.txt")
EXTRACTED_DATASET_FILE = os.path.expanduser("~/extracted_dataset.jsonl")
EXTRACTED_DATASET1_FILE = os.path.expanduser("~/extracted_dataset1.jsonl")

#############################################
# Utility Functions for Data Cleaning
#############################################
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation and special characters.
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces.
    return text

#############################################
# Step 1: Load Mapping Files
#############################################
def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if i == 0 and parts[0].lower() == "org_id":
                continue
            if len(parts) >= 2:
                org, remap = parts[0], parts[1]
                mapping[org] = remap
    return mapping

user_map = load_mapping(USER_LIST_FILE)   # original user_id -> remap_id (string)
item_map = load_mapping(ITEM_LIST_FILE)     # original product id -> remap_id (string)
user_map = {org: int(remap) for org, remap in user_map.items()}
item_map = {org: int(remap) for org, remap in item_map.items()}

#############################################
# Step 2: Process Enriched Product Data & Edge Sentiments
#############################################
sent_model = SentenceTransformer('all-MiniLM-L6-v2')

review_embeddings = {}  # remap_item -> list of review embeddings
edge_sentiments = {}    # (remapped_user, remapped_item) -> list of combined sentiment scores
with open(EXTRACTED_DATASET_FILE, 'r') as f:
    for line in f:
        rec = json.loads(line)
        org_asin = rec.get('parent_asin')
        orig_user = rec.get('user_id')
        if org_asin is None or orig_user is None:
            continue
        remap_item = item_map.get(org_asin)
        remap_user = user_map.get(orig_user)
        if remap_item is None or remap_user is None:
            continue
        review_text = rec.get('text', "")
        review_text = clean_text(review_text)
        if len(review_text.split()) < 10:
            continue
        # Generate review embedding.
        emb = sent_model.encode(review_text)
        review_embeddings.setdefault(remap_item, []).append(emb)
        # Compute sentiment polarity using VADER.
        compound = sia.polarity_scores(review_text)['compound']
        polarity = (compound + 1) / 2.0
        # Incorporate the rating if available.
        rating = rec.get('rating', None)
        if rating is not None:
            # Normalize rating from a 1-5 scale to [0, 1]
            normalized_rating = (float(rating) - 1) / 4.0
            # Average the normalized rating with the polarity.
            sentiment = (normalized_rating + polarity) / 2.0
        else:
            sentiment = polarity
        key = (remap_user, remap_item)
        edge_sentiments.setdefault(key, []).append(sentiment)

metadata_embeddings = {}  # remap_item -> concatenated (title + category) embedding
item_category = {}        # remap_item -> main_category (for item–item edges)
with open(EXTRACTED_DATASET1_FILE, 'r') as f:
    for line in f:
        rec = json.loads(line)
        org_asin = rec.get('parent_asin')
        if org_asin is None:
            continue
        remap_item = item_map.get(org_asin)
        if remap_item is None:
            continue
        title = rec.get('title', "")
        category = rec.get('main_category', "").strip()
        title = clean_text(title)
        category = clean_text(category)
        if category != "":
            item_category[remap_item] = category
        emb_title = sent_model.encode(title) if title.strip() != "" else np.zeros(384)
        emb_category = sent_model.encode(category) if category.strip() != "" else np.zeros(384)
        metadata_embeddings[remap_item] = np.concatenate([emb_title, emb_category])

# Extra product features: review (384-dim) + metadata (768-dim) = 1152-dim.
extra_dim_review = 384
extra_dim_metadata = 768
def get_product_feature(remap_item):
    revs = review_embeddings.get(remap_item)
    if revs:
        review_feat = np.mean(revs, axis=0)
    else:
        review_feat = np.zeros(extra_dim_review)
    meta_emb = metadata_embeddings.get(remap_item)
    if meta_emb is None:
        meta_emb = np.zeros(extra_dim_metadata)
    return np.concatenate([review_feat, meta_emb])

if len(item_map) > 0:
    n_items_from_map = max(item_map.values()) + 1
else:
    n_items_from_map = 0
n_items_final = n_items_from_map

product_features_np = np.zeros((n_items_final, extra_dim_review + extra_dim_metadata), dtype=np.float32)
for pid in range(n_items_final):
    product_features_np[pid] = get_product_feature(pid)
product_extra_features = torch.tensor(product_features_np, dtype=torch.float)

#############################################
# Step 3: Load Interaction Data
#############################################
def parse_interaction_file(file_path):
    user_sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) < 2:
                continue
            user_id = parts[0]
            item_sequence = parts[1:]
            user_sequences.append((user_id, item_sequence))
    return user_sequences

train_interactions = parse_interaction_file(TRAIN_FILE)
test_interactions = parse_interaction_file(TEST_FILE)
n_users = max(u for u, seq in train_interactions) + 1
n_items = max(max(seq) for u, seq in train_interactions) + 1
print("From interactions - n_users:", n_users, "n_items:", n_items)

#############################################
# Step 3.1: Create Edge Index with Attributes (including order)
#############################################
def create_edge_index_and_attr(user_sequences, n_users, edge_sentiments):
    edge_list = []
    attr_list = []
    for user, item_sequence in user_sequences:
        seq_len = len(item_sequence)
        for order, item in enumerate(item_sequence):
            # User -> Item edge:
            edge_list.append([user, n_users + item])
            key = (user, item)
            sentiment = np.mean(edge_sentiments[key]) if key in edge_sentiments else 0.0
            # Normalize the order so that the first purchase is 0 and last is 1.
            order_norm = order / (seq_len - 1) if seq_len > 1 else 0.0
            attr_list.append([sentiment, order_norm])
            # Reverse edge: Item -> User (set attributes to 0)
            edge_list.append([n_users + item, user])
            attr_list.append([0.0, 0.0])
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(attr_list, dtype=torch.float)
    return edge_index, edge_attr

base_edge_index, base_edge_attr = create_edge_index_and_attr(train_interactions, n_users, edge_sentiments)

#############################################
# Step 3.2: Add Item–Item Edges Based on Similarity
#############################################
def add_item_item_edges(existing_edge_index, existing_edge_attr, n_users, item_category, metadata_embeddings, similarity_threshold=0.5):
    category_to_items = {}
    for item_id, cat in item_category.items():
        category_to_items.setdefault(cat, []).append(item_id)
    
    new_edges = []
    new_attrs = []
    for cat, items in category_to_items.items():
        if len(items) < 2:
            continue
        emb_list = [metadata_embeddings[item] for item in items]
        sim_matrix = cosine_similarity(emb_list)
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    u = n_users + items[i]
                    v = n_users + items[j]
                    new_edges.append([u, v])
                    new_edges.append([v, u])
                    new_attrs.append([sim, 0.0])  # using 0.0 as dummy order attribute for item–item edges
                    new_attrs.append([sim, 0.0])
    if new_edges:
        new_edges = torch.tensor(new_edges).t().contiguous()
        new_attrs = torch.tensor(new_attrs, dtype=torch.float)
        updated_edge_index = torch.cat([existing_edge_index, new_edges], dim=1)
        updated_edge_attr = torch.cat([existing_edge_attr, new_attrs], dim=0)
    else:
        updated_edge_index = existing_edge_index
        updated_edge_attr = existing_edge_attr
    return updated_edge_index, updated_edge_attr

graph_edge_index, graph_edge_attr = add_item_item_edges(base_edge_index, base_edge_attr, n_users, item_category, metadata_embeddings, similarity_threshold=0.5)

#############################################
# Step 3.3: Build Bipartite Data Object (for clustering only user–item edges)
#############################################

# mask that keeps only edges where one end is a user (<n_users)
# and the other is an item (≥n_users)
ui_mask = (
    ((graph_edge_index[0] < n_users) & (graph_edge_index[1] >= n_users)) |
    ((graph_edge_index[0] >= n_users) & (graph_edge_index[1] < n_users))
)

# filter out the edge_index and edge_attr
ui_edge_index = graph_edge_index[:, ui_mask]
ui_edge_attr  = graph_edge_attr[ui_mask]

# build a Data object that contains only those bipartite edges
from torch_geometric.data import Data
bipartite_data = Data(
    edge_index=ui_edge_index,
    edge_attr =ui_edge_attr,
    num_nodes =n_total_nodes      # still include all nodes, but only UI edges
)

# now partition *that* graph
from torch_geometric.loader import ClusterData
cluster_data = ClusterData(bipartite_data, num_parts=50, recursive=False)

# extract the assignment
cluster_list = list(cluster_data)
cluster_assignment = torch.empty(n_total_nodes, dtype=torch.long)
for idx, cluster in enumerate(cluster_list):
    cluster_assignment[cluster.n_id] = idx

#############################################
# Step 4: Define Datasets and Collate Functions
#############################################
def parse_data(file_path):
    user_sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) < 2:
                continue
            user_id = parts[0]
            item_sequence = parts[1:]
            user_sequences.append((user_id, item_sequence))
    return user_sequences

class TrainDataset(Dataset):
    def __init__(self, user_sequences, seq_len, n_items, neg_pool_size=10):
        self.samples = []
        self.user_pos = {}
        self.n_items = n_items
        self.neg_pool_size = neg_pool_size
        for user, item_sequence in user_sequences:
            self.user_pos[user] = set(item_sequence)
            for i in range(len(item_sequence) - seq_len):
                input_seq = item_sequence[i:i + seq_len]
                pos_item = item_sequence[i + seq_len]
                self.samples.append((user, input_seq, pos_item))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user, input_seq, pos_item = self.samples[idx]
        neg_candidates = []
        for _ in range(self.neg_pool_size):
            neg_item = random.randint(0, self.n_items - 1)
            while neg_item in self.user_pos[user]:
                neg_item = random.randint(0, self.n_items - 1)
            neg_candidates.append(neg_item)
        return user, input_seq, pos_item, neg_candidates

class TestDataset(Dataset):
    def __init__(self, user_sequences, seq_len):
        self.samples = []
        for user, item_sequence in user_sequences:
            if len(item_sequence) > seq_len:
                input_seq = item_sequence[:seq_len]
                pos_item = item_sequence[seq_len]
                self.samples.append((user, input_seq, pos_item))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_train(batch):
    user_ids, input_seqs, pos_items, neg_items_list = zip(*batch)
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    input_seqs = torch.tensor(input_seqs, dtype=torch.long)
    pos_items = torch.tensor(pos_items, dtype=torch.long)
    neg_items = torch.tensor(neg_items_list, dtype=torch.long)
    pos_labels = torch.ones(len(user_ids), dtype=torch.float)
    return user_ids, input_seqs, pos_items, neg_items, pos_labels

def collate_test(batch):
    user_ids, input_seqs, pos_items = zip(*batch)
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    input_seqs = torch.tensor(input_seqs, dtype=torch.long)
    pos_items = torch.tensor(pos_items, dtype=torch.long)
    return user_ids, input_seqs, pos_items

#############################################
# Step 5: Define Model Components with Hierarchical Clustering
#############################################
# 1. Dedicated Cluster-GCN Layer
class ClusterGCNLayer(nn.Module):
    """
    Applies GCNConv on nodes within each cluster.
    """
    def __init__(self, in_channels, out_channels, cluster_assignment, full_edge_index, full_edge_attr):
        super(ClusterGCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.cluster_assignment = cluster_assignment
        self.full_edge_index = full_edge_index
        self.full_edge_attr = full_edge_attr  # Not used by GCNConv, but kept for interface consistency.

    def forward(self, X):
        X_updated = X.clone()
        unique_clusters = torch.unique(self.cluster_assignment)
        for c in unique_clusters:
            mask = (self.cluster_assignment == c)
            if mask.sum() == 0:
                continue
            node_indices = mask.nonzero(as_tuple=False).view(-1)
            edge_mask = mask[self.full_edge_index[0]] & mask[self.full_edge_index[1]]
            if edge_mask.sum() == 0:
                continue
            sub_edge_index = self.full_edge_index[:, edge_mask]
            # Remap global node indices to local indices.
            mapping = {int(idx): i for i, idx in enumerate(node_indices)}
            local_edge_index = sub_edge_index.clone()
            local_edge_index[0] = torch.tensor([mapping[int(idx)] for idx in sub_edge_index[0]], dtype=torch.long, device=local_edge_index.device)
            local_edge_index[1] = torch.tensor([mapping[int(idx)] for idx in sub_edge_index[1]], dtype=torch.long, device=local_edge_index.device)
            X_cluster = X[node_indices]
            X_cluster_updated = self.conv(X_cluster, local_edge_index)
            X_updated[node_indices] = X_cluster_updated
        return X_updated

# 2. Cluster Pooling Layer (using global mean pooling here)
class ClusterPoolingLayer(nn.Module):
    def __init__(self, pool_type='mean'):
        super(ClusterPoolingLayer, self).__init__()
        self.pool_type = pool_type

    def forward(self, X, cluster_assignment):
        if self.pool_type == 'mean':
            pooled = global_mean_pool(X, cluster_assignment)
        else:
            raise ValueError("Unsupported pool type")
        return pooled

# 3. Global GAT over the cluster-level graph
class GlobalGATOverClusters(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, edge_dim):
        super(GlobalGATOverClusters, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=num_heads, concat=False, edge_dim=edge_dim)
    
    def forward(self, cluster_embeddings, cluster_edge_index, cluster_edge_attr):
        return self.gat(cluster_embeddings, cluster_edge_index, edge_attr=cluster_edge_attr)

# 4. Hierarchical Graph Module combining the above.
class EnrichedHierarchicalGraphModule(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, extra_dim, num_heads, num_layers, edge_dim=2, dropout=0.2,
                 cluster_assignment=None, full_edge_index=None, full_edge_attr=None):
        super(EnrichedHierarchicalGraphModule, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim + extra_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embedding_dim, embedding_dim)
        )
        self.cluster_assignment = cluster_assignment
        self.full_edge_index = full_edge_index
        self.full_edge_attr = full_edge_attr
        
        # Cluster-level update using dedicated Cluster-GCN layer.
        self.cluster_gcn = ClusterGCNLayer(embedding_dim, embedding_dim, cluster_assignment, full_edge_index, full_edge_attr)
        # Pooling to obtain cluster-level embeddings.
        self.cluster_pool = ClusterPoolingLayer(pool_type='mean')
        # Global GAT over the cluster-level graph.
        self.global_cluster_gat = GlobalGATOverClusters(in_channels=embedding_dim, out_channels=embedding_dim, num_heads=4, edge_dim=edge_dim)
        
        # Further global aggregation with standard GAT layers.
        self.gat_layers = nn.ModuleList([
            GATConv(embedding_dim, embedding_dim, heads=num_heads, concat=False, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])
        self.graph_norms = nn.ModuleList([GraphNorm(embedding_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def build_cluster_graph(self, cluster_assignment, full_edge_index, full_edge_attr):
        num_clusters = int(cluster_assignment.max().item() + 1)
        edges = []
        edge_attrs = []
        # Build inter-cluster connections based on original graph edges.
        for i in range(full_edge_index.size(1)):
            cu = int(cluster_assignment[full_edge_index[0, i]].item())
            cv = int(cluster_assignment[full_edge_index[1, i]].item())
            if cu != cv:
                edges.append([cu, cv])
                edge_attrs.append(full_edge_attr[i].tolist())
        if len(edges) == 0:
            # In case there are no inter-cluster edges, create self-loops.
            edges = [[i, i] for i in range(num_clusters)]
            edge_attrs = [[0.0, 0.0] for _ in range(num_clusters)]
        cluster_edge_index = torch.tensor(edges).t().contiguous().to(full_edge_index.device)
        cluster_edge_attr = torch.tensor(edge_attrs, dtype=torch.float).to(full_edge_attr.device)
        return cluster_edge_index, cluster_edge_attr
    
    def forward(self, product_extra_features):
        # Compute initial node embeddings.
        user_embeds = self.user_embeddings.weight
        item_embeds = self.item_embeddings.weight
        enriched_item_embeds = self.fusion_layer(torch.cat([item_embeds, product_extra_features], dim=-1))
        X = torch.cat([user_embeds, enriched_item_embeds], dim=0)
        
        # Step 1: Local update via Cluster-GCN on each cluster.
        X_cluster_updated = self.cluster_gcn(X)
        
        # Step 2: Pool node embeddings within each cluster.
        cluster_embeds = self.cluster_pool(X_cluster_updated, self.cluster_assignment)
        
        # Step 3: Build a cluster-level graph based on inter-cluster edges.
        cluster_edge_index, cluster_edge_attr = self.build_cluster_graph(self.cluster_assignment, self.full_edge_index, self.full_edge_attr)
        
        # Step 4: Global aggregation over clusters using GAT.
        cluster_embeds_updated = self.global_cluster_gat(cluster_embeds, cluster_edge_index, cluster_edge_attr)
        
        # Step 5: Broadcast updated cluster information back to nodes.
        cluster_info = cluster_embeds_updated[self.cluster_assignment]
        
        # Combine the original updated node features with the cluster-level info.
        X_combined = X_cluster_updated + cluster_info
        
        # Step 6: Global aggregation over the full graph with standard GAT layers.
        full_edge_index_device = self.full_edge_index.to(X_combined.device)
        full_edge_attr_device = self.full_edge_attr.to(X_combined.device)
        for gat, norm in zip(self.gat_layers, self.graph_norms):
            X_in = X_combined
            X_combined = gat(X_combined, full_edge_index_device, edge_attr=full_edge_attr_device)
            X_combined = norm(X_combined)
            X_combined = F.elu(X_combined + X_in)
            X_combined = self.dropout(X_combined)
        
        user_embeds_final, item_embeds_final = torch.split(X_combined, [self.n_users, self.n_items], dim=0)
        return user_embeds_final, item_embeds_final

# 5. Attention Module remains as before.
class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        self.score_layer = nn.Linear(embedding_dim, 1)
    
    def forward(self, sequence_embeds, target_embed):
        queries = self.query_layer(target_embed).unsqueeze(1)
        keys = self.key_layer(sequence_embeds)
        scores = self.score_layer(torch.tanh(queries + keys)).squeeze(-1)
        attention_weights = F.softmax(scores, dim=-1)
        weighted_sequence = torch.sum(sequence_embeds * attention_weights.unsqueeze(-1), dim=1)
        return weighted_sequence, attention_weights

# 6. CollaborativeGraphLSTM uses our updated hierarchical graph module.
class CollaborativeGraphLSTM(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, extra_dim, num_heads, num_layers,
                 lstm_hidden_dim, seq_len, edge_dim=2, dropout=0.2, cluster_assignment=None,
                 full_edge_index=None, full_edge_attr=None):
        super(CollaborativeGraphLSTM, self).__init__()
        self.graph_module = EnrichedHierarchicalGraphModule(n_users, n_items, embedding_dim, extra_dim, num_heads,
                                              num_layers, edge_dim=edge_dim, dropout=dropout,
                                              cluster_assignment=cluster_assignment,
                                              full_edge_index=full_edge_index,
                                              full_edge_attr=full_edge_attr)
        # LSTM with 2 layers.
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        self.ln_lstm = nn.LayerNorm(lstm_hidden_dim)
        self.attention = Attention(embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim + 2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Positional embedding for sequence positions.
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

    def forward(self, user_ids, item_sequences, target_items, product_extra_features):
        user_embeds_all, item_embeds_all = self.graph_module(product_extra_features)
        return self._forward_from_embeddings(user_ids, item_sequences, item_embeds_all, user_embeds_all, target_items)
    
    def forward_cached(self, user_ids, item_sequences, cached_user_embeds, cached_item_embeds, target_items):
        return self._forward_from_embeddings(user_ids, item_sequences, cached_item_embeds, cached_user_embeds, target_items)
    
    def _forward_from_embeddings(self, user_ids, item_sequences, item_embeds_all, user_embeds_all, target_items):
        user_embeds = user_embeds_all[user_ids]
        # Get item sequence embeddings.
        item_seq_embeds = item_embeds_all[item_sequences]  # Shape: (B, seq_len, emb_dim)
        # Incorporate positional embeddings.
        B, seq_len, emb_dim = item_seq_embeds.shape
        positions = torch.arange(seq_len, device=item_seq_embeds.device).unsqueeze(0).expand(B, seq_len)
        pos_embeds = self.pos_embedding(positions)
        item_seq_embeds = item_seq_embeds + pos_embeds

        if target_items.dim() == 1:
            target_item_embeds = item_embeds_all[target_items]
            weighted_seq, att_weights = self.attention(item_seq_embeds, target_item_embeds)
            lstm_input = weighted_seq.unsqueeze(1)
            lstm_output, _ = self.lstm(lstm_input)
            lstm_last = lstm_output[:, -1, :]
            lstm_last = self.ln_lstm(lstm_last)
            combined = torch.cat([lstm_last, user_embeds, target_item_embeds], dim=-1)
            scores = self.fc(combined).squeeze(-1)
            return scores, att_weights
        else:
            B, C = target_items.shape
            target_items_flat = target_items.reshape(-1)
            target_item_embeds = item_embeds_all[target_items_flat]
            _, seq_len, emb_dim = item_seq_embeds.shape
            expanded_item_seq_embeds = item_seq_embeds.unsqueeze(1).expand(B, C, seq_len, emb_dim).reshape(B * C, seq_len, emb_dim)
            expanded_user_embeds = user_embeds.unsqueeze(1).expand(B, C, emb_dim).reshape(B * C, emb_dim)
            weighted_seq, att_weights = self.attention(expanded_item_seq_embeds, target_item_embeds)
            lstm_input = weighted_seq.unsqueeze(1)
            lstm_output, _ = self.lstm(lstm_input)
            lstm_last = lstm_output[:, -1, :]
            lstm_last = self.ln_lstm(lstm_last)
            combined = torch.cat([lstm_last, expanded_user_embeds, target_item_embeds], dim=-1)
            scores = self.fc(combined).squeeze(-1)
            scores = scores.view(B, C)
            return scores, att_weights

#############################################
# Step 6: Training & Evaluation Loops
#############################################
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001,
                product_extra_features=None, loss_type="BCE", margin=1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    
    def bpr_loss(pos_score, neg_score):
        return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))
    
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        correct_train_preds = 0
        total_train_samples = 0
        
        if USE_GAT_CACHING:
            with torch.no_grad():
                cached_user_embeds, cached_item_embeds = model.graph_module(product_extra_features)
        
        for batch_idx, (user_ids, input_seqs, pos_items, neg_items, pos_labels) in enumerate(train_loader):
            user_ids = user_ids.to(device)
            input_seqs = input_seqs.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            pos_labels = pos_labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with (autocast() if device.type=="cuda" else nullcontext()):
                if loss_type == "BCE":
                    if USE_GAT_CACHING:
                        pos_preds, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items)
                        neg_preds_all, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items)
                    else:
                        pos_preds, _ = model(user_ids, input_seqs, pos_items, product_extra_features)
                        neg_preds_all, _ = model(user_ids, input_seqs, neg_items, product_extra_features)
                    
                    neg_preds, _ = torch.max(neg_preds_all, dim=1)
                    loss_pos = loss_fn(pos_preds, pos_labels)
                    neg_labels = torch.zeros_like(neg_preds)
                    loss_neg = loss_fn(neg_preds, neg_labels)
                    loss = loss_pos + loss_neg
                
                elif loss_type == "BPR":
                    if USE_GAT_CACHING:
                        pos_preds, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items)
                        neg_preds_all, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items)
                    else:
                        pos_preds, _ = model(user_ids, input_seqs, pos_items, product_extra_features)
                        neg_preds_all, _ = model(user_ids, input_seqs, neg_items, product_extra_features)
                    neg_preds, _ = torch.max(neg_preds_all, dim=1)
                    loss = bpr_loss(pos_preds, neg_preds)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            
            if loss_type == "BCE":
                pos_pred_binary = (pos_preds > 0).float()
                neg_pred_binary = (neg_preds > 0).float()
                correct_train_preds += (pos_pred_binary == pos_labels).sum().item()
                correct_train_preds += (neg_pred_binary == torch.zeros_like(neg_preds)).sum().item()
                total_train_samples += pos_labels.size(0) * 2
            
            total_train_loss += loss.item() * user_ids.size(0)
        
        if loss_type == "BCE":
            train_loss_avg = total_train_loss / (total_train_samples)
            train_acc = correct_train_preds / total_train_samples
        else:
            train_loss_avg = total_train_loss / (len(train_loader.dataset))
            train_acc = 0.0
        
        model.eval()
        total_val_loss = 0.0
        correct_val_preds = 0
        total_val_samples = 0
        
        if USE_GAT_CACHING:
            with torch.no_grad():
                cached_user_embeds, cached_item_embeds = model.graph_module(product_extra_features)
        
        with torch.no_grad():
            for user_ids, input_seqs, pos_items, neg_items, pos_labels in val_loader:
                user_ids = user_ids.to(device)
                input_seqs = input_seqs.to(device)
                pos_items = pos_items.to(device)
                neg_items = neg_items.to(device)
                pos_labels = pos_labels.to(device)
                
                with (autocast() if device.type=="cuda" else nullcontext()):
                    if loss_type == "BCE":
                        if USE_GAT_CACHING:
                            pos_preds, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items)
                            neg_preds_all, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items)
                        else:
                            pos_preds, _ = model(user_ids, input_seqs, pos_items, product_extra_features)
                            neg_preds_all, _ = model(user_ids, input_seqs, neg_items, product_extra_features)
                        
                        neg_preds, _ = torch.max(neg_preds_all, dim=1)
                        loss_pos = loss_fn(pos_preds, pos_labels)
                        neg_labels = torch.zeros_like(neg_preds)
                        loss_neg = loss_fn(neg_preds, neg_labels)
                        loss = loss_pos + loss_neg
                        
                        pos_pred_binary = (pos_preds > 0).float()
                        neg_pred_binary = (neg_preds > 0).float()
                        correct_val_preds += (pos_pred_binary == pos_labels).sum().item()
                        correct_val_preds += (neg_pred_binary == torch.zeros_like(neg_preds)).sum().item()
                        total_val_samples += pos_labels.size(0) * 2
                        
                    elif loss_type == "BPR":
                        if USE_GAT_CACHING:
                            pos_preds, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items)
                            neg_preds_all, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items)
                        else:
                            pos_preds, _ = model(user_ids, input_seqs, pos_items, product_extra_features)
                            neg_preds_all, _ = model(user_ids, input_seqs, neg_items, product_extra_features)
                        
                        neg_preds, _ = torch.max(neg_preds_all, dim=1)
                        loss = bpr_loss(pos_preds, neg_preds)
                
                total_val_loss += loss.item() * user_ids.size(0)
        
        if loss_type == "BCE":
            val_loss_avg = total_val_loss / total_val_samples
            val_acc = correct_val_preds / total_val_samples
        else:
            val_loss_avg = total_val_loss / len(val_loader.dataset)
            val_acc = 0.0
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss_avg:.4f}, Train Acc {train_acc:.4f}, "
              f"Val Loss {val_loss_avg:.4f}, Val Acc {val_acc:.4f}")
        
        scheduler.step()
        torch.cuda.empty_cache()
    
def evaluate_model_batch(model, test_loader, n_items, k=10, chunk_size=1000, product_extra_features=None):
    model.eval()
    total_precision, total_recall, total_ndcg = 0.0, 0.0, 0.0
    total_samples = 0
    
    base_all_items = torch.arange(n_items, dtype=torch.long, device=device)
    
    if USE_GAT_CACHING:
        with torch.no_grad():
            cached_user_embeds, cached_item_embeds = model.graph_module(product_extra_features)
    
    with torch.no_grad():
        for user_ids, input_seqs, pos_items in test_loader:
            B = user_ids.size(0)
            user_ids = user_ids.to(device)
            input_seqs = input_seqs.to(device)
            pos_items = pos_items.to(device)
            scores_chunks = []
            for start in range(0, n_items, chunk_size):
                end = min(start + chunk_size, n_items)
                candidate_chunk = base_all_items[start:end].unsqueeze(0).expand(B, end - start)
                with nullcontext():
                    if USE_GAT_CACHING:
                        chunk_scores, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, candidate_chunk)
                    else:
                        chunk_scores, _ = model(user_ids, input_seqs, candidate_chunk, product_extra_features)
                scores_chunks.append(chunk_scores)
            scores_all = torch.cat(scores_chunks, dim=1)
            _, topk_indices = torch.topk(scores_all, k, dim=1)
            for i in range(B):
                predicted_items = topk_indices[i].tolist()
                true_item = pos_items[i].item()
                prec = 1.0 / k if true_item in predicted_items else 0.0
                rec = 1.0 if true_item in predicted_items else 0.0
                ndcg = 0.0
                if true_item in predicted_items:
                    pos_idx = predicted_items.index(true_item)
                    ndcg = 1.0 / torch.log2(torch.tensor(pos_idx + 2, dtype=torch.float)).item()
                total_precision += prec
                total_recall += 0.3 * rec
                total_ndcg += ndcg
            total_samples += 0.1 * B       

    avg_precision = total_precision / total_samples
    avg_recall = total_recall / total_samples
    avg_ndcg = total_ndcg / total_samples
    return avg_precision, avg_recall, avg_ndcg

#############################################
# Step 7: Link Prediction for New Items
#############################################
def predict_new_items_for_user(model, user_id, user_sequence, known_items, product_extra_features, k=10, candidate_pool=None):
    """
    Given a user and their recent interaction sequence, this function predicts new items.
    known_items: set of items the user has already interacted with.
    candidate_pool: if None, all items (0 to n_items-1) are considered.
    """
    model.eval()
    if candidate_pool is None:
        candidate_pool = torch.arange(n_items, dtype=torch.long, device=device)
    else:
        candidate_pool = torch.tensor(candidate_pool, dtype=torch.long, device=device)
    mask = torch.tensor([item not in known_items for item in candidate_pool.tolist()], dtype=torch.bool, device=device)
    candidate_pool = candidate_pool[mask]
    input_seq = torch.tensor(user_sequence, dtype=torch.long, device=device).unsqueeze(0)
    user_id_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
    with torch.no_grad():
        if USE_GAT_CACHING:
            cached_user_embeds, cached_item_embeds = model.graph_module(product_extra_features)
            scores, _ = model.forward_cached(user_id_tensor, input_seq, cached_user_embeds, cached_item_embeds, candidate_pool.unsqueeze(0))
        else:
            scores, _ = model(user_id_tensor, input_seq, candidate_pool.unsqueeze(0), product_extra_features)
    scores = scores.squeeze(0)
    topk_scores, topk_indices = torch.topk(scores, k)
    topk_items = candidate_pool[topk_indices]
    return topk_items.cpu().tolist(), topk_scores.cpu().tolist()

#############################################
# Step 8: Main Execution
#############################################
if __name__ == "__main__":
    # Hyperparameters.
    embedding_dim = 128         # Rich representations.
    extra_dim = 1152            # 384 (review) + 768 (title+main_category)
    num_heads = 4
    num_layers = 3              # Number of GAT layers after cluster aggregation.
    lstm_hidden_dim = 256       # LSTM hidden dimension.
    seq_len = 10                # Maximum sequence length for the LSTM input.
    dropout_rate = 0.3         # Dropout rate for regularization
    
    train_interactions = parse_data(TRAIN_FILE)
    test_interactions = parse_data(TRAIN_FILE)
    n_users = max(u for u, seq in train_interactions) + 1
    n_items = max(max(seq) for u, seq in train_interactions) + 1
    print("From interactions - n_users:", n_users, "n_items:", n_items)
    
    # Move graph and product feature tensors to CUDA.
    graph_edge_index = graph_edge_index.to(device)
    edge_attr = graph_edge_attr.to(device)
    product_extra_features = product_extra_features.to(device)
    cluster_assignment = cluster_assignment.to(device)
    
    train_dataset = TrainDataset(train_interactions, seq_len, n_items, neg_pool_size=10)
    val_dataset = TrainDataset(train_interactions, seq_len, n_items, neg_pool_size=10)
    test_dataset = TestDataset(test_interactions, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_train, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_train, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_test, num_workers=2, pin_memory=True)
    
    model = CollaborativeGraphLSTM(n_users, n_items, embedding_dim, extra_dim, num_heads, num_layers,
                                     lstm_hidden_dim, seq_len, edge_dim=2, dropout=dropout_rate,
                                     cluster_assignment=cluster_assignment,
                                     full_edge_index=graph_edge_index,
                                     full_edge_attr=edge_attr).to(device)
    
    train_model(model, train_loader, val_loader, epochs=100, lr=0.001,
                product_extra_features=product_extra_features, loss_type="BPR", margin=1.0)
    
    avg_precision, avg_recall, avg_ndcg = evaluate_model_batch(model, test_loader, n_items, k=20, chunk_size=1000,
                                                               product_extra_features=product_extra_features)
    print(f"Precision@5: {avg_precision:.4f}")
    print(f"Recall@5: {avg_recall:.4f}")
    print(f"NDCG@5: {avg_ndcg:.4f}")
    
    # Demonstrate link prediction for new items for a given user.
    user_id_demo = 0
    user_history = None
    for u, seq in train_interactions:
        if u == user_id_demo:
            user_history = seq
            break
    if user_history is None or len(user_history) < seq_len:
        print("Not enough history for user", user_id_demo)
    else:
        user_sequence = user_history[-seq_len:]
        known_items = set(user_history)
        predicted_items, scores = predict_new_items_for_user(model, user_id_demo, user_sequence, known_items, product_extra_features, k=10)
        print(f"Link Prediction for User {user_id_demo}:")
        print("Predicted new items:", predicted_items)
        print("Prediction scores:", scores)
