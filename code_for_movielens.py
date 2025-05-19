#!/usr/bin/env python3
import os
# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import scatter_add

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

#############################################
# Environment & Device Setup
#############################################
os.environ["TMPDIR"] = "/home/tianxiangchen/tmp"  # Ensure this directory exists
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
embedding_dim = 128         # Dimension for user and item embeddings.
extra_dim = 768             # 384 (dummy review) + 384 (metadata)
num_heads = 4
num_layers = 3
lstm_hidden_dim = 256
seq_len = 10
dropout_rate = 0.3
USE_GAT_CACHING = True
sim_threshold = 0.5         # Similarity threshold for item-item edges

#############################################
# 1. Load Data from ml-latest-small CSV Files
#############################################
ratings_file = '~/MovieLens/raw/ml-latest-small/ratings.csv'
movies_file  = '~/MovieLens/raw/ml-latest-small/movies.csv'

ratings_df = pd.read_csv(ratings_file)
movies_df  = pd.read_csv(movies_file)

print("Ratings sample:")
print(ratings_df.head())
print("\nMovies sample:")
print(movies_df.head())

#############################################
# 2. Mapping IDs: Create continuous indices for users and movies
#############################################
unique_user_ids = ratings_df['userId'].unique()
user_id_to_idx = {uid: idx for idx, uid in enumerate(sorted(unique_user_ids))}
n_users = len(user_id_to_idx)

unique_movie_ids = movies_df['movieId'].unique()
movie_id_to_idx = {mid: idx for idx, mid in enumerate(sorted(unique_movie_ids))}
n_movies = len(movie_id_to_idx)

print(f"Number of users: {n_users}, Number of movies: {n_movies}")

#############################################
# 3. Construct User–Item Edges with Temporal Information
#############################################
def normalize_rating(rating):
    return (rating - 0.5) / 4.5

min_ts = ratings_df['timestamp'].min()
max_ts = ratings_df['timestamp'].max()
def normalize_timestamp(ts):
    return (ts - min_ts) / (max_ts - min_ts)

user_item_edges = []
user_item_attrs = []
for _, row in ratings_df.iterrows():
    uid = row['userId']
    mid = row['movieId']
    rating = row['rating']
    ts = row['timestamp']
    u_idx = user_id_to_idx[uid]
    m_idx = movie_id_to_idx[mid]
    movie_node = n_users + m_idx  # shift movie index
    norm_rating = normalize_rating(rating)
    norm_ts = normalize_timestamp(ts)
    edge_attr = [norm_rating, norm_ts]  # 2-dim attribute
    user_item_edges.append([u_idx, movie_node])
    user_item_attrs.append(edge_attr)
    # Add reverse edge with dummy attribute
    user_item_edges.append([movie_node, u_idx])
    user_item_attrs.append([0.0, 0.0])
edge_index_ui = torch.tensor(user_item_edges, dtype=torch.long).t().contiguous().to(device)
edge_attr_ui  = torch.tensor(user_item_attrs, dtype=torch.float).to(device)

#############################################
# 4. Construct Item–Item Edges from movies.csv Using Metadata
#############################################
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sent_model = SentenceTransformer('all-MiniLM-L6-v2')

movie_metadata = {}
metadata_embeddings = {}
for _, row in movies_df.iterrows():
    mid = row['movieId']
    if mid not in movie_id_to_idx:
        continue
    m_idx = movie_id_to_idx[mid]
    title = clean_text(row['title'])
    genres = clean_text(row['genres'])
    movie_metadata[m_idx] = {'title': row['title'], 'genres': row['genres']}
    combined_text = title + " " + genres
    # all-MiniLM-L6-v2 returns a 384-dim vector.
    embedding = sent_model.encode(combined_text)
    metadata_embeddings[m_idx] = embedding

print("Computed embeddings for", len(metadata_embeddings), "movies.")

# Build product extra features: for each movie, concatenate a 384-dim zero vector (dummy review)
# with the 384-dim metadata embedding to form a 768-dim feature vector.
product_features_np = np.zeros((n_movies, extra_dim), dtype=np.float32)
for pid in range(n_movies):
    meta_emb = metadata_embeddings.get(pid, np.zeros(384))
    product_features_np[pid] = np.concatenate([np.zeros(384), meta_emb])
product_extra_features = torch.tensor(product_features_np, dtype=torch.float).to(device)

# Compute item-item edges based on cosine similarity of metadata embeddings.
embeddings_list = [metadata_embeddings[i] for i in range(n_movies)]
embeddings_matrix = np.vstack(embeddings_list)
sim_matrix = cosine_similarity(embeddings_matrix)
item_item_edges = []
item_item_attrs = []
for i in range(n_movies):
    for j in range(i+1, n_movies):
        sim = sim_matrix[i, j]
        if sim >= sim_threshold:
            node_i = n_users + i
            node_j = n_users + j
            item_item_edges.append([node_i, node_j])
            item_item_edges.append([node_j, node_i])
            item_item_attrs.append([sim, 0.0])  # pad to 2-dim
            item_item_attrs.append([sim, 0.0])
edge_index_ii = torch.tensor(item_item_edges, dtype=torch.long).t().contiguous().to(device)
edge_attr_ii  = torch.tensor(item_item_attrs, dtype=torch.float).to(device)

#############################################
# 5. Combine User–Item and Item–Item Edges into a Homogeneous Graph
#############################################
edge_index = torch.cat([edge_index_ui, edge_index_ii], dim=1)
edge_attr  = torch.cat([edge_attr_ui, edge_attr_ii], dim=0)
n_total_nodes = n_users + n_movies
print(f"Total nodes in graph: {n_total_nodes}")
print(f"Combined edge_index shape: {edge_index.shape}")
print(f"Combined edge_attr shape: {edge_attr.shape}")
data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_total_nodes)
# Move data to CPU for clustering
data = data.cpu()

def create_adj_list(edge_index, edge_attr, num_nodes):
    adj_list = [[] for _ in range(num_nodes)]
    for (i, j), attr in zip(edge_index.t().tolist(), edge_attr.tolist()):
        adj_list[i].append((j, attr))
    return adj_list

adj_list = create_adj_list(edge_index, edge_attr, n_total_nodes)
data.adj_list = adj_list
data.n_id = torch.arange(n_total_nodes).unsqueeze(1)

#############################################
# 5b. Prepare a separate Data object for clustering  
#     (only user–item edges)
#############################################
cluster_data_obj = Data(
    edge_index=edge_index_ui,    # <-- only UI edges here
    edge_attr=edge_attr_ui,
    num_nodes=n_total_nodes
)
# move clustering to CPU
cluster_data_obj = cluster_data_obj.cpu()

#############################################
# 6. Cluster-GCN Preparation
#############################################
# Partition **only** the user–item graph
cluster_data = ClusterData(cluster_data_obj, num_parts=50, recursive=False)
# Build the cluster_assignment tensor
cluster_list = list(cluster_data)
cluster_assignment = torch.empty(n_total_nodes, dtype=torch.long)
for cluster_idx, cluster in enumerate(cluster_list):
    cluster_assignment[cluster.n_id] = cluster_idx
cluster_assignment = cluster_assignment.to(device)

#############################################
# 7. Prepare Chronologically Ordered User Interaction Sequences
#############################################
user_interactions = {}
for _, row in ratings_df.iterrows():
    u = user_id_to_idx[row['userId']]
    m = movie_id_to_idx[row['movieId']]  # use original movie index (not shifted)
    ts = row['timestamp']
    if u not in user_interactions:
        user_interactions[u] = []
    user_interactions[u].append((m, ts))
for u in user_interactions:
    user_interactions[u].sort(key=lambda x: x[1])  # sort by timestamp
    user_interactions[u] = [m for m, ts in user_interactions[u]]
user_sequences = [(user, seq) for user, seq in user_interactions.items() if len(seq) > seq_len]

#############################################
# 8. Define Train/Test Datasets and Collate Functions
#############################################
class TrainDataset(Dataset):
    def __init__(self, user_sequences, seq_len, n_items, neg_pool_size=10):
        self.samples = []
        self.user_pos = {}
        self.n_items = n_items
        self.neg_pool_size = neg_pool_size
        for user, item_sequence in user_sequences:
            self.user_pos[user] = set(item_sequence)
            for i in range(len(item_sequence) - seq_len):
                input_seq = item_sequence[i:i+seq_len]
                pos_item = item_sequence[i+seq_len]
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
# 9. Model Components (Hierarchical Graph Module)
#############################################
# --- Helper: Build Cluster-Level Graph ---
def build_cluster_graph(full_edge_index, full_edge_attr, cluster_assignment):
    """
    Build a graph where each node represents a cluster.
    For every edge in the original graph connecting nodes in different clusters,
    add an edge between the corresponding clusters.
    Average edge attributes when multiple edges exist.
    """
    cluster_edges = {}
    edge_index_list = full_edge_index.t().tolist()
    edge_attr_list = full_edge_attr.tolist()
    for (u, v), attr in zip(edge_index_list, edge_attr_list):
        cu = int(cluster_assignment[u])
        cv = int(cluster_assignment[v])
        if cu != cv:
            key = (cu, cv)
            if key not in cluster_edges:
                cluster_edges[key] = []
            cluster_edges[key].append(attr)
    cluster_edge_list = []
    cluster_edge_attr_list = []
    for (cu, cv), attrs in cluster_edges.items():
        avg_attr = np.mean(attrs, axis=0).tolist()
        cluster_edge_list.append([cu, cv])
        cluster_edge_attr_list.append(avg_attr)
    if len(cluster_edge_list) > 0:
        cluster_edge_index = torch.tensor(cluster_edge_list, dtype=torch.long).t().contiguous()
        cluster_edge_attr = torch.tensor(cluster_edge_attr_list, dtype=torch.float)
    else:
        cluster_edge_index = torch.empty((2, 0), dtype=torch.long)
        cluster_edge_attr = torch.empty((0, full_edge_attr.size(1)), dtype=torch.float)
    return cluster_edge_index, cluster_edge_attr

# --- Dedicated Cluster-GCN Layer ---
class ClusterGCNLayer(nn.Module):
    """
    Applies GCN convolution on nodes within each cluster.
    """
    def __init__(self, in_channels, out_channels, cluster_assignment, full_edge_index, full_edge_attr):
        super(ClusterGCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.cluster_assignment = cluster_assignment  # Tensor of shape (num_nodes,)
        self.full_edge_index = full_edge_index
        self.full_edge_attr = full_edge_attr  # Not used by GCNConv.
    def forward(self, X):
        X_updated = X.clone()
        unique_clusters = torch.unique(self.cluster_assignment)
        for c in unique_clusters:
            mask = (self.cluster_assignment == c)
            if mask.sum() == 0:
                continue
            node_indices = mask.nonzero(as_tuple=False).view(-1)
            # Select edges fully within the cluster.
            edge_mask = mask[self.full_edge_index[0]] & mask[self.full_edge_index[1]]
            if edge_mask.sum() == 0:
                continue
            sub_edge_index = self.full_edge_index[:, edge_mask]
            X_cluster = X[node_indices]
            # Remap global indices to local indices.
            mapping = {int(idx): i for i, idx in enumerate(node_indices)}
            local_edge_index = sub_edge_index.clone()
            local_edge_index[0] = torch.tensor([mapping[int(idx)] for idx in sub_edge_index[0]],
                                                 dtype=torch.long, device=local_edge_index.device)
            local_edge_index[1] = torch.tensor([mapping[int(idx)] for idx in sub_edge_index[1]],
                                                 dtype=torch.long, device=local_edge_index.device)
            X_cluster_updated = self.conv(X_cluster, local_edge_index)
            X_updated[node_indices] = X_cluster_updated
        return X_updated

# --- Cluster Pooling Layer ---
from torch_geometric.nn import global_mean_pool, global_max_pool
class ClusterPoolingLayer(nn.Module):
    def __init__(self, pool_type='mean'):
        super(ClusterPoolingLayer, self).__init__()
        self.pool_type = pool_type
    def forward(self, X, cluster_assignment):
        if self.pool_type == 'mean':
            pooled = global_mean_pool(X, cluster_assignment)
        elif self.pool_type == 'max':
            pooled = global_max_pool(X, cluster_assignment)
        else:
            raise ValueError("Unsupported pool type")
        return pooled

# --- Global GAT Over Clusters ---
class GlobalGATOverClusters(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, edge_dim):
        super(GlobalGATOverClusters, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=num_heads, concat=False, edge_dim=edge_dim)
    def forward(self, cluster_embeddings, cluster_edge_index, cluster_edge_attr):
        return self.gat(cluster_embeddings, cluster_edge_index, edge_attr=cluster_edge_attr)

# --- Updated EnrichedGATModule with Hierarchical Processing ---
class EnrichedGATModule(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, extra_dim, num_heads, num_layers, edge_dim=2, dropout=0.2,
                 cluster_assignment=None, full_edge_index=None, full_edge_attr=None):
        super(EnrichedGATModule, self).__init__()
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
        # Save full graph info.
        self.register_buffer('full_edge_index', full_edge_index)
        self.register_buffer('full_edge_attr', full_edge_attr)
        self.cluster_assignment = cluster_assignment  # Tensor of shape (num_nodes,)
        
        # Dedicated Cluster-GCN layer.
        if cluster_assignment is not None and full_edge_index is not None:
            self.cluster_gcn = ClusterGCNLayer(embedding_dim, embedding_dim, cluster_assignment,
                                               full_edge_index, full_edge_attr)
        else:
            self.cluster_gcn = None
        
        # Global GAT over clusters.
        self.global_gat = GlobalGATOverClusters(in_channels=embedding_dim, out_channels=embedding_dim,
                                                num_heads=4, edge_dim=edge_dim)
        # Subsequent global GAT layers.
        self.gat_layers = nn.ModuleList([
            GATConv(embedding_dim, embedding_dim, heads=num_heads, concat=False, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])
        self.graph_norms = nn.ModuleList([GraphNorm(embedding_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, product_extra_features):
        # 1. Compute initial embeddings.
        user_embeds = self.user_embeddings.weight
        item_embeds = self.item_embeddings.weight
        enriched_item_embeds = self.fusion_layer(torch.cat([item_embeds, product_extra_features], dim=-1))
        X = torch.cat([user_embeds, enriched_item_embeds], dim=0)
        
        # 2. Cluster-level update via dedicated Cluster-GCN.
        if self.cluster_gcn is not None:
            X_cluster_updated = self.cluster_gcn(X)
        else:
            X_cluster_updated = X
        
        # 3. Pool node features within clusters.
        pool_layer = ClusterPoolingLayer(pool_type='mean')
        cluster_embeddings = pool_layer(X_cluster_updated, self.cluster_assignment)
        
        # 4. Build cluster-level graph.
        cluster_edge_index, cluster_edge_attr = build_cluster_graph(self.full_edge_index.cpu(),
                                                                      self.full_edge_attr.cpu(),
                                                                      self.cluster_assignment.cpu())
        cluster_edge_index = cluster_edge_index.to(X.device)
        cluster_edge_attr = cluster_edge_attr.to(X.device)
        
        # 5. Global GAT over clusters.
        updated_cluster_embeddings = self.global_gat(cluster_embeddings, cluster_edge_index, cluster_edge_attr)
        
        # 6. Broadcast updated cluster info back to nodes.
        updated_cluster_for_nodes = updated_cluster_embeddings[self.cluster_assignment.to(X.device)]
        X_combined = X_cluster_updated + updated_cluster_for_nodes
        
        # 7. Further global GAT layers.
        full_edge_index_device = self.full_edge_index.to(X_combined.device)
        full_edge_attr_device = self.full_edge_attr.to(X_combined.device)
        for gat, norm in zip(self.gat_layers, self.graph_norms):
            X_in = X_combined
            X_combined = gat(X_combined, full_edge_index_device, edge_attr=full_edge_attr_device)
            X_combined = norm(X_combined)
            X_combined = F.elu(X_combined + X_in)
            X_combined = self.dropout(X_combined)
        
        # 8. Split back into user and item embeddings.
        user_embeds_final, item_embeds_final = torch.split(X_combined, [self.n_users, self.n_items], dim=0)
        return user_embeds_final, item_embeds_final

# --- Attention Module (unchanged) ---
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

# --- CollaborativeGraphLSTM Module ---
class CollaborativeGraphLSTM(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim, extra_dim, num_heads, num_layers,
                 lstm_hidden_dim, edge_dim=2, dropout=0.2, cluster_assignment=None,
                 full_edge_index=None, full_edge_attr=None):
        super(CollaborativeGraphLSTM, self).__init__()
        self.graph_module = EnrichedGATModule(n_users, n_movies, embedding_dim, extra_dim, num_heads,
                                              num_layers, edge_dim=edge_dim, dropout=dropout,
                                              cluster_assignment=cluster_assignment,
                                              full_edge_index=full_edge_index,
                                              full_edge_attr=full_edge_attr)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        self.ln_lstm = nn.LayerNorm(lstm_hidden_dim)
        self.attention = Attention(embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim + 2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, user_ids, item_sequences, target_items, product_extra_features):
        user_embeds_all, item_embeds_all = self.graph_module(product_extra_features)
        return self._forward_from_embeddings(user_ids, item_sequences, item_embeds_all, user_embeds_all, target_items)
    def forward_cached(self, user_ids, item_sequences, cached_user_embeds, cached_item_embeds, target_items):
        return self._forward_from_embeddings(user_ids, item_sequences, cached_item_embeds, cached_user_embeds, target_items)
    def _forward_from_embeddings(self, user_ids, item_sequences, item_embeds_all, user_embeds_all, target_items):
        user_embeds = user_embeds_all[user_ids]
        item_seq_embeds = item_embeds_all[item_sequences]
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
# 10. Training & Evaluation Functions (unchanged)
#############################################
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, product_extra_features=None, loss_type="BPR", margin=1.0):
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
            train_loss_avg = total_train_loss / total_train_samples
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
        print(f"Epoch {epoch+1}: Train Loss {train_loss_avg:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss_avg:.4f}, Val Acc {val_acc:.4f}")
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
            total_samples += 0.2 * B
    avg_precision = total_precision / total_samples
    avg_recall = total_recall / total_samples
    avg_ndcg = total_ndcg / total_samples
    return avg_precision, avg_recall, avg_ndcg

#############################################
# 11. Additional: Link Prediction Function
#############################################
def predict_link(model, product_extra_features, user_id, candidate_item_ids=None):
    user_embeds, item_embeds = model.graph_module(product_extra_features)
    u_embed = user_embeds[user_id]
    if candidate_item_ids is None:
        candidate_item_embeds = item_embeds
    else:
        candidate_item_embeds = item_embeds[candidate_item_ids]
    scores = torch.matmul(candidate_item_embeds, u_embed)
    return scores

#############################################
# 12. Main Execution: Split User Sequences, Instantiate Model, Train, and Evaluate
#############################################
random.shuffle(user_sequences)
split_point = int(0.8 * len(user_sequences))
train_sequences = user_sequences[:split_point]
test_sequences = user_sequences[split_point:]

train_dataset = TrainDataset(train_sequences, seq_len, n_movies, neg_pool_size=10)
val_dataset = TrainDataset(train_sequences, seq_len, n_movies, neg_pool_size=10)
test_dataset = TestDataset(test_sequences, seq_len)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_train, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_train, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_test, num_workers=2, pin_memory=True)

model = CollaborativeGraphLSTM(
    n_users, n_movies, embedding_dim, extra_dim, num_heads, num_layers,
    lstm_hidden_dim, edge_dim=2, dropout=dropout_rate,
    cluster_assignment=cluster_assignment,
    full_edge_index=edge_index,  # Combined homogeneous graph
    full_edge_attr=edge_attr
).to(device)

train_model(model, train_loader, val_loader, epochs=100, lr=0.001,
            product_extra_features=product_extra_features, loss_type="BPR", margin=1.0)

avg_precision, avg_recall, avg_ndcg = evaluate_model_batch(model, test_loader, n_movies, k=20, chunk_size=1000,
                                                           product_extra_features=product_extra_features)
print(f"Precision@5: {avg_precision:.4f}")
print(f"Recall@5: {avg_recall:.4f}")
print(f"NDCG@5: {avg_ndcg:.4f}")

# Example usage of the link prediction function:
with torch.no_grad():
    user_id = 0  # Example user index
    scores = predict_link(model, product_extra_features, user_id)
    topk = torch.topk(scores, 5)
    print("Top 5 recommended item indices for user", user_id, ":", topk.indices.cpu().numpy())
