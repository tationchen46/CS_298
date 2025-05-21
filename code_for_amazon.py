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
# --- Mapping Loader Helper ---
def load_mapping(file_path):# Load ID remapping from a two-column text file
    mapping = {}# Initialize an empty dictionary to hold the mappings
    # Initialize an empty dictionary to hold the mappings
    with open(file_path, 'r') as f:
        lines = f.readlines()# Read all lines into a list
        # Process each line with its index
        for i, line in enumerate(lines):
            parts = line.strip().split()# Split on whitespace
            # Skip header if present (first line starting with 'org_id')
            if i == 0 and parts[0].lower() == "org_id":
                continue
            # Ensure there are at least two columns
            if len(parts) >= 2:
                org, remap = parts[0], parts[1]# Original ID and its new mapping
                mapping[org] = remap# Store as string for now
    return mapping# Return raw string-to-string mapping

# Load and convert mappings to integer IDs
user_map = load_mapping(USER_LIST_FILE)   # original user_id -> remap_id (string)
item_map = load_mapping(ITEM_LIST_FILE)     # original product id -> remap_id (string)
# Convert remapped IDs from strings to ints
user_map = {org: int(remap) for org, remap in user_map.items()}
item_map = {org: int(remap) for org, remap in item_map.items()}

#############################################
# Step 2: Process Enriched Product Data & Edge Sentiments
#############################################

# Load a pretrained SBERT model for encoding review and metadata text
sent_model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare containers:
review_embeddings = {}  # remap_item -> list of review embeddings
edge_sentiments = {}    # (remapped_user, remapped_item) -> list of combined sentiment scores

# Stream through your extracted JSON review file
with open(EXTRACTED_DATASET_FILE, 'r') as f:
    for line in f:
        rec = json.loads(line)# parse JSON record
        org_asin = rec.get('parent_asin')# original item ID
        orig_user = rec.get('user_id')# original user ID
        if org_asin is None or orig_user is None:# skip if missing
            continue
        remap_item = item_map.get(org_asin)# map to new item index
        remap_user = user_map.get(orig_user)# map to new user index
        if remap_item is None or remap_user is None:# skip if unknown mapping
            continue
        review_text = rec.get('text', "")# clean raw review text
        review_text = clean_text(review_text)
        if len(review_text.split()) < 10:# ignore very short reviews
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
        #Accumulate sentiment per edge (user→item)
        key = (remap_user, remap_item)
        edge_sentiments.setdefault(key, []).append(sentiment)

# Now load metadata (title + category) for each item
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

        # Clean title and category fields
        title = rec.get('title', "")
        category = rec.get('main_category', "").strip()
        title = clean_text(title)
        category = clean_text(category)
        if category != "":
            item_category[remap_item] = category

        # Embed title (or zero if missing)    
        emb_title = sent_model.encode(title) if title.strip() != "" else np.zeros(384)
        # Embed category (or zero if missing)
        emb_category = sent_model.encode(category) if category.strip() != "" else np.zeros(384)
        # Concatenate into one 768-D vector
        metadata_embeddings[remap_item] = np.concatenate([emb_title, emb_category])

# Extra product features: review (384-dim) + metadata (768-dim) = 1152-dim.
extra_dim_review = 384
extra_dim_metadata = 768
# Helper: for each item, average its review embeddings + attach its metadata embedding
def get_product_feature(remap_item):
    revs = review_embeddings.get(remap_item)
    # 1) mean over all its review embeddings, or zeros if none
    if revs:
        review_feat = np.mean(revs, axis=0)
    else:
        review_feat = np.zeros(extra_dim_review)
    # 2) pull its metadata embedding, or zeros if missing
    meta_emb = metadata_embeddings.get(remap_item)
    if meta_emb is None:
        meta_emb = np.zeros(extra_dim_metadata)
    return np.concatenate([review_feat, meta_emb])# final 1152-D vector

# Determine how many items you have after remapping
if len(item_map) > 0:
    n_items_from_map = max(item_map.values()) + 1
else:
    n_items_from_map = 0
n_items_final = n_items_from_map

# Build a big NumPy array of shape [n_items_final, 1152]
product_features_np = np.zeros((n_items_final, extra_dim_review + extra_dim_metadata), dtype=np.float32)
for pid in range(n_items_final):
    product_features_np[pid] = get_product_feature(pid)
# Convert into a PyTorch tensor for your model
product_extra_features = torch.tensor(product_features_np, dtype=torch.float)

#############################################
# Step 3: Load Interaction Data
#############################################
def parse_interaction_file(file_path):
    user_sequences = []# Will hold (user_id, [item_sequence]) tuples
    with open(file_path, 'r') as f:# Open the interactions file
        for line in f:# Read it line by line
            parts = list(map(int, line.strip().split()))# Split on whitespace and convert to int
            if len(parts) < 2:# Skip lines without at least one item
                continue
            user_id = parts[0]# The first number is the user ID
            item_sequence = parts[1:]# The rest form the ordered item sequence
            user_sequences.append((user_id, item_sequence))# Store the tuple
    return user_sequences# Return the list of all user → sequence mappings

# Parse your train and test files
train_interactions = parse_interaction_file(TRAIN_FILE)# [(u, [i1,i2,…]), …]
test_interactions = parse_interaction_file(TEST_FILE)
# Compute total counts
n_users = max(u for u, seq in train_interactions) + 1
n_items = max(max(seq) for u, seq in train_interactions) + 1
print("From interactions - n_users:", n_users, "n_items:", n_items)

#############################################
# Step 3.1: Create Edge Index with Attributes (including order)
#############################################

# --- Edge Index and Attribute Creation ---
def create_edge_index_and_attr(user_sequences, n_users, edge_sentiments):
    """
    Builds a bipartite edge index (user→item and item→user) and edge attributes
    capturing sentiment and normalized purchase order for the user→item edges.
    """
    edge_list = []# List to collect [src, dst] pairs
    attr_list = []# List to collect [sentiment, order_norm] per edge
    
    # Iterate over each user and their item sequence
    for user, item_sequence in user_sequences:
        seq_len = len(item_sequence)# Number of interactions
        for order, item in enumerate(item_sequence):
            # 1) User→Item edge: offset item index by n_users
            edge_list.append([user, n_users + item])
            # Compute average sentiment if available
            key = (user, item)
            sentiment = np.mean(edge_sentiments[key]) if key in edge_sentiments else 0.0
            # Normalize the order so that the first purchase is 0 and last is 1.
            order_norm = order / (seq_len - 1) if seq_len > 1 else 0.0
            # Store attributes for this directed edge
            attr_list.append([sentiment, order_norm])
            # Reverse edge: Item -> User (set attributes to 0)
            edge_list.append([n_users + item, user])
            attr_list.append([0.0, 0.0])
    # Convert lists to PyTorch tensors
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(attr_list, dtype=torch.float)
    return edge_index, edge_attr # Return directed edge index and attributes

# Create base edge index and attributes from training data
base_edge_index, base_edge_attr = create_edge_index_and_attr(train_interactions, n_users, edge_sentiments)
# Train interactions: [(user, [items]), ...]

#############################################
# Step 3.2: Add Item–Item Edges Based on Similarity
#############################################

# --- Add Item-Item Edges Function ---
def add_item_item_edges(existing_edge_index, existing_edge_attr, n_users, item_category, metadata_embeddings, similarity_threshold=0.5):
    """
    Adds item–item connections for items sharing a main_category when their metadata embeddings
    exceed the similarity_threshold. Returns an augmented edge_index and edge_attr.
    """

    # 1) Group items by their main category
    category_to_items = {} # category -> list of item IDs
    for item_id, cat in item_category.items():
        category_to_items.setdefault(cat, []).append(item_id)
    
    new_edges = []# List of new [src, dst] edges
    new_attrs = []# List of new [similarity, 0.0] attributes

    # 2) For each category, compute pairwise similarities
    for cat, items in category_to_items.items():
        if len(items) < 2:
            continue # Need at least two items to form edges
        emb_list = [metadata_embeddings[item] for item in items]# Metadata embeddings
        sim_matrix = cosine_similarity(emb_list)# Pairwise cosine similarities
        # 2a) Add edges for pairs above threshold
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    u = n_users + items[i]  # Global node index for item i
                    v = n_users + items[j]  # Global node index for item j
                    # Add both directional edges
                    new_edges.append([u, v])
                    new_edges.append([v, u])
                    new_attrs.append([sim, 0.0])  # using 0.0 as dummy order attribute for item–item edges
                    new_attrs.append([sim, 0.0])
    # 3) Concatenate new edges/attrs if any, else keep existing
    if new_edges:
        new_edges = torch.tensor(new_edges).t().contiguous()
        new_attrs = torch.tensor(new_attrs, dtype=torch.float)
        # Expand the graph
        updated_edge_index = torch.cat([existing_edge_index, new_edges], dim=1)
        updated_edge_attr = torch.cat([existing_edge_attr, new_attrs], dim=0)
    else:
        updated_edge_index = existing_edge_index
        updated_edge_attr = existing_edge_attr
    return updated_edge_index, updated_edge_attr # Return the augmented graph data
# Build the final graph with item–item edges included
graph_edge_index, graph_edge_attr = add_item_item_edges(base_edge_index, base_edge_attr, n_users, item_category, metadata_embeddings, similarity_threshold=0.5)

#############################################
# Step 3.3: Build Bipartite Data and Cluster Assignment
#############################################

# 1) Total number of nodes
n_total_nodes = n_users + n_items

# 2) Mask for user–item edges only
ui_mask = (
    ((graph_edge_index[0] < n_users) & (graph_edge_index[1] >= n_users)) |
    ((graph_edge_index[0] >= n_users) & (graph_edge_index[1] < n_users))
)
ui_edge_index = graph_edge_index[:, ui_mask]
ui_edge_attr  = graph_edge_attr[ui_mask]

# 3) Build the bipartite Data object
bipartite_data = Data(
    edge_index=ui_edge_index,
    edge_attr = ui_edge_attr,
    num_nodes = n_total_nodes
)

# 4) Partition into 50 clusters (using loader.ClusterData!)
from torch_geometric.loader import ClusterData
cluster_data = ClusterData(bipartite_data, num_parts=50, recursive=False)

# 5) Prepare the assignment tensor
cluster_assignment = torch.empty(n_total_nodes, dtype=torch.long)

# 6) Extract `partptr` and `node_perm` from the partition object
#    (these live under `cluster_data.partition` when you import from torch_geometric.loader) 
partptr    = cluster_data.partition.partptr      # shape [num_parts+1]
node_perm  = cluster_data.partition.node_perm    # shape [num_nodes]

# 7) For each cluster c, the nodes with global IDs node_perm[ partptr[c] : partptr[c+1] ]
#    belong to cluster c
for c in range(len(cluster_data)):
    start = int(partptr[c])
    end   = int(partptr[c + 1])
    node_ids = node_perm[start:end]               # Tensor of global node IDs
    cluster_assignment[node_ids] = c

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

#############################################
# This TrainDataset class turns each user’s interaction history into training examples for next-item
#prediction—splitting each sequence into fixed-length input windows plus the true “next” item—and on the
#fly samples a small set of items the user hasn’t seen as negatives. It also keeps a per-user record of
#positives so you never accidentally draw a positive item as a negative.
#############################################

class TrainDataset(Dataset):
    def __init__(self, user_sequences, seq_len, n_items, neg_pool_size=10):
        # Will hold (user, input_sequence, positive_next_item) tuples
        self.samples = []
        # Maps each user to the set of items they’ve interacted with
        self.user_pos = {}
        # Total number of items (for negative sampling range)
        self.n_items = n_items
        # How many negatives to sample per positive
        self.neg_pool_size = neg_pool_size
        # Build up samples and record which items each user has seen
        for user, item_sequence in user_sequences:
            # Store the user's full history as a set
            self.user_pos[user] = set(item_sequence)
            # Slide a window of length seq_len over their sequence
            for i in range(len(item_sequence) - seq_len):
                # The input is the last seq_len items
                input_seq = item_sequence[i : i + seq_len]
                # The target (positive) is the very next item
                pos_item = item_sequence[i + seq_len]
                # Record this training triple
                self.samples.append((user, input_seq, pos_item))
    
    def __len__(self):
        # Number of training examples
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve the idx-th example
        user, input_seq, pos_item = self.samples[idx]
        # On-the-fly negative sampling:
        neg_candidates = []
        for _ in range(self.neg_pool_size):
            # Draw a random item ID
            neg_item = random.randint(0, self.n_items - 1)
            # Retry if the user actually interacted with it
            while neg_item in self.user_pos[user]:
                neg_item = random.randint(0, self.n_items - 1)
            neg_candidates.append(neg_item)

        # Return: user ID, history list, positive target, list of negatives
        return user, input_seq, pos_item, neg_candidates

##This TestDataset class prepares fixed‐length “first-seq_len then next-item” examples for evaluation:
##for each user with a history longer than seq_len, it takes the very first seq_len interactions as the
##input sequence and the next interaction as the positive label. Unlike the training version, it doesn’t
##sample negatives—it just returns (user, input_seq, pos_item) tuples for testing.

# --- Test Dataset for Evaluation ---
class TestDataset(Dataset):  # Prepare fixed-length input and next-item target for testing
    """
    Takes each user’s interaction sequence, and if it's longer than seq_len,
    returns (user, first seq_len items, the (seq_len+1)-th item) for evaluation.
    """
    def __init__(self, user_sequences, seq_len):
        self.samples = []  # List to store (user, input_seq, pos_item) tuples
        for user, item_sequence in user_sequences:
            # Only consider users with at least seq_len+1 interactions
            if len(item_sequence) > seq_len:
                input_seq = item_sequence[:seq_len]      # First seq_len items as input
                pos_item  = item_sequence[seq_len]       # Next item as the positive label
                self.samples.append((user, input_seq, pos_item))  # Record sample

    def __len__(self):
        return len(self.samples)  # Total number of test examples

    def __getitem__(self, idx):
        # Retrieve the idx-th test sample
        return self.samples[idx]  # (user, input_seq, pos_item)

#The collate_train function takes a list of raw samples and:
#Unpacks them into separate lists of user IDs, input sequences, positive items, and negative‐item lists.
#Converts each into a batched torch.LongTensor ([B] for user IDs & pos_items, [B×L] for
#input_seqs, [B×N] for neg_items).
#Creates a float tensor of ones ([B]) as the positive‐label vector.
#It returns (user_ids, input_seqs, pos_items, neg_items, pos_labels) ready to feed into your training loop.


# --- Collate Function for Training ---
def collate_train(batch):  # Combine raw samples into batched tensors for training
    # Unpack batch tuples into separate lists
    user_ids, input_seqs, pos_items, neg_items_list = zip(*batch)
    # Convert lists of IDs and sequences into PyTorch tensors
    user_ids = torch.tensor(user_ids, dtype=torch.long)               # [B]
    input_seqs = torch.tensor(input_seqs, dtype=torch.long)           # [B, seq_len]
    pos_items = torch.tensor(pos_items, dtype=torch.long)             # [B]
    neg_items = torch.tensor(neg_items_list, dtype=torch.long)        # [B, neg_pool_size]
    # Create a tensor of ones as positive labels for BCE loss
    pos_labels = torch.ones(len(user_ids), dtype=torch.float)         # [B]
    # Return batched inputs and labels
    return user_ids, input_seqs, pos_items, neg_items, pos_labels


def collate_test(batch):# Combine raw test samples into batched tensors for evaluation
    user_ids, input_seqs, pos_items = zip(*batch)# Unpack batch tuples into separate lists
    # Convert lists to PyTorch tensors
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    input_seqs = torch.tensor(input_seqs, dtype=torch.long)
    pos_items = torch.tensor(pos_items, dtype=torch.long)
    # Return batched inputs for the test loop
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
        super(ClusterGCNLayer, self).__init__()# Initialize the parent nn.Module
        self.conv = GCNConv(in_channels, out_channels)## Graph convolution layer (GCN) for feature propagation
        self.cluster_assignment = cluster_assignment## Tensor mapping each node index to its cluster ID
        self.full_edge_index = full_edge_index# Edge index tensor of the full graph (source and target node pairs)
        self.full_edge_attr = full_edge_attr  # Edge attributes tensor (not used by GCNConv but stored for consistency)

    def forward(self, X):# Forward pass: apply cluster-wise GCN convolutions
        X_updated = X.clone()# Copy input features so we can update them without modifying original X
        unique_clusters = torch.unique(self.cluster_assignment)# Find all distinct cluster IDs
        for c in unique_clusters:# Iterate over each cluster
            mask = (self.cluster_assignment == c)# Boolean mask for nodes in cluster c
            if mask.sum() == 0:# Skip empty clusters
                continue
            node_indices = mask.nonzero(as_tuple=False).view(-1)# Get the indices of nodes in this cluster
            edge_mask = mask[self.full_edge_index[0]] & mask[self.full_edge_index[1]]# Mask edges where both endpoints are in this cluster
            if edge_mask.sum() == 0:# Skip if no intra-cluster edges
                continue
            sub_edge_index = self.full_edge_index[:, edge_mask]# Extract the edges within this cluster

            # Remap global node indices to local indices.
            mapping = {int(idx): i for i, idx in enumerate(node_indices)}# Map from global to local node indices
            local_edge_index = sub_edge_index.clone()# Copy edge index for local indexing
            # Replace global node IDs with local ones for source nodes
            local_edge_index[0] = torch.tensor([mapping[int(idx)] for idx in sub_edge_index[0]], dtype=torch.long, device=local_edge_index.device)
            # Replace global node IDs with local ones for target nodes
            local_edge_index[1] = torch.tensor([mapping[int(idx)] for idx in sub_edge_index[1]], dtype=torch.long, device=local_edge_index.device)
            X_cluster = X[node_indices]# Extract features for nodes in this cluster
            X_cluster_updated = self.conv(X_cluster, local_edge_index)## Apply GCN on the cluster subgraph
            X_updated[node_indices] = X_cluster_updated## Write updated features back into the correct positions
        return X_updated ## Return the feature tensor with per-cluster updates

# 2. Cluster Pooling Layer (using global mean pooling here)
class ClusterPoolingLayer(nn.Module):# Define a module for pooling node features by cluster
    def __init__(self, pool_type='mean'):
        super(ClusterPoolingLayer, self).__init__()# Initialize the parent nn.Module
        self.pool_type = pool_type# Store the pooling method (currently only 'mean')

    def forward(self, X, cluster_assignment):# Forward pass: pool features per cluster
        if self.pool_type == 'mean':# If mean pooling is desired
            pooled = global_mean_pool(X, cluster_assignment)# Compute mean of node features within each cluster
        else:
            # Raise an error for unsupported pooling types
            raise ValueError("Unsupported pool type")
        return pooled# Return the tensor of pooled cluster embeddings

# 3. Global GAT over the cluster-level graph
class GlobalGATOverClusters(nn.Module):# Define a module to apply GAT over cluster embeddings
    """
    Applies a multi-head Graph Attention (GAT) over cluster-level nodes to capture inter-cluster relationships.
    """
    def __init__(self, in_channels, out_channels, num_heads, edge_dim):
        super(GlobalGATOverClusters, self).__init__() # Initialize the base nn.Module
        # Initialize a GATConv layer: no concat, with specified edge feature dimension
        self.gat = GATConv(in_channels, out_channels, heads=num_heads, concat=False, edge_dim=edge_dim)
    
    def forward(self, cluster_embeddings, cluster_edge_index, cluster_edge_attr):# Forward pass: global attention
        # cluster_embeddings: [num_clusters, in_channels]
        # cluster_edge_index: [2, num_edges]
        # cluster_edge_attr: edge feature tensor for cluster-level graph
        # Returns updated cluster embeddings after attention
        return self.gat(cluster_embeddings, cluster_edge_index, edge_attr=cluster_edge_attr)

# 4. Hierarchical Graph Module combining the above.
class EnrichedHierarchicalGraphModule(nn.Module):# Main hierarchical graph module
    """
    Combines local cluster-level GCN, cluster pooling, global cluster GAT,
    and full-graph GAT layers to produce user and item embeddings.
    """
    def __init__(self, n_users, n_items, embedding_dim, extra_dim, num_heads, num_layers, edge_dim=2, dropout=0.2,
                 cluster_assignment=None, full_edge_index=None, full_edge_attr=None):
        super(EnrichedHierarchicalGraphModule, self).__init__()# Initialize base nn.Module
        self.n_users = n_users# Total number of users
        self.n_items = n_items# Total number of items
        # Learnable embeddings for users and items
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        # MLP to fuse item embeddings with extra features (e.g., attributes)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim + extra_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embedding_dim, embedding_dim)
        )
        # Store clustering and full-graph info for later use
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
        # Graph normalization for each GAT layer
        self.graph_norms = nn.ModuleList([GraphNorm(embedding_dim) for _ in range(num_layers)])
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def build_cluster_graph(self, cluster_assignment, full_edge_index, full_edge_attr):# Construct cluster-level graph
        num_clusters = int(cluster_assignment.max().item() + 1)# Number of clusters
        edges = []# List to collect inter-cluster edge pairs
        edge_attrs = []# List for corresponding edge feature vectors
        # Build inter-cluster connections based on original graph edges.
        for i in range(full_edge_index.size(1)):
            cu = int(cluster_assignment[full_edge_index[0, i]].item())# Cluster of source node
            cv = int(cluster_assignment[full_edge_index[1, i]].item())# Cluster of target node
            if cu != cv:# Only keep inter-cluster edges
                edges.append([cu, cv])
                edge_attrs.append(full_edge_attr[i].tolist())
        if len(edges) == 0:# Handle case with no inter-cluster edges
            # In case there are no inter-cluster edges, create self-loops.
            edges = [[i, i] for i in range(num_clusters)]
            edge_attrs = [[0.0, 0.0] for _ in range(num_clusters)]
        # Convert to PyTorch tensors
        cluster_edge_index = torch.tensor(edges).t().contiguous().to(full_edge_index.device)
        cluster_edge_attr = torch.tensor(edge_attrs, dtype=torch.float).to(full_edge_attr.device)
        return cluster_edge_index, cluster_edge_attr # Return cluster graph data
    
    def forward(self, product_extra_features):
        # Compute initial node embeddings.
        user_embeds = self.user_embeddings.weight# Shape: [n_users, embedding_dim]
        item_embeds = self.item_embeddings.weight# Shape: [n_items, embedding_dim]
        # Fuse item embeddings with extra features via MLP
        enriched_item_embeds = self.fusion_layer(torch.cat([item_embeds, product_extra_features], dim=-1))
        # Concatenate user and item features to form a single node feature matrix
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
        #Full-graph global GAT layers with residual, norm, and dropout
        full_edge_index_device = self.full_edge_index.to(X_combined.device)
        full_edge_attr_device = self.full_edge_attr.to(X_combined.device)
        for gat, norm in zip(self.gat_layers, self.graph_norms):
            X_in = X_combined# Preserve input for residual connection
            X_combined = gat(X_combined, full_edge_index_device, edge_attr=full_edge_attr_device)
            X_combined = norm(X_combined)# Normalize features
            X_combined = F.elu(X_combined + X_in)# Residual + activation
            X_combined = self.dropout(X_combined)# Apply dropout
            
        #Split back into separate user and item embeddings
        user_embeds_final, item_embeds_final = torch.split(X_combined, [self.n_users, self.n_items], dim=0)
        return user_embeds_final, item_embeds_final# Return final embeddings

# 5. Attention Module remains as before.
class Attention(nn.Module):# Module to compute attention-weighted summary of a sequence
    """
    Computes attention scores between a target embedding and a sequence of embeddings,
    producing a weighted sum of the sequence.
    """
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()# Initialize base nn.Module
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)  # Projects target to query space
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)    # Projects sequence embeddings to key space
        self.score_layer = nn.Linear(embedding_dim, 1)             # Scores compatibility of query-key pairs
    
    def forward(self, sequence_embeds, target_embed):  # Forward pass: compute attention
        # sequence_embeds: [batch_size, seq_len, embedding_dim]
        # target_embed:    [batch_size, embedding_dim]
        # 1) Compute queries from target and expand along sequence dimension
        queries = self.query_layer(target_embed).unsqueeze(1)  # Shape: [batch, 1, embedding_dim]
        # 2) Compute keys from each element in the sequence
        keys = self.key_layer(sequence_embeds)  # Shape: [batch, seq_len, embedding_dim]
        # 3) Compute raw scores via a learnable scoring mechanism
        scores = self.score_layer(torch.tanh(queries + keys)).squeeze(-1)  # Shape: [batch, seq_len]
        # 4) Normalize scores into attention weights
        attention_weights = F.softmax(scores, dim=-1)  # Shape: [batch, seq_len]
        # 5) Compute weighted sum of sequence embeddings
        weighted_sequence = torch.sum(
            sequence_embeds * attention_weights.unsqueeze(-1),
            dim=1
        )  # Shape: [batch, embedding_dim]
        return weighted_sequence, attention_weights  # Return summary vector and attention weights

# 6. CollaborativeGraphLSTM uses our updated hierarchical graph module.
class CollaborativeGraphLSTM(nn.Module):# Main recommendation module combining graph and sequence
    """
    Integrates hierarchical graph embeddings with LSTM and attention to predict next items.
    """
    def __init__(self, n_users, n_items, embedding_dim, extra_dim, num_heads, num_layers,
                 lstm_hidden_dim, seq_len, edge_dim=2, dropout=0.2, cluster_assignment=None,
                 full_edge_index=None, full_edge_attr=None):
        super(CollaborativeGraphLSTM, self).__init__()# Initialize base nn.Module
        # Graph module produces user/item embeddings
        self.graph_module = EnrichedHierarchicalGraphModule(n_users, n_items, embedding_dim, extra_dim, num_heads,
                                              num_layers, edge_dim=edge_dim, dropout=dropout,
                                              cluster_assignment=cluster_assignment,
                                              full_edge_index=full_edge_index,
                                              full_edge_attr=full_edge_attr)
        # LSTM with 2 layers.
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        # Normalize LSTM output
        self.ln_lstm = nn.LayerNorm(lstm_hidden_dim)
        # Attention to focus on relevant sequence parts
        self.attention = Attention(embedding_dim)
        # Final MLP to combine LSTM, user, and item information into a score
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim + 2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Positional embedding for sequence positions.
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

    def forward(self, user_ids, item_sequences, target_items, product_extra_features):# Full forward pass
        # Compute fresh embeddings from graph module
        user_embeds_all, item_embeds_all = self.graph_module(product_extra_features)
        # Delegate to helper that uses these embeddings
        return self._forward_from_embeddings(user_ids, item_sequences, item_embeds_all, user_embeds_all, target_items)
    
    def forward_cached(self, user_ids, item_sequences, cached_user_embeds, cached_item_embeds, target_items):
        # Use cached graph embeddings when available to save compute
        return self._forward_from_embeddings(user_ids, item_sequences, cached_item_embeds, cached_user_embeds, target_items)
    
    def _forward_from_embeddings(self, user_ids, item_sequences, item_embeds_all, user_embeds_all, target_items):
        # Gather per-batch user embeddings
        user_embeds = user_embeds_all[user_ids]# Shape: [B, emb_dim]
        # Get item sequence embeddings.
        item_seq_embeds = item_embeds_all[item_sequences]  # Shape: (B, seq_len, emb_dim)
        # Incorporate positional embeddings.
        B, seq_len, emb_dim = item_seq_embeds.shape
        positions = torch.arange(seq_len, device=item_seq_embeds.device).unsqueeze(0).expand(B, seq_len)
        pos_embeds = self.pos_embedding(positions)
        item_seq_embeds = item_seq_embeds + pos_embeds# Combine content and position info

        if target_items.dim() == 1:# Single target per sample
            # Get embeddings for each target item
            target_item_embeds = item_embeds_all[target_items]
            # Compute attention-weighted summary of the sequence
            weighted_seq, att_weights = self.attention(item_seq_embeds, target_item_embeds)
            lstm_input = weighted_seq.unsqueeze(1)# Shape: [B, 1, emb_dim]
            lstm_output, _ = self.lstm(lstm_input)# Run through LSTM
            lstm_last = lstm_output[:, -1, :]# Last time-step output
            lstm_last = self.ln_lstm(lstm_last)# Normalize LSTM output
            # Concatenate LSTM output, user embedding, and target item embedding
            combined = torch.cat([lstm_last, user_embeds, target_item_embeds], dim=-1)
            # Compute final score
            scores = self.fc(combined).squeeze(-1)# Shape: [B]
            return scores, att_weights# Return scores and attention weights
        else:# Multiple candidate targets per sample
            B, C = target_items.shape
            target_items_flat = target_items.reshape(-1)# Flatten to [B*C]
            target_item_embeds = item_embeds_all[target_items_flat]# Shape: [B*C, emb_dim]
            _, seq_len, emb_dim = item_seq_embeds.shape
            # Expand sequence and user embeddings to match candidate count
            expanded_item_seq_embeds = item_seq_embeds.unsqueeze(1).expand(B, C, seq_len, emb_dim).reshape(B * C, seq_len, emb_dim)
            expanded_user_embeds = user_embeds.unsqueeze(1).expand(B, C, emb_dim).reshape(B * C, emb_dim)
            # Compute attention-weighted sequence summary
            weighted_seq, att_weights = self.attention(expanded_item_seq_embeds, target_item_embeds)
            lstm_input = weighted_seq.unsqueeze(1)
            lstm_output, _ = self.lstm(lstm_input)
            lstm_last = lstm_output[:, -1, :]
            lstm_last = self.ln_lstm(lstm_last)
            # Combine features
            combined = torch.cat([lstm_last, expanded_user_embeds, target_item_embeds], dim=-1)
            scores = self.fc(combined).squeeze(-1)# Shape: [B*C]
            scores = scores.view(B, C)  # Reshape back to [B, C]
            return scores, att_weights  # Return scores matrix and attention weights

#############################################
# Step 6: Training & Evaluation Loops
#############################################
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001,
                product_extra_features=None, loss_type="BCE", margin=1.0):
    # 1) Optimizer and scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)# AdamW optimizer with weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)# Cosine LR schedule
    # 2) Define loss functions
    loss_fn = nn.BCEWithLogitsLoss()# Binary cross-entropy with logits for BCE mode
    
    def bpr_loss(pos_score, neg_score):# Bayesian Personalized Ranking loss
        return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))
    # 3) Mixed-precision scaler
    scaler = GradScaler()
    # 4) Training epochs loop
    for epoch in range(epochs):
        model.train()# Set model to training mode
        total_train_loss = 0.0# Accumulate loss
        correct_train_preds = 0# For tracking correct predictions (BCE)
        total_train_samples = 0# Number of prediction labels counted
        # 5) Optional GAT caching before batch loop
        if USE_GAT_CACHING:
            with torch.no_grad():
                cached_user_embeds, cached_item_embeds = model.graph_module(product_extra_features)# Precompute embeddings
                
        # 6) Iterate over training batches
        for batch_idx, (user_ids, input_seqs, pos_items, neg_items, pos_labels) in enumerate(train_loader):
            # Move all inputs to device
            user_ids = user_ids.to(device)
            input_seqs = input_seqs.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            pos_labels = pos_labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)# Reset gradients
            
            # 7) Forward pass with optional mixed precision
            with (autocast() if device.type=="cuda" else nullcontext()):
                # 7a) Compute positive and negative predictions
                if loss_type == "BCE":
                    if USE_GAT_CACHING:
                        pos_preds, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items)# Cached forward for pos
                        neg_preds_all, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items)# Cached forward for negs
                    else:
                        pos_preds, _ = model(user_ids, input_seqs, pos_items, product_extra_features)# Fresh forward for pos
                        neg_preds_all, _ = model(user_ids, input_seqs, neg_items, product_extra_features)# Fresh forward for negs
                    
                    neg_preds, _ = torch.max(neg_preds_all, dim=1)# Use hardest negative
                    # 7b) Compute BCE losses for pos and neg
                    loss_pos = loss_fn(pos_preds, pos_labels)
                    neg_labels = torch.zeros_like(neg_preds)
                    loss_neg = loss_fn(neg_preds, neg_labels)
                    loss = loss_pos + loss_neg
                
                elif loss_type == "BPR":# BPR loss branch
                    if USE_GAT_CACHING:
                        pos_preds, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items)
                        neg_preds_all, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items)
                    else:
                        pos_preds, _ = model(user_ids, input_seqs, pos_items, product_extra_features)
                        neg_preds_all, _ = model(user_ids, input_seqs, neg_items, product_extra_features)
                    neg_preds, _ = torch.max(neg_preds_all, dim=1)
                    loss = bpr_loss(pos_preds, neg_preds)
                    
            # 8) Backpropagation with gradient scaling
            scaler.scale(loss).backward()# Scale and backpropagate
            scaler.unscale_(optimizer)# Unscale for gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)# Clip gradients
            scaler.step(optimizer)# Update weights
            scaler.update()# Update scaler state
            
            # 9) Tracking accuracy for BCE mode
            if loss_type == "BCE":
                pos_pred_binary = (pos_preds > 0).float()
                neg_pred_binary = (neg_preds > 0).float()
                correct_train_preds += (pos_pred_binary == pos_labels).sum().item()# Positives correct
                correct_train_preds += (neg_pred_binary == torch.zeros_like(neg_preds)).sum().item()# Negatives correct
                total_train_samples += pos_labels.size(0) * 2# Two labels per sample
            
            total_train_loss += loss.item() * user_ids.size(0)# Accumulate weighted loss
            
        # 10) Compute epoch metrics
        if loss_type == "BCE":
            train_loss_avg = total_train_loss / (total_train_samples)
            train_acc = correct_train_preds / total_train_samples
        else:
            train_loss_avg = total_train_loss / (len(train_loader.dataset))
            train_acc = 0.0
        # 11) Validation phase
        model.eval()# Set to eval mode
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
                    # Similar forward and loss computation as training
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
        
        # 12) Compute validation metrics
        if loss_type == "BCE":
            val_loss_avg = total_val_loss / total_val_samples
            val_acc = correct_val_preds / total_val_samples
        else:
            val_loss_avg = total_val_loss / len(val_loader.dataset)
            val_acc = 0.0
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss_avg:.4f}, Train Acc {train_acc:.4f}, "
              f"Val Loss {val_loss_avg:.4f}, Val Acc {val_acc:.4f}")
        # 13) Scheduler step and cleanup
        scheduler.step()# Update learning rate
        torch.cuda.empty_cache()# Clear unused GPU memory

# --- Evaluation Loop ---    
def evaluate_model_batch(model, test_loader, n_items, k=10, chunk_size=1000, product_extra_features=None):
    model.eval()# Set model to evaluation mode
    total_precision, total_recall, total_ndcg = 0.0, 0.0, 0.0# Accumulators
    total_samples = 0# Counter for normalized metrics
    
    # Prepare tensor of all item IDs
    base_all_items = torch.arange(n_items, dtype=torch.long, device=device)
    
    # Optional cache graph embeddings
    if USE_GAT_CACHING:
        with torch.no_grad():
            cached_user_embeds, cached_item_embeds = model.graph_module(product_extra_features)
    # Disable gradient computation
    with torch.no_grad():
        for user_ids, input_seqs, pos_items in test_loader:# Batch loop
            B = user_ids.size(0)# Batch size
            user_ids = user_ids.to(device)# Move to device
            input_seqs = input_seqs.to(device)
            pos_items = pos_items.to(device)
            scores_chunks = []# Collect score slices
            # Score in chunks to avoid memory blowup
            for start in range(0, n_items, chunk_size):
                end = min(start + chunk_size, n_items)
                candidate_chunk = base_all_items[start:end].unsqueeze(0).expand(B, end - start)
                with nullcontext():
                    # Compute scores
                    if USE_GAT_CACHING:
                        chunk_scores, _ = model.forward_cached(user_ids, input_seqs, cached_user_embeds, cached_item_embeds, candidate_chunk)
                    else:
                        chunk_scores, _ = model(user_ids, input_seqs, candidate_chunk, product_extra_features)
                scores_chunks.append(chunk_scores)
            # Concatenate all chunks
            scores_all = torch.cat(scores_chunks, dim=1)
            # Get top-k item indices
            _, topk_indices = torch.topk(scores_all, k, dim=1)
            # Compute metrics per sample
            for i in range(B):
                predicted_items = topk_indices[i].tolist()# Predicted top-k
                true_item = pos_items[i].item()# Ground-truth
                prec = 1.0 / k if true_item in predicted_items else 0.0# Precision@k
                rec = 1.0 if true_item in predicted_items else 0.0# Recall@k
                ndcg = 0.0
                if true_item in predicted_items:
                    pos_idx = predicted_items.index(true_item)
                    ndcg = 1.0 / torch.log2(torch.tensor(pos_idx + 2, dtype=torch.float)).item()
                total_precision += prec
                total_recall += rec
                total_ndcg += ndcg
            total_samples += B # One test per user
            
    # Compute averages
    avg_precision = total_precision / total_samples
    avg_recall = total_recall / total_samples
    avg_ndcg = total_ndcg / total_samples
    return avg_precision, avg_recall, avg_ndcg# Return evaluation metrics

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
    
    train_model(model, train_loader, val_loader, epochs=35, lr=0.001,
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
