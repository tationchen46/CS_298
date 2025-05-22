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
# 1. Get an array of all unique user IDs from the ratings DataFrame
unique_user_ids = ratings_df['userId'].unique()

# 2. Sort the user IDs and build a dict mapping each raw ID to a 0-based index
user_id_to_idx = {
    uid: idx 
    for idx, uid in enumerate(sorted(unique_user_ids))
}

# 3. Count how many unique users we have
n_users = len(user_id_to_idx)

# 4. Get an array of all unique movie IDs from the movies DataFrame
unique_movie_ids = movies_df['movieId'].unique()

# 5. Sort the movie IDs and build a dict mapping each raw ID to a 0-based index
movie_id_to_idx = {
    mid: idx 
    for idx, mid in enumerate(sorted(unique_movie_ids))
}

# 6. Count how many unique movies we have
n_movies = len(movie_id_to_idx)

# 7. Print out the totals to confirm everything looks correct
print(f"Number of users: {n_users}, Number of movies: {n_movies}")


#############################################
# 3. Construct User–Item Edges with Temporal Information
#############################################
# Function to map ratings from [0.5, 5.0] → [0.0, 1.0]
def normalize_rating(rating):
    return (rating - 0.5) / 4.5

# Compute global min/max timestamps for normalization
min_ts = ratings_df['timestamp'].min()
max_ts = ratings_df['timestamp'].max()

# Function to map timestamps → [0.0, 1.0]
def normalize_timestamp(ts):
    return (ts - min_ts) / (max_ts - min_ts)

# Lists to accumulate edges and their 2-dim attributes
user_item_edges = []
user_item_attrs = []

# Loop over every row in the ratings DataFrame
for _, row in ratings_df.iterrows():
    # Extract raw IDs and values
    uid    = row['userId']
    mid    = row['movieId']
    rating = row['rating']
    ts     = row['timestamp']

    # Map raw IDs → zero-based indices
    u_idx = user_id_to_idx[uid]
    m_idx = movie_id_to_idx[mid]

    # Shift movie index by n_users so movie nodes come after user nodes
    movie_node = n_users + m_idx

    # Normalize rating and timestamp to [0,1]
    norm_rating = normalize_rating(rating)
    norm_ts     = normalize_timestamp(ts)

    # Pack into a 2-element attribute vector
    edge_attr = [norm_rating, norm_ts]

    # Add user→movie edge with its attribute
    user_item_edges.append([u_idx, movie_node])
    user_item_attrs.append(edge_attr)

    # Add reverse movie→user edge with dummy zeros
    user_item_edges.append([movie_node, u_idx])
    user_item_attrs.append([0.0, 0.0])

# Convert edge list → tensor of shape [2, num_edges] and move to device
edge_index_ui = torch.tensor(user_item_edges, dtype=torch.long).t().contiguous().to(device)

# Convert attribute list → tensor of shape [num_edges, 2] and move to device
edge_attr_ui  = torch.tensor(user_item_attrs, dtype=torch.float).to(device)


#############################################
# 4. Construct Item–Item Edges from movies.csv Using Metadata
#############################################
# Define a function to clean and normalize text strings
def clean_text(text):
    # Convert all characters to lowercase
    text = text.lower()
    # Remove any character that is not a lowercase letter, digit, or whitespace
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Collapse multiple spaces into one and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Initialize a pre-trained sentence-transformer model for embedding text
sent_model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare containers to hold raw metadata and their embeddings
movie_metadata = {}       # maps movie index → original metadata
metadata_embeddings = {}  # maps movie index → 384-dim embedding

# Iterate over each row in the movies DataFrame
for _, row in movies_df.iterrows():
    # Extract the raw movieId
    mid = row['movieId']
    # Skip any movie not present in our indexed mapping
    if mid not in movie_id_to_idx:
        continue
    # Look up the zero-based index for this movie
    m_idx = movie_id_to_idx[mid]
    # Clean the title text
    title = clean_text(row['title'])
    # Clean the genres text
    genres = clean_text(row['genres'])
    # Save the original (uncleaned) title and genres for reference
    movie_metadata[m_idx] = {'title': row['title'], 'genres': row['genres']}
    # Concatenate cleaned title and genres into one string
    combined_text = title + " " + genres
    # Encode this combined text into a 384-dimensional embedding
    embedding = sent_model.encode(combined_text)
    # Store the embedding in our dict keyed by movie index
    metadata_embeddings[m_idx] = embedding

# Report how many movies we successfully embedded
print("Computed embeddings for", len(metadata_embeddings), "movies.")


# Build product extra features: for each movie, concatenate a 384-dim zero vector (dummy review)
# with the 384-dim metadata embedding to form a 768-dim feature vector.

# Initialize a NumPy array of shape [n_movies, extra_dim] filled with zeros.
# This will hold the extra feature vectors for each movie.
product_features_np = np.zeros((n_movies, extra_dim), dtype=np.float32)

# Iterate over every movie index from 0 to n_movies–1
for pid in range(n_movies):
    # Look up the metadata embedding for this movie index.
    # If not found, use a zero vector of length 384 as a placeholder.
    meta_emb = metadata_embeddings.get(pid, np.zeros(384))
    
    # Build the final feature vector by concatenating:
    # 1) a zero-vector of length 384 (for any other extra features you might add)
    # 2) the 384-dimensional metadata embedding
    # The result is a vector of length extra_dim (384 + 384 = 768).
    product_features_np[pid] = np.concatenate([np.zeros(384), meta_emb])

# Convert the NumPy array into a PyTorch tensor of type float
# and move it to the target device (GPU or CPU).
product_extra_features = torch.tensor(product_features_np, dtype=torch.float).to(device)


# Compute item-item edges based on cosine similarity of metadata embeddings.
# Build a list of metadata embeddings in the correct order for all movies
embeddings_list = [metadata_embeddings[i] for i in range(n_movies)]

# Stack the list into a NumPy array of shape [n_movies, embedding_dim]
embeddings_matrix = np.vstack(embeddings_list)

# Compute pairwise cosine similarity between every pair of movie embeddings
sim_matrix = cosine_similarity(embeddings_matrix)

# Prepare containers for item–item edges and their attributes
item_item_edges = []
item_item_attrs = []

# Iterate over each unique movie pair (i < j) to avoid duplicates
for i in range(n_movies):
    for j in range(i + 1, n_movies):
        # Lookup the similarity score between movie i and movie j
        sim = sim_matrix[i, j]
        # Only consider edges where similarity exceeds the threshold
        if sim >= sim_threshold:
            # Compute the global node indices by shifting movie indices by n_users
            node_i = n_users + i
            node_j = n_users + j

            # Add the directed edge i → j
            item_item_edges.append([node_i, node_j])
            # Add the reverse edge j → i for an undirected effect
            item_item_edges.append([node_j, node_i])

            # Create a 2-dimensional attribute: [similarity, padding]
            item_item_attrs.append([sim, 0.0])
            item_item_attrs.append([sim, 0.0])

# Convert the edge list into a PyTorch tensor of shape [2, num_edges]
edge_index_ii = torch.tensor(
    item_item_edges, dtype=torch.long
).t().contiguous().to(device)

# Convert the attribute list into a PyTorch tensor of shape [num_edges, 2]
edge_attr_ii = torch.tensor(
    item_item_attrs, dtype=torch.float
).to(device)


#############################################
# 5. Combine User–Item and Item–Item Edges into a Homogeneous Graph
#############################################
# 1. Combine user–item and item–item edge indices into one [2, E_total] tensor
edge_index = torch.cat([edge_index_ui, edge_index_ii], dim=1)

# 2. Concatenate their corresponding edge attribute tensors into [E_total, 2]
edge_attr  = torch.cat([edge_attr_ui, edge_attr_ii], dim=0)

# 3. Compute total number of nodes (users + movies)
n_total_nodes = n_users + n_movies

# 4. Print the total node count for verification
print(f"Total nodes in graph: {n_total_nodes}")

# 5. Print shapes of combined edge_index and edge_attr for debugging
print(f"Combined edge_index shape: {edge_index.shape}")
print(f"Combined edge_attr shape: {edge_attr.shape}")

# 6. Create a PyG Data object with edges, attributes, and node count
data = Data(
    edge_index=edge_index,
    edge_attr=edge_attr,
    num_nodes=n_total_nodes
)

# 7. Move the Data object to CPU memory (required by some clustering routines)
data = data.cpu()

# 8. Helper function to build a Python adjacency list with attributes
def create_adj_list(edge_index, edge_attr, num_nodes):
    # Initialize a list of empty lists, one per node
    adj_list = [[] for _ in range(num_nodes)]
    # Iterate over each edge (i→j) and its attribute vector
    for (i, j), attr in zip(edge_index.t().tolist(), edge_attr.tolist()):
        # Append the target node and attribute to node i's list
        adj_list[i].append((j, attr))
    return adj_list

# 9. Generate the adjacency list for the graph
adj_list = create_adj_list(edge_index, edge_attr, n_total_nodes)

# 10. Attach the adjacency list to the Data object for later use
data.adj_list = adj_list

# 11. Create a tensor of node IDs [ [0], [1], ..., [n_total_nodes-1] ]
data.n_id = torch.arange(n_total_nodes).unsqueeze(1)


#############################################
# 5b. Prepare a separate Data object for clustering  
#     (only user–item edges)
#############################################
# 1) Total number of nodes
n_total_nodes=n_users+n_movies
cluster_data_obj = Data(
    edge_index=edge_index_ui,    # <-- only UI edges here
    edge_attr=edge_attr_ui,
    num_nodes=n_total_nodes
).cpu()
# move clustering to CPU
#cluster_data_obj = cluster_data_obj.cpu()

#############################################
# 6. Cluster-GCN Preparation
#############################################
# Partition **only** the user–item graph
cluster_data = ClusterData(cluster_data_obj, num_parts=50, recursive=False)

# Prepare the assignment tensor
cluster_assignment = torch.empty(n_total_nodes, dtype=torch.long)
# Extract `partptr` and `node_perm` from the partition object
#    (these live under `cluster_data.partition` when you import from torch_geometric.loader) 
partptr    = cluster_data.partition.partptr      # shape [num_parts+1]
node_perm  = cluster_data.partition.node_perm    # shape [num_nodes]

#For each cluster c, the nodes with global IDs node_perm[ partptr[c] : partptr[c+1] ]
#    belong to cluster c
for c in range(len(cluster_data)):
    start = int(partptr[c])
    end   = int(partptr[c + 1])
    node_ids = node_perm[start:end]               # Tensor of global node IDs
    cluster_assignment[node_ids] = c

#############################################
# 7. Prepare Chronologically Ordered User Interaction Sequences
#############################################
# 1. Initialize a dictionary to collect each user’s (movie, timestamp) interactions
user_interactions = {}

# 2. Loop over every rating record in the DataFrame
for _, row in ratings_df.iterrows():
    # 2a. Map the raw userId to its zero-based index
    u = user_id_to_idx[row['userId']]
    # 2b. Map the raw movieId to its zero-based index
    m = movie_id_to_idx[row['movieId']]
    # 2c. Get the timestamp of this interaction
    ts = row['timestamp']

    # 2d. If this user hasn’t appeared yet, create an empty list for them
    if u not in user_interactions:
        user_interactions[u] = []

    # 2e. Append the tuple (movie_idx, timestamp) to this user’s list
    user_interactions[u].append((m, ts))

# 3. For each user, sort their interactions by ascending timestamp
for u in user_interactions:
    # 3a. Sort the list of (movie, ts) tuples by the ts field
    user_interactions[u].sort(key=lambda x: x[1])
    # 3b. Replace the list of tuples with just the ordered movie sequence
    user_interactions[u] = [m for m, ts in user_interactions[u]]

# 4. Build the final list of (user_idx, interaction_sequence) pairs,
#    keeping only those with sequence length > seq_len
user_sequences = [
    (user, seq)
    for user, seq in user_interactions.items()
    if len(seq) > seq_len
]

#############################################
# 8. Define Train/Test Datasets and Collate Functions
#############################################
from torch.utils.data import Dataset
import random

class TrainDataset(Dataset):
    """
    PyTorch Dataset for training sequence-based recommenders.
    Generates (user, input_sequence, positive_item, negative_candidates) samples.
    """
    def __init__(self, user_sequences, seq_len, n_items, neg_pool_size=10):
        # List to hold all (user, input_seq, pos_item) tuples
        self.samples = []
        # Dict mapping user → set of all items they've interacted with
        self.user_pos = {}
        # Total number of items (for sampling negatives)
        self.n_items = n_items
        # How many negative samples to draw per positive
        self.neg_pool_size = neg_pool_size

        # Build samples from each user's interaction sequence
        for user, item_sequence in user_sequences:
            # Record all items this user has seen
            self.user_pos[user] = set(item_sequence)
            # Slide a window of length `seq_len` across the sequence
            for i in range(len(item_sequence) - seq_len):
                # The input sequence of historical items
                input_seq = item_sequence[i : i + seq_len]
                # The next item as the positive target
                pos_item = item_sequence[i + seq_len]
                # Store the training sample tuple
                self.samples.append((user, input_seq, pos_item))

    def __len__(self):
        # Total number of training examples
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve the (user, input_seq, pos_item) for this index
        user, input_seq, pos_item = self.samples[idx]

        # Prepare a list of negative candidates not in this user's history
        neg_candidates = []
        for _ in range(self.neg_pool_size):
            # Sample a random item index
            neg_item = random.randint(0, self.n_items - 1)
            # Re-sample if it's actually a positive for this user
            while neg_item in self.user_pos[user]:
                neg_item = random.randint(0, self.n_items - 1)
            neg_candidates.append(neg_item)

        # Return the user ID, the input sequence, the positive item,
        # and a list of negative item indices
        return user, input_seq, pos_item, neg_candidates


class TestDataset(Dataset):
    """
    PyTorch Dataset for testing sequence-based recommenders.
    Generates (user, input_sequence, positive_item) samples.
    """
    def __init__(self, user_sequences, seq_len):
        # 1. Initialize list to hold all test samples
        self.samples = []
        # 2. Loop over each (user, full interaction sequence)
        for user, item_sequence in user_sequences:
            # 3. Only consider sequences longer than the context length
            if len(item_sequence) > seq_len:
                # 4. Take the first `seq_len` items as the input history
                input_seq = item_sequence[:seq_len]
                # 5. The next item in the sequence is the positive test target
                pos_item = item_sequence[seq_len]
                # 6. Store the tuple (user, input_sequence, positive_item)
                self.samples.append((user, input_seq, pos_item))

    def __len__(self):
        # 7. Return the total number of test samples
        return len(self.samples)

    def __getitem__(self, idx):
        # 8. Retrieve and return the idx-th test sample
        return self.samples[idx]

def collate_train(batch):
    """
    Collate function to combine a list of training samples into batched tensors.
    Each sample is a tuple: (user_id, input_seq, pos_item, neg_items_list).
    """
    # Unzip the batch into separate lists:
    #   user_ids:        tuple of user indices
    #   input_seqs:      tuple of input sequences (each a list of item indices)
    #   pos_items:       tuple of positive (next) item indices
    #   neg_items_list:  tuple of lists of negative item indices
    user_ids, input_seqs, pos_items, neg_items_list = zip(*batch)

    # Convert user IDs to a LongTensor of shape [batch_size]
    user_ids = torch.tensor(user_ids, dtype=torch.long)

    # Convert input sequences to a LongTensor of shape [batch_size, seq_len]
    input_seqs = torch.tensor(input_seqs, dtype=torch.long)

    # Convert positive items to a LongTensor of shape [batch_size]
    pos_items = torch.tensor(pos_items, dtype=torch.long)

    # Convert negative candidate lists to a LongTensor of shape [batch_size, neg_pool_size]
    neg_items = torch.tensor(neg_items_list, dtype=torch.long)

    # Create a FloatTensor of ones for positive labels: shape [batch_size]
    pos_labels = torch.ones(len(user_ids), dtype=torch.float)

    # Return all batched tensors
    return user_ids, input_seqs, pos_items, neg_items, pos_labels


def collate_test(batch):
    """
    Collate function to combine a list of test samples into batched tensors.
    Each sample is a tuple: (user_id, input_seq, pos_item).
    """
    # Unzip the batch into separate lists:
    #   user_ids:   tuple of user indices
    #   input_seqs: tuple of input sequences (each a list of item indices)
    #   pos_items:  tuple of positive (next) item indices
    user_ids, input_seqs, pos_items = zip(*batch)

    # Convert user IDs to a LongTensor of shape [batch_size]
    user_ids = torch.tensor(user_ids, dtype=torch.long)

    # Convert input sequences to a LongTensor of shape [batch_size, seq_len]
    input_seqs = torch.tensor(input_seqs, dtype=torch.long)

    # Convert positive items to a LongTensor of shape [batch_size]
    pos_items = torch.tensor(pos_items, dtype=torch.long)

    # Return the batched tensors
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
    # A dict to collect lists of attributes for each inter-cluster edge key
    cluster_edges = {}
    # Transpose edge_index ([2, E]) to get a list of (u, v) pairs of length E
    edge_index_list = full_edge_index.t().tolist()
    # Convert the edge_attr tensor ([E, F]) to a plain Python list of length E
    edge_attr_list = full_edge_attr.tolist()
    # Iterate over every original edge and its attributes
    for (u, v), attr in zip(edge_index_list, edge_attr_list):
        # Look up which cluster node u belongs to
        cu = int(cluster_assignment[u])
        # Look up which cluster node v belongs to
        cv = int(cluster_assignment[v])
        # Only consider edges that go between two different clusters
        if cu != cv:
            # Use the tuple (cu, cv) as the key to group edges
            key = (cu, cv)
            # If this is the first time we see this cluster pair, start a new list
            if key not in cluster_edges:
                cluster_edges[key] = []
            # Append the original edge’s attribute vector to that list
            cluster_edges[key].append(attr)
    # Prepare lists to build the final tensors
    cluster_edge_list = []
    cluster_edge_attr_list = []
    # For each unique inter-cluster key, average its collected attributes
    for (cu, cv), attrs in cluster_edges.items():
        # Compute mean across all attribute vectors in attrs (axis=0)
        avg_attr = np.mean(attrs, axis=0).tolist()
        # Record the cluster-to-cluster edge
        cluster_edge_list.append([cu, cv])
        # Record the averaged attribute for that edge
        cluster_edge_attr_list.append(avg_attr)
    # If we found any inter-cluster edges, convert lists back to tensors
    if len(cluster_edge_list) > 0:
        # edge_index should be shape [2, num_edges]
        cluster_edge_index = torch.tensor(cluster_edge_list, dtype=torch.long).t().contiguous()
        # edge_attr should be shape [num_edges, feature_dim]
        cluster_edge_attr = torch.tensor(cluster_edge_attr_list, dtype=torch.float)
    else:
        # No edges found → return empty tensors with correct shapes
        cluster_edge_index = torch.empty((2, 0), dtype=torch.long)
        cluster_edge_attr = torch.empty((0, full_edge_attr.size(1)), dtype=torch.float)
    # Return the new inter-cluster edge_index and edge_attr
    return cluster_edge_index, cluster_edge_attr

# --- Dedicated Cluster-GCN Layer ---
class ClusterGCNLayer(nn.Module):
    """
    Applies GCN convolution on nodes within each cluster.
    """
    def __init__(self, in_channels, out_channels, cluster_assignment, full_edge_index, full_edge_attr):
        #calling the initializer of the parent class
        super(ClusterGCNLayer, self).__init__()
        # Graph convolution layer for one cluster: maps in_channels→out_channels
        self.conv = GCNConv(in_channels, out_channels)  
        # Register the cluster assignment vector as a non-learnable buffer
        self.register_buffer('cluster_assignment', cluster_assignment)  
        # Register the global edge index tensor (shape [2, E]) as a buffer
        self.register_buffer('full_edge_index',    full_edge_index)  
        # Register the global edge attributes tensor (shape [E, F]) as a buffer
        self.register_buffer('full_edge_attr',     full_edge_attr)  
    def forward(self, X):
        # Clone the input features so we can update cluster-wise without overwriting X in-place
        X_updated = X.clone()  
        # Find all unique cluster IDs present in this batch
        unique_clusters = torch.unique(self.cluster_assignment)
        # Loop over each cluster ID
        for c in unique_clusters:
            # Create a boolean mask selecting nodes in cluster c
            mask = (self.cluster_assignment == c)
            # Skip empty clusters
            if mask.sum() == 0:
                continue
            # Get the global node indices belonging to this cluster
            node_indices = mask.nonzero(as_tuple=False).view(-1)
            # Build a mask for edges where both endpoints lie in this cluster
            edge_mask = mask[self.full_edge_index[0]] & mask[self.full_edge_index[1]]
            # If there are no intra-cluster edges, skip
            if edge_mask.sum() == 0:
                continue
            # Extract the subgraph’s edge_index for this cluster
            sub_edge_index = self.full_edge_index[:, edge_mask]
            # Gather the feature vectors for nodes in this cluster
            X_cluster = X[node_indices]
            # Create a mapping from global node ID → local cluster index (0…Nc−1)
            mapping = {int(idx): i for i, idx in enumerate(node_indices)}
            
            # Remap the global edge indices to local indices for GCNConv
            local_edge_index = sub_edge_index.clone()
            local_edge_index[0] = torch.tensor([mapping[int(idx)] for idx in sub_edge_index[0]],
                                                 dtype=torch.long, device=local_edge_index.device)
            local_edge_index[1] = torch.tensor([mapping[int(idx)] for idx in sub_edge_index[1]],
                                                 dtype=torch.long, device=local_edge_index.device)
            # Apply GCNConv to the cluster’s features and its local edge index
            X_cluster_updated = self.conv(X_cluster, local_edge_index)
            # Write the updated features back into the corresponding rows of X_updated
            X_updated[node_indices] = X_cluster_updated
        
        # Return the feature matrix where each cluster has been convolved independently
        return X_updated

# --- Cluster Pooling Layer ---
from torch_geometric.nn import global_mean_pool, global_max_pool
class ClusterPoolingLayer(nn.Module):
    """
    Aggregates node features into cluster‐level representations
    using either mean or max pooling.
    """
    def __init__(self, pool_type='mean'):
        super(ClusterPoolingLayer, self).__init__()
        # Store which pooling method to use: 'mean' for average, 'max' for maximum
        self.pool_type = pool_type
    def forward(self, X, cluster_assignment):
        """
        X:              Tensor of shape [N, F] with node features
        cluster_assignment: Tensor of shape [N] with cluster IDs for each node
        """
        
        if self.pool_type == 'mean':
            # Compute the mean of features X for each cluster ID
            pooled = global_mean_pool(X, cluster_assignment)
        elif self.pool_type == 'max':
            # Compute the maximum of features X for each cluster ID
            pooled = global_max_pool(X, cluster_assignment)
        else:
            # Raise an error if an unsupported pooling type is specified
            raise ValueError("Unsupported pool type")
        # Return a tensor of shape [num_clusters, F] with aggregated cluster features
        return pooled

# --- Global GAT Over Clusters ---
class GlobalGATOverClusters(nn.Module):
    """
    Applies a GATConv layer over the cluster-level graph,
    allowing attention over inter-cluster edges with attributes.
    """
    def __init__(self, in_channels, out_channels, num_heads, edge_dim):
        super(GlobalGATOverClusters, self).__init__()
        # Initialize a GATConv that:
        # - maps from in_channels → out_channels
        # - uses num_heads attention heads
        # - concatenation of heads disabled (outputs averaged)
        # - takes edge attributes of dimension edge_dim
        self.gat = GATConv(in_channels, out_channels, heads=num_heads, concat=False, edge_dim=edge_dim)
    def forward(self, cluster_embeddings, cluster_edge_index, cluster_edge_attr):
        """
        cluster_embeddings: Tensor [C, in_channels] of cluster-level features
        cluster_edge_index: Tensor [2, E'] listing edges between clusters
        cluster_edge_attr:  Tensor [E', edge_dim] with attributes per inter-cluster edge
        """
        # Apply the GATConv: attention incorporates both node features and edge_attr
        # returns Tensor [C, out_channels]
        return self.gat(cluster_embeddings, cluster_edge_index, edge_attr=cluster_edge_attr)

# --- Updated EnrichedGATModule with Hierarchical Processing ---
class EnrichedGATModule(nn.Module):
    """
    Full pipeline combining user/item embeddings, extra features,
    Cluster-GCN, cluster pooling, global GAT over clusters,
    and further GAT layers on the original graph.
    """
    def __init__(self, n_users, n_items, embedding_dim, extra_dim, num_heads, num_layers, edge_dim=2, dropout=0.2,
                 cluster_assignment=None, full_edge_index=None, full_edge_attr=None):
        super(EnrichedGATModule, self).__init__()# Initialize nn.Module internals
        # Save counts for splitting outputs later
        self.n_users = n_users
        self.n_items = n_items
        # Embedding tables for users and items
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        # Fuse item embedding + extra features into final item vectors
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim + extra_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embedding_dim, embedding_dim)
        )
        # Store the full graph structure as buffers (non-learnable)
        self.register_buffer('full_edge_index', full_edge_index)# [2, E]
        self.register_buffer('full_edge_attr', full_edge_attr)# [E, edge_dim]
        self.register_buffer('cluster_assignment', cluster_assignment)# [num_nodes]
        #self.cluster_assignment = cluster_assignment  # Tensor of shape (num_nodes,)
        
        # If cluster info is provided, create a dedicated Cluster-GCN layer
        if cluster_assignment is not None and full_edge_index is not None:
            self.cluster_gcn = ClusterGCNLayer(embedding_dim, embedding_dim, cluster_assignment,
                                               full_edge_index, full_edge_attr)
        else:
            self.cluster_gcn = None # skip local cluster updates if missing
        
        # First GAT over the cluster-level graph
        self.global_gat = GlobalGATOverClusters(in_channels=embedding_dim, out_channels=embedding_dim,
                                                num_heads=4, edge_dim=edge_dim)
        # Additional GATConv layers on the original full graph
        self.gat_layers = nn.ModuleList([
            GATConv(embedding_dim, embedding_dim, heads=num_heads, concat=False, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])
        # GraphNorms to stabilize each GAT layer’s outputs
        self.graph_norms = nn.ModuleList([GraphNorm(embedding_dim) for _ in range(num_layers)])
        # Final dropout after each GAT+skip connection
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, product_extra_features):
        ## 1. Build initial node feature matrix X
        user_embeds = self.user_embeddings.weight # [n_users, D]
        item_embeds = self.item_embeddings.weight # [n_items, D]
        # Fuse item embeddings with external features
        enriched_item_embeds = self.fusion_layer(torch.cat([item_embeds, product_extra_features], dim=-1))# [n_items, D+extra_dim]
        X = torch.cat([user_embeds, enriched_item_embeds], dim=0)# [n_users+n_items, D]
        
        # 2. Cluster-level update via dedicated Cluster-GCN.
        if self.cluster_gcn is not None:
            X_cluster_updated = self.cluster_gcn(X)# apply ClusterGCNLayer
        else:
            X_cluster_updated = X # no change if no cluster info
        
        # 3. Pool node features within clusters.
        pool_layer = ClusterPoolingLayer(pool_type='mean')
        cluster_embeddings = pool_layer(X_cluster_updated, self.cluster_assignment)
        # cluster_embeddings: [num_clusters, D]
        
        # 4. Construct the inter-cluster graph on-the-fly.
        cluster_edge_index, cluster_edge_attr = build_cluster_graph(self.full_edge_index.cpu(),
                                                                      self.full_edge_attr.cpu(),
                                                                      self.cluster_assignment.cpu())
        # Move back to device
        cluster_edge_index = cluster_edge_index.to(X.device)
        cluster_edge_attr = cluster_edge_attr.to(X.device)
        
        # 5. Apply GAT over the cluster graph
        updated_cluster_embeddings = self.global_gat(cluster_embeddings, cluster_edge_index, cluster_edge_attr)# [num_clusters, D]
        
        # 6. Broadcast updated cluster features back to each node
        updated_cluster_for_nodes = updated_cluster_embeddings[self.cluster_assignment.to(X.device)]# [n_users+n_items, D]
        X_combined = X_cluster_updated + updated_cluster_for_nodes
        
        # 7. Further global GAT layers on the original full graph
        full_edge_index_device = self.full_edge_index.to(X_combined.device)
        full_edge_attr_device = self.full_edge_attr.to(X_combined.device)
        for gat, norm in zip(self.gat_layers, self.graph_norms):
            X_in = X_combined# for skip connection
            X_combined = gat(X_combined, full_edge_index_device, edge_attr=full_edge_attr_device)
            X_combined = norm(X_combined)# normalize
            X_combined = F.elu(X_combined + X_in)# skip + activation
            X_combined = self.dropout(X_combined)# dropout
        
        # 8. Split back into user and item embeddings.
        user_embeds_final, item_embeds_final = torch.split(X_combined, [self.n_users, self.n_items], dim=0)
        return user_embeds_final, item_embeds_final

# --- Attention Module (unchanged) ---
class Attention(nn.Module):
    """
    A simple additive attention mechanism that computes attention scores
    between a target embedding and each embedding in a sequence,
    then produces a weighted sum of the sequence embeddings.
    """
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        # Linear layer to transform the target embedding into a "query" vector
        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        # Linear layer to transform each sequence embedding into a "key" vector
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        # Linear layer to project the combined query+key to a single attention score
        self.score_layer = nn.Linear(embedding_dim, 1)
    def forward(self, sequence_embeds, target_embed):
        """
        Args:
            sequence_embeds: Tensor of shape [B, L, D], where
                B = batch size, L = sequence length, D = embedding_dim
            target_embed:    Tensor of shape [B, D], the "query" embedding

        Returns:
            weighted_sequence:   Tensor [B, D], the attention-weighted sum
            attention_weights:   Tensor [B, L], the softmax scores over sequence
        """
        # 1) Compute query vectors: [B, D] → [B, 1, D] for broadcasting
        queries = self.query_layer(target_embed).unsqueeze(1)
        # 2) Compute key vectors for each sequence element: [B, L, D]
        keys = self.key_layer(sequence_embeds)
        # 3) Compute unnormalized scores:
        #    - Elementwise add query to each key: broadcasts to [B, L, D]
        #    - Apply nonlinearity (tanh)
        #    - Project to a single value per position: [B, L, 1] → squeeze to [B, L]
        scores = self.score_layer(torch.tanh(queries + keys)).squeeze(-1)
         # 4) Normalize scores with softmax over the sequence length dimension
        attention_weights = F.softmax(scores, dim=-1)
        # 5) Weight the original sequence embeddings by these weights
        #    - Expand weights to [B, L, 1] and multiply elementwise with sequence_embeds
        #    - Sum across the sequence dimension to get [B, D]
        weighted_sequence = torch.sum(sequence_embeds * attention_weights.unsqueeze(-1), dim=1)
        # 6) Return the aggregated vector and the attention map
        return weighted_sequence, attention_weights

# --- CollaborativeGraphLSTM Module ---
class CollaborativeGraphLSTM(nn.Module):
    """
    Combines a graph-based embedding module with an LSTM + attention sequence model
    to predict user–item interactions.
    """
    def __init__(self, n_users, n_movies, embedding_dim, extra_dim, num_heads, num_layers,
                 lstm_hidden_dim, edge_dim=2, dropout=0.2, cluster_assignment=None,
                 full_edge_index=None, full_edge_attr=None):
        super(CollaborativeGraphLSTM, self).__init__()
        # Graph module that produces user/item embeddings (with enriched GAT + clustering)
        self.graph_module = EnrichedGATModule(n_users, n_movies, embedding_dim, extra_dim, num_heads,
                                              num_layers, edge_dim=edge_dim, dropout=dropout,
                                              cluster_assignment=cluster_assignment,
                                              full_edge_index=full_edge_index,
                                              full_edge_attr=full_edge_attr)
        # LSTM to model sequential dynamics over the weighted sequence embeddings
        self.lstm = nn.LSTM(
            input_size=embedding_dim,       # dimensionality of attention output
            hidden_size=lstm_hidden_dim,    # size of LSTM hidden state
            batch_first=True,               # batch at dim0, sequence at dim1
            num_layers=2,                   # two stacked LSTM layers
            dropout=dropout                 # dropout between LSTM layers
        )
        # LayerNorm to stabilize the LSTM’s final hidden state
        self.ln_lstm = nn.LayerNorm(lstm_hidden_dim)
        # Attention mechanism to weight sequence elements by relevance to target item
        self.attention = Attention(embedding_dim)
        # Final feedforward layers to produce a scalar score from combined features
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim + 2 * embedding_dim, 128),  # combine LSTM+user+item dims
            nn.ReLU(),                                            # non-linearity
            nn.Linear(128, 1)                                     # output one score per example
        )
    def forward(self, user_ids, item_sequences, target_items, product_extra_features):
        """
        Full forward pass computing fresh embeddings from the graph module.
        """
        # 1) Run graph module to get updated user/item embeddings
        user_embeds_all, item_embeds_all = self.graph_module(product_extra_features)
        # 2) Delegate to common embedding-based forward routine
        return self._forward_from_embeddings(user_ids, item_sequences, item_embeds_all, user_embeds_all, target_items)
    def forward_cached(self, user_ids, item_sequences, cached_user_embeds, cached_item_embeds, target_items):
        """
        Forward pass using precomputed embeddings (to save compute during training).
        """
        # Reuse the same downstream logic but with provided embeddings
        return self._forward_from_embeddings(user_ids, item_sequences, cached_item_embeds, cached_user_embeds, target_items)
    def _forward_from_embeddings(self, user_ids, item_sequences, item_embeds_all, user_embeds_all, target_items):
        """
        Core routine that handles both single-target and multi-target scoring.
        """
        # Lookup per-user embedding for each user in the batch
        user_embeds = user_embeds_all[user_ids] # [B, D]
         # Gather sequence embeddings: [B, L] → [B, L, D]
        item_seq_embeds = item_embeds_all[item_sequences]
        # Case 1: single target per example (target_items is [B])
        if target_items.dim() == 1:
            # Lookup the target item embedding: [B]
            target_item_embeds = item_embeds_all[target_items]              # [B, D]
            # Attention over the sequence conditioned on the target
            weighted_seq, att_weights = self.attention(item_seq_embeds, target_item_embeds)
            # Prepare LSTM input: add sequence dimension (L=1)
            lstm_input = weighted_seq.unsqueeze(1)                          # [B, 1, D]
            # Run through LSTM: returns (output, (h_n, c_n))
            lstm_output, _ = self.lstm(lstm_input)                          
            # Take the last time step’s output and normalize
            lstm_last = lstm_output[:, -1, :]                               # [B, H]
            lstm_last = self.ln_lstm(lstm_last)                             
            # Concatenate LSTM, user, and target-item embeddings
            combined = torch.cat([lstm_last, user_embeds, target_item_embeds], dim=-1)
            # Produce final score and squeeze to [B]
            scores = self.fc(combined).squeeze(-1)
            return scores, att_weights
        # Case 2: multiple candidates per user (target_items is [B, C])
        else:
            B, C = target_items.shape
            # Flatten candidates: [B, C] → [B*C]
            target_items_flat = target_items.reshape(-1)
            # Lookup flattened target embeddings: [B*C, D]
            target_item_embeds = item_embeds_all[target_items_flat]

            # Expand sequence embeddings to match flattened batch
            _, seq_len, emb_dim = item_seq_embeds.shape

            # [B, 1, L, D]-># [B*C, L, D]
            expanded_item_seq_embeds = item_seq_embeds.unsqueeze(1).expand(B, C, seq_len, emb_dim).reshape(B * C, seq_len, emb_dim)
            # Similarly expand user embeddings: [B, D] → [B*C, D]
            expanded_user_embeds = user_embeds.unsqueeze(1).expand(B, C, emb_dim).reshape(B * C, emb_dim)
            # Apply attention & LSTM as before on flattened batch
            weighted_seq, att_weights = self.attention(expanded_item_seq_embeds, target_item_embeds)
            lstm_input = weighted_seq.unsqueeze(1)
            lstm_output, _ = self.lstm(lstm_input)
            lstm_last = lstm_output[:, -1, :]
            lstm_last = self.ln_lstm(lstm_last)
            # Concatenate and score
            combined = torch.cat([lstm_last, expanded_user_embeds, target_item_embeds], dim=-1)
            scores = self.fc(combined).squeeze(-1)
            # Reshape back to [B, C]
            scores = scores.view(B, C)
            return scores, att_weights

#############################################
# 10. Training & Evaluation Functions (unchanged)
#############################################
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=0.001,
    product_extra_features=None,
    loss_type="BPR",
    margin=1.0
):
    # 1. Set up optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 2. Learning rate scheduler that anneals cosine-ly over `epochs`
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # 3. Binary cross-entropy loss for BCE mode
    loss_fn = nn.BCEWithLogitsLoss()

    # 4. Define BPR loss function
    def bpr_loss(pos_score, neg_score):
        return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))

    # 5. Mixed-precision scaler for faster training on GPU
    scaler = GradScaler()

    # 6. Main training loop over epochs
    for epoch in range(epochs):
        model.train()  # put model into training mode
        total_train_loss = 0.0
        correct_train_preds = 0
        total_train_samples = 0

        # 7. If using GAT caching, precompute graph embeddings on CPU
        if USE_GAT_CACHING:
            with torch.no_grad():
                cpu_model = model.cpu()
                cpu_feats = product_extra_features.cpu()
                u_cpu, i_cpu = cpu_model.graph_module(cpu_feats)
                cached_user_embeds = u_cpu.to(device)
                cached_item_embeds = i_cpu.to(device)
                model.to(device)

        # 8. Iterate over training batches
        for batch_idx, (user_ids, input_seqs, pos_items, neg_items, pos_labels) in enumerate(train_loader):
            # 8a. Move batch data to device
            user_ids = user_ids.to(device)
            input_seqs = input_seqs.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            pos_labels = pos_labels.to(device)

            # 8b. Zero out previous gradients
            optimizer.zero_grad(set_to_none=True)

            # 8c. Forward/backward pass under autocast for mixed precision
            with (autocast() if device.type == "cuda" else nullcontext()):
                if loss_type == "BCE":
                    # 8c.i. Compute positive and negative predictions
                    if USE_GAT_CACHING:
                        pos_preds, _ = model.forward_cached(
                            user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items
                        )
                        neg_preds_all, _ = model.forward_cached(
                            user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items
                        )
                    else:
                        pos_preds, _ = model(
                            user_ids, input_seqs, pos_items, product_extra_features
                        )
                        neg_preds_all, _ = model(
                            user_ids, input_seqs, neg_items, product_extra_features
                        )
                    # 8c.ii. For BCE, take the hardest negative per example
                    neg_preds, _ = torch.max(neg_preds_all, dim=1)
                    # 8c.iii. Compute BCE loss on positives and negatives
                    loss_pos = loss_fn(pos_preds, pos_labels)
                    neg_labels = torch.zeros_like(neg_preds)
                    loss_neg = loss_fn(neg_preds, neg_labels)
                    loss = loss_pos + loss_neg

                elif loss_type == "BPR":
                    # 8c.iv. Compute predictions same as above
                    if USE_GAT_CACHING:
                        pos_preds, _ = model.forward_cached(
                            user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items
                        )
                        neg_preds_all, _ = model.forward_cached(
                            user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items
                        )
                    else:
                        pos_preds, _ = model(
                            user_ids, input_seqs, pos_items, product_extra_features
                        )
                        neg_preds_all, _ = model(
                            user_ids, input_seqs, neg_items, product_extra_features
                        )
                    # 8c.v. Take hardest negative and compute BPR loss
                    neg_preds, _ = torch.max(neg_preds_all, dim=1)
                    loss = bpr_loss(pos_preds, neg_preds)

            # 8d. Backpropagate with mixed-precision scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            # 8e. Track accuracy if using BCE
            if loss_type == "BCE":
                pos_pred_binary = (pos_preds > 0).float()
                neg_pred_binary = (neg_preds > 0).float()
                correct_train_preds += (pos_pred_binary == pos_labels).sum().item()
                correct_train_preds += (neg_pred_binary == torch.zeros_like(neg_preds)).sum().item()
                total_train_samples += pos_labels.size(0) * 2

            # 8f. Accumulate training loss (scaled by batch size)
            total_train_loss += loss.item() * user_ids.size(0)

        # 9. Compute average training loss and accuracy
        if loss_type == "BCE":
            train_loss_avg = total_train_loss / total_train_samples
            train_acc = correct_train_preds / total_train_samples
        else:
            train_loss_avg = total_train_loss / len(train_loader.dataset)
            train_acc = 0.0

        # 10. Validation phase
        model.eval()
        total_val_loss = 0.0
        correct_val_preds = 0
        total_val_samples = 0

        # 10a. Recompute cached embeddings if needed
        if USE_GAT_CACHING:
            with torch.no_grad():
                cached_user_embeds, cached_item_embeds = model.graph_module(product_extra_features)

        # 10b. Iterate over validation batches
        with torch.no_grad():
            for user_ids, input_seqs, pos_items, neg_items, pos_labels in val_loader:
                # 10b.i. Move to device
                user_ids = user_ids.to(device)
                input_seqs = input_seqs.to(device)
                pos_items = pos_items.to(device)
                neg_items = neg_items.to(device)
                pos_labels = pos_labels.to(device)

                with (autocast() if device.type == "cuda" else nullcontext()):
                    if loss_type == "BCE":
                        # 10b.ii. Compute predictions
                        if USE_GAT_CACHING:
                            pos_preds, _ = model.forward_cached(
                                user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items
                            )
                            neg_preds_all, _ = model.forward_cached(
                                user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items
                            )
                        else:
                            pos_preds, _ = model(
                                user_ids, input_seqs, pos_items, product_extra_features
                            )
                            neg_preds_all, _ = model(
                                user_ids, input_seqs, neg_items, product_extra_features
                            )
                        neg_preds, _ = torch.max(neg_preds_all, dim=1)
                        # 10b.iii. Compute BCE validation loss
                        loss_pos = loss_fn(pos_preds, pos_labels)
                        neg_labels = torch.zeros_like(neg_preds)
                        loss_neg = loss_fn(neg_preds, neg_labels)
                        loss = loss_pos + loss_neg
                        # 10b.iv. Count correct predictions
                        pos_pred_binary = (pos_preds > 0).float()
                        neg_pred_binary = (neg_preds > 0).float()
                        correct_val_preds += (pos_pred_binary == pos_labels).sum().item()
                        correct_val_preds += (neg_pred_binary == torch.zeros_like(neg_preds)).sum().item()
                        total_val_samples += pos_labels.size(0) * 2

                    elif loss_type == "BPR":
                        # 10b.v. Compute BPR validation loss
                        if USE_GAT_CACHING:
                            pos_preds, _ = model.forward_cached(
                                user_ids, input_seqs, cached_user_embeds, cached_item_embeds, pos_items
                            )
                            neg_preds_all, _ = model.forward_cached(
                                user_ids, input_seqs, cached_user_embeds, cached_item_embeds, neg_items
                            )
                        else:
                            pos_preds, _ = model(
                                user_ids, input_seqs, pos_items, product_extra_features
                            )
                            neg_preds_all, _ = model(
                                user_ids, input_seqs, neg_items, product_extra_features
                            )
                        neg_preds, _ = torch.max(neg_preds_all, dim=1)
                        loss = bpr_loss(pos_preds, neg_preds)

                # 10b.vi. Accumulate validation loss
                total_val_loss += loss.item() * user_ids.size(0)

        # 11. Compute average validation loss and accuracy
        if loss_type == "BCE":
            val_loss_avg = total_val_loss / total_val_samples
            val_acc = correct_val_preds / total_val_samples
        else:
            val_loss_avg = total_val_loss / len(val_loader.dataset)
            val_acc = 0.0

        # 12. Log epoch metrics
        print(
            f"Epoch {epoch+1}: "
            f"Train Loss {train_loss_avg:.4f}, Train Acc {train_acc:.4f}, "
            f"Val Loss {val_loss_avg:.4f}, Val Acc {val_acc:.4f}"
        )

        # 13. Step the scheduler and clear cache
        scheduler.step()
        torch.cuda.empty_cache()


def evaluate_model_batch(model, test_loader, n_items, k=10, chunk_size=1000, product_extra_features=None):
    """
    Evaluate the model on the test dataset in batches, computing Precision@k, Recall@k, and NDCG@k.
    """
    # Put model in evaluation mode (disables dropout, etc.)
    model.eval()
    # Initialize accumulators for metrics
    total_precision, total_recall, total_ndcg = 0.0, 0.0, 0.0
    total_samples = 0

    # Base tensor of all item indices [0, 1, ..., n_items-1] on the correct device
    base_all_items = torch.arange(n_items, dtype=torch.long, device=device)

    # If caching GAT outputs, compute them once outside the loop
    if USE_GAT_CACHING:
        with torch.no_grad():
            cached_user_embeds, cached_item_embeds = model.graph_module(product_extra_features)

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches from the test DataLoader
        for user_ids, input_seqs, pos_items in test_loader:
            B = user_ids.size(0)  # batch size

            # Move batch data to the target device
            user_ids = user_ids.to(device)
            input_seqs = input_seqs.to(device)
            pos_items = pos_items.to(device)

            scores_chunks = []  # to store scores for each chunk of items

            # Process items in chunks to avoid memory overflow
            for start in range(0, n_items, chunk_size):
                end = min(start + chunk_size, n_items)
                # Create a [B, chunk_size] tensor of candidate item IDs
                candidate_chunk = base_all_items[start:end].unsqueeze(0).expand(B, end - start)

                # Compute scores for this chunk, using cached embeddings if available
                if USE_GAT_CACHING:
                    chunk_scores, _ = model.forward_cached(
                        user_ids, input_seqs, cached_user_embeds, cached_item_embeds, candidate_chunk
                    )
                else:
                    chunk_scores, _ = model(
                        user_ids, input_seqs, candidate_chunk, product_extra_features
                    )

                scores_chunks.append(chunk_scores)  # save the chunk’s scores

            # Concatenate all chunk scores to get [B, n_items]
            scores_all = torch.cat(scores_chunks, dim=1)

            # Select top-k item indices for each user in the batch
            _, topk_indices = torch.topk(scores_all, k, dim=1)

            # Compute metrics for each example in the batch
            for i in range(B):
                predicted_items = topk_indices[i].tolist()    # list of top-k predictions
                true_item = pos_items[i].item()               # the ground-truth next item

                # Precision@k: fraction of correct predictions (either 1/k or 0)
                prec = 1.0 / k if true_item in predicted_items else 0.0
                # Recall@k: 1 if true item is among predictions, else 0
                rec = 1.0 if true_item in predicted_items else 0.0

                # NDCG@k: discounted gain if true item is in predictions
                ndcg = 0.0
                if true_item in predicted_items:
                    pos_idx = predicted_items.index(true_item)  # position within top-k
                    # Compute 1 / log2(rank+1), where rank=pos_idx+1
                    ndcg = 1.0 / torch.log2(torch.tensor(pos_idx + 2, dtype=torch.float)).item()

                # Accumulate metrics
                total_precision += prec
                total_recall += rec
                total_ndcg += ndcg

            # Update count of processed samples
            total_samples += B

    # Compute average metrics over all test samples
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

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_train, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_train, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_test, num_workers=2, pin_memory=True)

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
