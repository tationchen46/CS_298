#!/usr/bin/env python3
# coding: utf-8

# -----------------------------------------------------------------------------
# 0) ENVIRONMENT & DEVICE SETUP
#    - configure CUDA & tokenizer settings
#    - initialize NLTK VADER and select GPU/CPU device
# -----------------------------------------------------------------------------
import os
import json
import random
import re
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.data import Data
from torch_geometric.loader import ClusterData
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.nn.norm import GraphNorm

# disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# limit CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TMPDIR"] = "/home/tianxiangchen/tmp"

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------------------------------------------------
# 1) FILE PATHS
# -----------------------------------------------------------------------------
TRAIN_FILE         = os.path.expanduser("~/train.txt")
TEST_FILE          = os.path.expanduser("~/test.txt")
USER_LIST_FILE     = os.path.expanduser("~/user_list.txt")
ITEM_LIST_FILE     = os.path.expanduser("~/item_list.txt")
EXTRACTED_DATASET  = os.path.expanduser("~/extracted_dataset.jsonl")
EXTRACTED_DATASET1 = os.path.expanduser("~/extracted_dataset1.jsonl")

# -----------------------------------------------------------------------------
# 2) UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_mapping(fp):
    m = {}
    with open(fp, 'r') as f:
        for i, ln in enumerate(f):
            parts = ln.strip().split()
            if i == 0 and parts[0].lower() == "org_id":
                continue
            if len(parts) >= 2:
                m[parts[0]] = int(parts[1])
    return m

def parse_interactions(fp):
    seqs = []
    with open(fp, 'r') as f:
        for ln in f:
            parts = list(map(int, ln.strip().split()))
            if len(parts) < 2:
                continue
            seqs.append((parts[0], parts[1:]))
    return seqs

def get_sinusoid_encoding_table(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float()
                    * -(math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

# -----------------------------------------------------------------------------
# 3) LOAD ID MAPPINGS
# -----------------------------------------------------------------------------
user_map = load_mapping(USER_LIST_FILE)
item_map = load_mapping(ITEM_LIST_FILE)

# -----------------------------------------------------------------------------
# 4) REVIEW EMBEDDINGS & EDGE SENTIMENTS
# -----------------------------------------------------------------------------
sent_model = SentenceTransformer('all-MiniLM-L6-v2')

review_embeddings = {}
edge_sentiments   = {}
with open(EXTRACTED_DATASET, 'r') as f:
    for ln in f:
        rec = json.loads(ln)
        u0, i0 = rec.get('user_id'), rec.get('parent_asin')
        if u0 not in user_map or i0 not in item_map:
            continue
        u, i = user_map[u0], item_map[i0]
        text = clean_text(rec.get('text', ""))
        if len(text.split()) < 10:
            continue
        emb = sent_model.encode(text)
        review_embeddings.setdefault(i, []).append(emb)
        comp = sia.polarity_scores(text)['compound']
        pol  = (comp + 1) / 2.0
        r    = rec.get('rating', None)
        if r is not None:
            norm_r = (float(r)-1)/4.0
            sent   = (pol + norm_r)/2.0
        else:
            sent = pol
        edge_sentiments.setdefault((u,i), []).append(sent)

# -----------------------------------------------------------------------------
# 5) METADATA EMBEDDINGS & ITEM CATEGORIES
# -----------------------------------------------------------------------------
metadata_embeddings = {}
item_category      = {}
with open(EXTRACTED_DATASET1, 'r') as f:
    for ln in f:
        rec = json.loads(ln)
        i0 = rec.get('parent_asin')
        if i0 not in item_map:
            continue
        i = item_map[i0]
        title = clean_text(rec.get('title', ""))
        cat   = clean_text(rec.get('main_category', ""))
        if cat:
            item_category[i] = cat
        et = sent_model.encode(title) if title else np.zeros(384)
        ec = sent_model.encode(cat)   if cat   else np.zeros(384)
        metadata_embeddings[i] = np.concatenate([et, ec])

# -----------------------------------------------------------------------------
# 6) PRODUCT EXTRA FEATURES
# -----------------------------------------------------------------------------
extra_dim_review   = 384
extra_dim_metadata = 768
def get_product_feature(i):
    revs = review_embeddings.get(i)
    rev_feat = np.mean(revs, axis=0) if revs else np.zeros(extra_dim_review)
    meta_feat = metadata_embeddings.get(i, np.zeros(extra_dim_metadata))
    return np.concatenate([rev_feat, meta_feat])
n_items_from_map = max(item_map.values())+1 if item_map else 0
pf_np = np.zeros((n_items_from_map, extra_dim_review+extra_dim_metadata), dtype=np.float32)
for pid in range(n_items_from_map):
    pf_np[pid] = get_product_feature(pid)
product_extra_features = torch.tensor(pf_np, dtype=torch.float).to(device)

# -----------------------------------------------------------------------------
# 7) INTERACTIONS → USER–ITEM EDGES
# -----------------------------------------------------------------------------
train_inter = parse_interactions(TRAIN_FILE)
test_inter  = parse_interactions(TEST_FILE)
n_users = max(u for u,_ in train_inter) + 1
n_items = max(max(seq) for _,seq in train_inter) + 1
print(f"n_users={n_users}, n_items={n_items}")

def create_edge_index_and_attr(seqs, n_users):
    edges, attrs = [], []
    for u, seq in seqs:
        L = len(seq)
        for idx, i in enumerate(seq):
            edges.append([u, n_users+i])
            sent  = np.mean(edge_sentiments.get((u,i), [0.0]))
            order = idx/(L-1) if L>1 else 0.0
            attrs.append([sent, order])
            edges.append([n_users+i, u])
            attrs.append([0.0, 0.0])
    return torch.tensor(edges).t().contiguous(), torch.tensor(attrs, dtype=torch.float)

base_ei, base_ea = create_edge_index_and_attr(train_inter, n_users)

# -----------------------------------------------------------------------------
# 8) ADD ITEM–ITEM EDGES
# -----------------------------------------------------------------------------
def add_item_item_edges(ei, ea, n_users, cat2items, meta_embs, thresh=0.5):
    new_e, new_a = [], []
    for cat, items in cat2items.items():
        if len(items) < 2: continue
        sim_mat = cosine_similarity([meta_embs[i] for i in items])
        for x in range(len(items)):
            for y in range(x+1, len(items)):
                sim = sim_mat[x,y]
                if sim >= thresh:
                    u = n_users+items[x]; v = n_users+items[y]
                    new_e += [[u,v],[v,u]]
                    new_a += [[sim,0.0],[sim,0.0]]
    if new_e:
        ne = torch.tensor(new_e).t().contiguous()
        na = torch.tensor(new_a, dtype=torch.float)
        return torch.cat([ei, ne], dim=1), torch.cat([ea, na], dim=0)
    return ei, ea

graph_ei, graph_ea = add_item_item_edges(base_ei, base_ea, n_users,
                                         item_category, metadata_embeddings)
graph_ei, graph_ea = graph_ei.to(device), graph_ea.to(device)

# -----------------------------------------------------------------------------
# 9) CLUSTER ONLY USER–ITEM GRAPH
# -----------------------------------------------------------------------------
n_total = n_users + n_items
ui_data = Data(edge_index=base_ei.to(device),
               edge_attr=base_ea.to(device),
               num_nodes=n_total)
ui_data.n_id = torch.arange(n_total, device=device).unsqueeze(1)
cluster_data = ClusterData(ui_data, num_parts=50, recursive=False)
clusters = list(cluster_data)
cluster_assignment = torch.empty(n_total, dtype=torch.long, device=device)
for idx, c in enumerate(clusters):
    cluster_assignment[c.n_id] = idx

# -----------------------------------------------------------------------------
# 10) DATASETS & DATALOADERS
# -----------------------------------------------------------------------------
class TrainDataset(Dataset):
    def __init__(self, seqs, seq_len, n_items, neg=10):
        self.samples, self.user_pos = [], {}
        self.n_items, self.neg = n_items, neg
        for u, seq in seqs:
            self.user_pos[u] = set(seq)
            for i in range(len(seq)-seq_len):
                self.samples.append((u, seq[i:i+seq_len], seq[i+seq_len]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        u, inp, pos = self.samples[idx]
        negs = []
        for _ in range(self.neg):
            ni = random.randint(0, self.n_items-1)
            while ni in self.user_pos[u]:
                ni = random.randint(0, self.n_items-1)
            negs.append(ni)
        return u, inp, pos, negs

class TestDataset(Dataset):
    def __init__(self, seqs, seq_len):
        self.samples = [(u, s[:seq_len], s[seq_len])
                        for u,s in seqs if len(s)>seq_len]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_train(batch):
    us, seqs, ps, ns = zip(*batch)
    return (torch.tensor(us), torch.tensor(seqs),
            torch.tensor(ps), torch.tensor(ns),
            torch.ones(len(us), dtype=torch.float))

def collate_test(batch):
    us, seqs, ps = zip(*batch)
    return torch.tensor(us), torch.tensor(seqs), torch.tensor(ps)

# -----------------------------------------------------------------------------
# 11) MODEL COMPONENTS
# -----------------------------------------------------------------------------
class ClusterGCNLayer(nn.Module):
    def __init__(self, in_c, out_c, assign, full_ei):
        super().__init__()
        self.conv    = GCNConv(in_c, out_c)
        self.assign  = assign
        self.full_ei = full_ei
    def forward(self, X):
        X_up = X.clone()
        for c in torch.unique(self.assign):
            mask  = (self.assign == c)
            nodes = mask.nonzero().view(-1)
            emask = mask[self.full_ei[0]] & mask[self.full_ei[1]]
            if emask.sum()==0: continue
            sub = self.full_ei[:, emask]
            mapping = {int(n): i for i,n in enumerate(nodes)}
            li = sub.clone()
            li[0] = torch.tensor([mapping[int(x)] for x in sub[0]], device=li.device)
            li[1] = torch.tensor([mapping[int(x)] for x in sub[1]], device=li.device)
            xc = X[nodes]
            xc_up = self.conv(xc, li)
            X_up[nodes] = xc_up
        return X_up

class ClusterPoolingLayer(nn.Module):
    def forward(self, X, assign):
        return global_mean_pool(X, assign)

class GlobalGATOverClusters(nn.Module):
    def __init__(self, in_c, out_c, heads, edge_dim):
        super().__init__()
        self.gat = GATConv(in_c, out_c, heads=heads, concat=False, edge_dim=edge_dim)
    def forward(self, ce, cei, cea):
        return self.gat(ce, cei, edge_attr=cea)

class EnrichedHierarchicalGraphModule(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, extra_dim,
                 heads, n_layers, edge_dim, drop,
                 assign, full_ei, full_ea):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.fuse     = nn.Sequential(
            nn.Linear(emb_dim+extra_dim, 2*emb_dim),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.assign   = assign
        self.full_ei  = full_ei
        self.full_ea  = full_ea
        self.cluster_gcn  = ClusterGCNLayer(emb_dim, emb_dim, assign, full_ei)
        self.cluster_pool = ClusterPoolingLayer()
        self.global_gat   = GlobalGATOverClusters(emb_dim, emb_dim, heads, edge_dim)
        self.gat_layers   = nn.ModuleList([GATConv(emb_dim, emb_dim, heads=heads, concat=False, edge_dim=edge_dim)
                                           for _ in range(n_layers)])
        self.norms        = nn.ModuleList([GraphNorm(emb_dim) for _ in range(n_layers)])
        self.dropout      = nn.Dropout(drop)

    def build_cluster_graph(self):
        num_c = int(self.assign.max().item()+1)
        edges, attrs = [], []
        for i in range(self.full_ei.size(1)):
            cu = int(self.assign[self.full_ei[0,i]].item())
            cv = int(self.assign[self.full_ei[1,i]].item())
            if cu!=cv:
                edges.append([cu,cv]); attrs.append(self.full_ea[i].tolist())
        if not edges:
            edges = [[i,i] for i in range(num_c)]
            attrs = [[0.0,0.0]]*num_c
        cei = torch.tensor(edges).t().contiguous().to(self.full_ei.device)
        cea = torch.tensor(attrs, dtype=torch.float).to(self.full_ea.device)
        return cei, cea

    def forward(self, extra_feats):
        ue = self.user_emb.weight
        ie = self.item_emb.weight
        fused = self.fuse(torch.cat([ie, extra_feats], dim=-1))
        X = torch.cat([ue, fused], dim=0)
        Xc = self.cluster_gcn(X)
        ce = self.cluster_pool(Xc, self.assign)
        cei, cea = self.build_cluster_graph()
        cu = self.global_gat(ce, cei, cea)
        cu_b = cu[self.assign]
        X_comb = Xc + cu_b
        fei = self.full_ei.to(X_comb.device)
        fea = self.full_ea.to(X_comb.device)
        for gat, nm in zip(self.gat_layers, self.norms):
            Xin = X_comb
            X_comb = gat(X_comb, fei, edge_attr=fea)
            X_comb = nm(X_comb)
            X_comb = F.elu(X_comb + Xin)
            X_comb = self.dropout(X_comb)
        u_fin, i_fin = torch.split(X_comb, [ue.size(0), ie.size(0)], dim=0)
        return u_fin, i_fin

class Attention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.s = nn.Linear(emb_dim, 1)
    def forward(self, seq_emb, tgt_emb):
        Q = self.q(tgt_emb).unsqueeze(1)
        K = self.k(seq_emb)
        scores = self.s(torch.tanh(Q + K)).squeeze(-1)
        W = F.softmax(scores, dim=-1)
        V = torch.sum(seq_emb * W.unsqueeze(-1), dim=1)
        return V, W

class CollaborativeGraphLSTM(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, extra_dim,
                 heads, n_layers, lstm_hid, seq_len,
                 edge_dim=2, drop=0.2,
                 assign=None, full_ei=None, full_ea=None):
        super().__init__()
        self.graph_mod = EnrichedHierarchicalGraphModule(
            n_users, n_items, emb_dim, extra_dim,
            heads, n_layers, edge_dim, drop,
            assign, full_ei, full_ea
        )
        self.lstm    = nn.LSTM(emb_dim, lstm_hid, batch_first=True, num_layers=2, dropout=drop)
        self.ln_lstm = nn.LayerNorm(lstm_hid)
        self.attn    = Attention(emb_dim)
        self.fc      = nn.Sequential(
            nn.Linear(lstm_hid + 2*emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.register_buffer('pos_embedding', get_sinusoid_encoding_table(seq_len, emb_dim, device))

    def forward(self, us, seqs, tgt, extra_feats):
        ue_all, ie_all = self.graph_mod(extra_feats)
        return self._forward(us, seqs, ie_all, ue_all, tgt)

    def forward_cached(self, us, seqs, cached_u, cached_i, tgt):
        return self._forward(us, seqs, cached_i, cached_u, tgt)

    def _forward(self, us, seqs, ie_all, ue_all, tgt):
        B, S = seqs.shape
        ue = ue_all[us]
        seq_emb = ie_all[seqs]
        pos = self.pos_embedding.unsqueeze(0).expand(B, S, -1)
        seq_emb = seq_emb + pos
        if tgt.dim() == 1:
            te, _ = ie_all[tgt], None
            V, W  = self.attn(seq_emb, te)
            L, _  = self.lstm(V.unsqueeze(1))
            L     = self.ln_lstm(L[:, -1, :])
            cat   = torch.cat([L, ue, te], dim=-1)
            return self.fc(cat).squeeze(-1), W
        else:
            B, C = tgt.shape
            flat = tgt.reshape(-1)
            te   = ie_all[flat]
            seq_e= seq_emb.unsqueeze(1).expand(B, C, S, -1).reshape(B*C, S, -1)
            ue_e = ue.unsqueeze(1).expand(B, C, -1).reshape(B*C, -1)
            V, W = self.attn(seq_e, te)
            L, _ = self.lstm(V.unsqueeze(1))
            L    = self.ln_lstm(L[:, -1, :])
            cat  = torch.cat([L, ue_e, te], dim=-1)
            sc   = self.fc(cat).squeeze(-1).view(B, C)
            return sc, W

# -----------------------------------------------------------------------------
# 12) TRAIN / EVAL / PREDICT LOOPS
# -----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, product_extra_features=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bpr_loss  = lambda pos, neg: -torch.mean(torch.log(torch.sigmoid(pos-neg) + 1e-8))
    scaler    = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        # cache GNN outputs once
        with torch.no_grad():
            cached_u, cached_i = model.graph_mod(product_extra_features)
        for us, seqs, pos, negs, _ in train_loader:
            us, seqs, pos, negs = us.to(device), seqs.to(device), pos.to(device), negs.to(device)
            optimizer.zero_grad()
            with autocast():
                pos_sc, _     = model.forward_cached(us, seqs, cached_u, cached_i, pos)
                neg_sc_all, _ = model.forward_cached(us, seqs, cached_u, cached_i, negs)
                neg_sc, _     = torch.max(neg_sc_all, dim=1)
                loss = bpr_loss(pos_sc, neg_sc)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * us.size(0)
        scheduler.step()
        torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}/{epochs} – Loss: {total_loss/len(train_loader.dataset):.4f}")

def evaluate_model_batch(model, test_loader, n_items, k=10, chunk_size=1000, product_extra_features=None):
    model.eval()
    total_prec = total_rec = total_ndcg = 0.0
    total_samples = 0
    base_items = torch.arange(n_items, device=device)
    with torch.no_grad():
        for us, seqs, pos in test_loader:
            B = us.size(0)
            us, seqs, pos = us.to(device), seqs.to(device), pos.to(device)
            scores_chunks = []
            for st in range(0, n_items, chunk_size):
                en = min(st+chunk_size, n_items)
                cands = base_items[st:en].unsqueeze(0).expand(B, en-st)
                sc, _ = model.forward(us, seqs, cands, product_extra_features)
                scores_chunks.append(sc)
            all_sc = torch.cat(scores_chunks, dim=1)
            _, topk = torch.topk(all_sc, k, dim=1)
            for i in range(B):
                preds = topk[i].tolist()
                true = pos[i].item()
                if true in preds:
                    idx = preds.index(true)
                    total_prec += 1.0/k
                    total_rec  += 1.0
                    total_ndcg += 1.0/math.log2(idx+2)
                total_samples += 1
    return total_prec/total_samples, total_rec/total_samples, total_ndcg/total_samples

def predict_new_items_for_user(model, user_id, user_seq, known_set, product_extra_features, k=10):
    model.eval()
    with torch.no_grad():
        base_items = torch.arange(n_items, device=device)
        mask = torch.tensor([i not in known_set for i in range(n_items)], device=device)
        cands = base_items[mask]
        us = torch.tensor([user_id], device=device)
        seq = torch.tensor(user_seq, device=device).unsqueeze(0)
        sc, _ = model.forward(us, seq, cands.unsqueeze(0), product_extra_features)
        vals, idx = torch.topk(sc, k)
        return cands[idx].cpu().tolist(), vals.cpu().tolist()

# -----------------------------------------------------------------------------
# 13) MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    embedding_dim = 128
    extra_dim     = extra_dim_review + extra_dim_metadata
    num_heads     = 4
    num_layers    = 3
    lstm_hidden   = 256
    seq_len       = 10
    dropout_rate  = 0.3
    batch_size    = 64    # reduced to fit memory
    epochs        = 100
    lr            = 1e-3

    train_ds = TrainDataset(train_inter, seq_len, n_items, neg=10)
    val_ds   = TrainDataset(train_inter, seq_len, n_items, neg=10)
    test_ds  = TestDataset(test_inter, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_train, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_train, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_test,  num_workers=2, pin_memory=True)

    model = CollaborativeGraphLSTM(
        n_users, n_items, embedding_dim, extra_dim,
        num_heads, num_layers, lstm_hidden, seq_len,
        edge_dim=2, drop=dropout_rate,
        assign=cluster_assignment,
        full_ei=graph_ei, full_ea=graph_ea
    ).to(device)

    train_model(model, train_loader, val_loader, epochs=epochs,
                lr=lr, product_extra_features=product_extra_features)

    prec, rec, ndcg = evaluate_model_batch(
        model, test_loader, n_items, k=10,
        product_extra_features=product_extra_features
    )
    print(f"Precision@10: {prec:.4f}, Recall@10: {rec:.4f}, NDCG@10: {ndcg:.4f}")

    user_demo = 0
    hist = next(s for u,s in train_inter if u==user_demo and len(s)>=seq_len)
    known = set(hist)
    seq_in = hist[-seq_len:]
    preds, scores = predict_new_items_for_user(
        model, user_demo, seq_in, known,
        product_extra_features, k=10
    )
    print("Predicted new items for user", user_demo, ":", preds)
    print("Scores:", scores)
