import numpy as np
import pandas as pd
import networkx as nx

def build_graph(edges_df):
    """
    Build an undirected graph from the social_graph.csv file.
    The file has columns 'user_a' and 'user_b'.
    """
    return nx.from_pandas_edgelist(edges_df, source="user_a", target="user_b")

def compute_graph_features(G, ids):
    deg = dict(G.degree())
    # bc = nx.betweenness_centrality(G, k=min(1000, len(G)), seed=42)  # too slow
    clust = nx.clustering(G)

    df = pd.DataFrame({"user_hash": ids})
    df["deg"] = df["user_hash"].map(deg).fillna(0)
    # df["betweenness"] = df["user_hash"].map(bc).fillna(0)
    df["clustering"] = df["user_hash"].map(clust).fillna(0)
    return df


def build_graph_augmented_features(edges_df, ids):
    """
    Return graph statistics only (safe for Kaggle memory limits).
    Embeddings are skipped here to avoid OOM errors.
    """
    G = build_graph(edges_df)
    stats = compute_graph_features(G, ids)

    # Return stats and an empty DataFrame for embeddings
    emb_df = pd.DataFrame({"user_hash": ids})
    return stats, emb_df
