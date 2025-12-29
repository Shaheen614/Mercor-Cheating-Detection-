import numpy as np,pandas as pd,torch,networkx as nx

def build_graph(edges): return nx.from_pandas_edgelist(edges,"src","dst")

def index_nodes(G,ids):
    for uid in ids:
        if uid not in G: G.add_node(uid)
    nodes=list(G.nodes()); node_to_idx={n:i for i,n in enumerate(nodes)}
    edges_idx=[[node_to_idx[u],node_to_idx[v]] for u,v in G.edges()]
    edges_idx+=[[node_to_idx[v],node_to_idx[u]] for u,v in G.edges()]
    return nodes,node_to_idx,np.array(edges_idx)

def gcn_embeddings(nodes,edges_idx,d_out=32,epochs=40,seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    n=len(nodes); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A=torch.zeros((n,n)); [A.__setitem__((u,v),1.0) for u,v in edges_idx]; A+=torch.eye(n)
    D=torch.diag(A.sum(1)); D_inv=torch.diag(torch.pow(torch.diag(D),-0.5)); D_inv[torch.isinf(D_inv)]=0
    A_hat=(D_inv@A@D_inv).to(device)
    H0=torch.randn((n,32),device=device); W0=torch.randn((32,64),device=device,requires_grad=True); W1=torch.randn((64,d_out),device=device,requires_grad=True)
    opt=torch.optim.Adam([W0,W1],lr=0.01)
    for _ in range(epochs):
        H1=torch.relu(A_hat@(H0@W0)); H2=A_hat@(H1@W1)
        loss=torch.mean((A_hat@H2-H2)**2)*0.7+torch.mean((H2-H0[:,:d_out])**2)*0.3
        opt.zero_grad(); loss.backward(); opt.step()
    return H2.detach().cpu().numpy()

def compute_graph_features(G,ids):
    deg=dict(G.degree()); bc=nx.betweenness_centrality(G,k=min(1000,len(G)),seed=42); clust=nx.clustering(G)
    df=pd.DataFrame({"user_hash":ids})
    df["deg"]=df["user_hash"].map(deg).fillna(0); df["betweenness"]=df["user_hash"].map(bc).fillna(0); df["clustering"]=df["user_hash"].map(clust).fillna(0)
    return df

def build_graph_augmented_features(edges_df,ids):
    G=build_graph(edges_df); nodes,node_to_idx,edges_idx=index_nodes(G,ids)
    emb=gcn_embeddings(nodes,edges_idx)
    node_to_row={n:i for i,n in enumerate(nodes)}
    E=np.array([emb[node_to_row[i]] for i in ids])
    E_df=pd.DataFrame(E,columns=[f"gnn_emb_{i}" for i in range(E.shape[1])]); E_df.insert(0,"user_hash",ids)
    stats=compute_graph_features(G,ids)
    return stats,E_df
