import numpy as np

COSTS = {"FN":600,"FP_autoblock":300,"FP_review":150,"TP_review":5,"correct":0}

def compute_total_cost(y_true,y_prob,t_review,t_block):
    auto_pass=y_prob<t_review
    review=(y_prob>=t_review)&(y_prob<t_block)
    auto_block=y_prob>=t_block
    tp_review=np.sum(review&(y_true==1))
    fn=np.sum(auto_pass&(y_true==1))
    fp_review=np.sum(review&(y_true==0))
    fp_autoblock=np.sum(auto_block&(y_true==0))
    return fn*COSTS["FN"]+fp_autoblock*COSTS["FP_autoblock"]+fp_review*COSTS["FP_review"]+tp_review*COSTS["TP_review"]

def grid_search_cost(y_true,y_prob,grid=101):
    best={"cost":float("inf"),"t_review":None,"t_block":None}
    ts=np.linspace(0,1,grid)
    for tr in ts:
        for tb in ts:
            if tb<=tr: continue
            cost=compute_total_cost(y_true,y_prob,tr,tb)
            if cost<best["cost"]: best={"cost":cost,"t_review":tr,"t_block":tb}
    return best

def assign_pu_weights(df,label_col="is_cheating",high_clean_col="high_conf_clean"):
    w=np.ones(len(df))
    labeled=df[label_col].notna()
    pos=labeled&(df[label_col]==1)
    neg=labeled&(df[label_col]==0)
    pu=(~labeled)&(df.get(high_clean_col,0)==1)
    unl=(~labeled)&(~pu)
    w[pos]=6.0; w[neg]=1.5; w[pu]=0.5; w[unl]=0.2
    return w
