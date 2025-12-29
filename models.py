import numpy as np,lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

class IsotonicCalibrator:
    def __init__(self): self.cal=IsotonicRegression(out_of_bounds="clip")
    def fit(self,y_pred,y_true): self.cal.fit(y_pred,y_true.astype(int))
    def transform(self,y_pred): return self.cal.transform(y_pred)

def train_lgbm(X,y,feature_cols,w,seed=42,n_splits=5):
    labeled=y.notna().values; X_lab=X.loc[labeled,feature_cols]; y_lab=y.loc[labeled].astype(int).values; w_lab=w[labeled]
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    oof=np.zeros(len(X_lab)); models=[]
    params={"objective":"binary","metric":["auc"],"learning_rate":0.05,"num_leaves":64,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":1,"min_data_in_leaf":50,"lambda_l2":5.0,"seed":seed}
    for tr,va in skf.split(X_lab,y_lab):
        dtrain=lgb.Dataset(X_lab.iloc[tr],label=y_lab[tr],weight=w_lab[tr]); dvalid=lgb.Dataset(X_lab.iloc[va],label=y_lab[va],weight=w_lab[va])
        m=lgb.train(params,dtrain,num_boost_round=2000,valid_sets=[dvalid],early_stopping_rounds=200,verbose_eval=False)
        models.append(m); oof[va]=m.predict(X_lab.iloc[va],num_iteration=m.best_iteration)
    auc=roc_auc_score(y_lab,oof) if len(np.unique(y_lab))>1 else float("nan")
    return models,oof,auc

def predict_lgbm(models,X,feature_cols):
    return np.mean([m.predict(X[feature_cols],num_iteration=m.best_iteration) for m in models],axis=0)
