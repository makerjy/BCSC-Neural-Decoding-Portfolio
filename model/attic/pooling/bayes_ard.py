
"""
One hierarchical Bayesian model over a single unified representation.

    y_i ~ Cat(softmax(W z_i + b_s(i)))
    W_q ~ N(0, alpha_q^-1)     ARD: per-coordinate relevance, learned by evidence
    b_s ~ N(0, tau)            subject offsets, factorisation built into the generative form

z_i is one object: the tangent vector of the delay-augmented covariance (space in the diagonal
blocks, dynamics in the cross-lag blocks) together with window geometry and window means.

Why this and not MKL: a kernel compresses each geometry into one scalar similarity, so the
model can weight axes but not coordinates inside an axis. That cost the MKL run 0.09 against
the block pipeline. ARD keeps one model while restoring per-coordinate freedom, and the prior
does the regularising, so it does not need many trials to behave.

Controls on identical folds: shrinkage LDA on the same z (the pipeline), ARD without subject
offsets, ARD with them, and the per-subject reference.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA="npz2"; PHON=["a","i","e","u","o","s","z","p","b","t","d"]
WMIN,WMAX=0.0,1.01; BASE=(-0.2,-0.11)

def load_subject(path):
    d=np.load(path,allow_pickle=True); X,t=d["X"].astype(float),d["times"]
    meta=pd.DataFrame({c:[str(v) for v in d[c]] for c in ["task","tms_target","phoneme1"]})
    keep=((meta.task=="single_phoneme")&(meta.tms_target=="control")).values
    bm=(t>=BASE[0])&(t<=BASE[1]); wm=(t>=WMIN)&(t<=WMAX); Xk=X[keep]
    b=Xk[:,:,bm].mean(2,keepdims=True) if bm.any() else 0.0
    ph=meta.phoneme1.values[keep]; m=np.isin(ph,PHON)
    return (Xk-b)[:,:,wm][m], np.array([PHON.index(v) for v in ph[m]])

def spd_log(C):
    w,V=np.linalg.eigh(C); w=np.clip(w,1e-10,None)
    return (V*np.log(w)[...,None,:])@np.swapaxes(V,-1,-2)

def cov(A,shrink=0.08):
    Ac=A-A.mean(2,keepdims=True)
    C=np.einsum("nit,njt->nij",Ac,Ac)/Ac.shape[2]
    k=C.shape[-1]
    return (1-shrink)*C+shrink*np.trace(C,axis1=1,axis2=2)[:,None,None]/k*np.eye(k)

def triu(S):
    d=S.shape[-1]; iu=np.triu_indices(d)
    sc=np.where(iu[0]==iu[1],1.0,np.sqrt(2.0))
    return S[...,iu[0],iu[1]]*sc

def recentre_log(C,sid,tr):
    L=spd_log(C); out=L.copy()
    for s in np.unique(sid):
        tr_s=tr[sid[tr]==s]
        if len(tr_s)<10: continue
        out[sid==s]=L[sid==s]-L[tr_s].mean(0)
    return out

def unified_z(X,sid,tr,d,p,tau,K_win):
    pc=PCA(d).fit(X[tr].transpose(0,2,1).reshape(-1,X.shape[1]))
    Z=np.stack([pc.transform(z.T).T for z in X],0)
    T=Z.shape[2]
    if p>1:
        span=(p-1)*tau
        A=np.concatenate([Z[:,:,i*tau:T-span+i*tau] for i in range(p)],1)
    else: A=Z
    blocks=[triu(recentre_log(cov(A),sid,tr))]
    ed=np.linspace(0,T,K_win+1).astype(int)
    for i in range(K_win):
        seg=Z[:,:,ed[i]:ed[i+1]]
        blocks.append(triu(recentre_log(cov(seg),sid,tr)))
        M=seg.mean(2)
        for s in np.unique(sid):
            tr_s=tr[sid[tr]==s]
            if len(tr_s)>=10: M[sid==s]=M[sid==s]-M[tr_s].mean(0)
        blocks.append(M)
    Zf=np.hstack(blocks)
    sc=StandardScaler().fit(Zf[tr])
    return sc.transform(Zf)

def ard_fit(Phi,Y,n_iter=40,tol=1e-3):
    """Evidence maximisation for a linear-Gaussian model with per-coordinate precision.
    Phi (n,d), Y (n,C) one-hot. Returns weights (d,C)."""
    n,d=Phi.shape
    alpha=np.ones(d); beta=1.0
    PtP=Phi.T@Phi; PtY=Phi.T@Y
    W=np.zeros((d,Y.shape[1]))
    for _ in range(n_iter):
        A=beta*PtP+np.diag(alpha)
        try: Sigma=np.linalg.inv(A+1e-9*np.eye(d))
        except np.linalg.LinAlgError: break
        W=beta*Sigma@PtY
        g=1.0-alpha*np.diag(Sigma)
        w2=(W**2).mean(1)
        alpha_new=np.clip(g/np.maximum(w2,1e-12),1e-6,1e6)
        resid=((Y-Phi@W)**2).sum()
        beta_new=float(np.clip((n*Y.shape[1]-g.sum()*Y.shape[1])/max(resid,1e-9),1e-6,1e6))
        if np.mean(np.abs(np.log(alpha_new+1e-12)-np.log(alpha+1e-12)))<tol:
            alpha,beta=alpha_new,beta_new; break
        alpha,beta=alpha_new,beta_new
    return W,alpha

def ard_predict(Phi,W,n_cls):
    return np.argmax(Phi@W,axis=1)

def onehot(y,n_cls):
    Y=np.zeros((len(y),n_cls)); Y[np.arange(len(y)),y]=1.0
    return Y-1.0/n_cls

def _fold(args):
    X,y,sid,tr,te,d,p,tau,K_win,n_sub=args
    n_cls=len(np.unique(y))
    Z=unified_z(X,sid,tr,d,p,tau,K_win)
    out={}
    m=make_pipeline(StandardScaler(),LDA(solver="lsqr",shrinkage="auto"))
    m.fit(Z[tr],y[tr]); out["pipeline_LDA_pooled"]=balanced_accuracy_score(y[te],m.predict(Z[te]))
    Y=onehot(y[tr],n_cls)
    W,alpha=ard_fit(Z[tr],Y)
    out["ARD_pooled"]=balanced_accuracy_score(y[te],ard_predict(Z[te],W,n_cls))
    S=np.eye(n_sub)[sid]
    Zs=np.hstack([Z,S])
    W2,a2=ard_fit(Zs[tr],Y)
    out["ARD_pooled+subj"]=balanced_accuracy_score(y[te],ard_predict(Zs[te],W2,n_cls))
    keep=np.argsort(alpha)[:min(400,len(alpha))]
    out["ARD_relevance_frac"]=float(np.mean(alpha<np.median(alpha)))
    per=[]
    for s in range(n_sub):
        mtr=tr[sid[tr]==s]; mte=te[sid[te]==s]
        if len(mte)<11 or len(mtr)<33: continue
        mm=make_pipeline(StandardScaler(),LDA(solver="lsqr",shrinkage="auto"))
        mm.fit(Z[mtr],y[mtr]); per.append(balanced_accuracy_score(y[mte],mm.predict(Z[mte])))
    out["per_subject_LDA"]=float(np.mean(per))
    return out

def main(seeds,d,p,tau,K_win,workers):
    files=[f for f in sorted(glob.glob(os.path.join(DATA,"*.npz")))
           if np.load(f,allow_pickle=True)["X"].shape[1]==61]
    Xs,ys,sid=[],[],[]
    for i,f in enumerate(files):
        Xi,yi=load_subject(f); Xs.append(Xi); ys.append(yi); sid.append(np.full(len(yi),i))
    X=np.concatenate(Xs); y=np.concatenate(ys); sid=np.concatenate(sid); n_sub=len(files)
    print("pooled trials=%d subjects=%d d=%d p=%d windows=%d"%(len(X),n_sub,d,p,K_win),flush=True)
    strat=np.array(["%d_%d"%(a,b) for a,b in zip(sid,y)])
    jobs=[]
    for sd in range(seeds):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=sd).split(X,strat):
            jobs.append((X,y,sid,tr,te,d,p,tau,K_win,n_sub))
    NAMES=["per_subject_LDA","pipeline_LDA_pooled","ARD_pooled","ARD_pooled+subj"]
    R={k:[] for k in NAMES}; rel=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_fold,jobs):
            for k in NAMES: R[k].append(out[k])
            rel.append(out["ARD_relevance_frac"])
            print("  fold  "+"  ".join("%s %.3f"%(k[:16],out[k]) for k in NAMES),flush=True)
    print()
    print("=== ph11 (chance 0.091) ===")
    base=np.array(R["per_subject_LDA"])
    for k in NAMES:
        a=np.array(R[k]); _,p0=ttest_1samp(a,1/11); _,pb=ttest_rel(a,base)
        print("  %-22s %.3f  vs-chance p=%.7f  vs-per-subject %+.3f (p=%.4f)"%(
            k,a.mean(),p0,a.mean()-base.mean(),pb))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=1); ap.add_argument("--d",type=int,default=14)
    ap.add_argument("--p",type=int,default=2); ap.add_argument("--tau",type=int,default=20)
    ap.add_argument("--windows",type=int,default=4); ap.add_argument("--workers",type=int,default=5)
    a=ap.parse_args(); main(a.seeds,a.d,a.p,a.tau,a.windows,a.workers)
