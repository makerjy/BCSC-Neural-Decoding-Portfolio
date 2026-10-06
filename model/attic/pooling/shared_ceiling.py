
"""
How much of each subject's reliable phoneme structure is shared?

Pooling can only use the shared part. Per-subject models use all of it. So the pooled ceiling
is set by shared_fraction = (between-subject geometry correlation) / (within-subject split-half
reliability), and no alignment method can raise it -- alignment finds shared structure, it does
not create it.

Measured here:
  reliability   : split-half correlation of a subject's phoneme RDM with itself
  cross-subject : correlation of one subject's RDM with the group mean of the others
  shared frac   : cross / reliability

Then two untested front-end filters are run against the same folds:
  joint diagonalisation of class covariances
  CORAL (second-order statistic matching)
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.linalg import eigh, fractional_matrix_power as fmp
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_rel, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

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

def rdm(Z,y,idx):
    P=np.stack([Z[idx][y[idx]==c].mean(0).ravel() for c in range(11)])
    D=np.corrcoef(P); return squareform(1-D,checks=False)

def reliability(Z,y,rng,reps=20):
    out=[]
    for _ in range(reps):
        a,b=[],[]
        for c in range(11):
            i=np.where(y==c)[0]; rng.shuffle(i); h=len(i)//2
            a.append(i[:h]); b.append(i[h:2*h])
        r=spearmanr(rdm(Z,y,np.concatenate(a)),rdm(Z,y,np.concatenate(b))).statistic
        out.append(r)
    return float(np.mean(out))

def coral(Zs,Zt_mean_cov,eps=1e-6):
    d=Zs.shape[1]; F=Zs.transpose(0,2,1).reshape(-1,d)
    Cs=np.cov(F.T)+eps*np.eye(d)
    W=np.real(fmp(Cs,-0.5))@np.real(fmp(Zt_mean_cov,0.5))
    return np.einsum("ij,njt->nit",W.T,Zs)

def jd_filter(Zs,ys,n_cls):
    d=Zs.shape[1]
    Cc=np.stack([Covariances("oas").fit_transform(Zs[ys==c]).mean(0) for c in range(n_cls)])
    Cb=Cc.mean(0); Sb=np.cov(np.stack([c.ravel() for c in Cc]).T).reshape(d,d,d,d).trace(axis1=2,axis2=3) if False else Cc.var(0)
    w,V=eigh(Sb+1e-8*np.eye(d),Cb+1e-8*np.eye(d))
    return V[:,np.argsort(w)[::-1]].T

def clf(Ftr,ytr,Fte,yte):
    m=make_pipeline(StandardScaler(),LDA(solver="lsqr",shrinkage="auto"))
    m.fit(Ftr,ytr); return balanced_accuracy_score(yte,m.predict(Fte))

def feats(Z,sid,tr,K,tau):
    T=Z.shape[2]; span=tau
    A=np.concatenate([Z[:,:,:T-span],Z[:,:,span:]],1)
    Ac=A-A.mean(2,keepdims=True)
    C=np.einsum("nit,njt->nij",Ac,Ac)/Ac.shape[2]
    k=C.shape[-1]; C=0.92*C+0.08*np.trace(C,axis1=1,axis2=2)[:,None,None]/k*np.eye(k)
    ts=TangentSpace(metric="logeuclid").fit(C[tr]); F=ts.transform(C)
    ed=np.linspace(0,T,K+1).astype(int); blocks=[F]
    for i in range(K):
        seg=Z[:,:,ed[i]:ed[i+1]]
        Cw=Covariances("oas").fit_transform(seg)
        t2=TangentSpace(metric="logeuclid").fit(Cw[tr])
        blocks+= [t2.transform(Cw),seg.mean(2)]
    sc=[StandardScaler().fit(b[tr]) for b in blocks]
    return np.hstack([s.transform(b) for s,b in zip(sc,blocks)])

def main(seeds,tau):
    files=[f for f in sorted(glob.glob(os.path.join(DATA,"*.npz")))
           if np.load(f,allow_pickle=True)["X"].shape[1]==61]
    Xs,ys,sid=[],[],[]
    for i,f in enumerate(files):
        X,y=load_subject(f); Xs.append(X); ys.append(y); sid.append(np.full(len(y),i))
    X=np.concatenate(Xs); y=np.concatenate(ys); sid=np.concatenate(sid); n_sub=len(files)
    pcg=PCA(20).fit(X.transpose(0,2,1).reshape(-1,X.shape[1]))
    Z=np.stack([pcg.transform(z.T).T for z in X],0)[:,:,::4]
    rng=np.random.default_rng(0)
    rel,cross=[],[]
    R=[rdm(Z[sid==s],y[sid==s],np.arange((sid==s).sum())) for s in range(n_sub)]
    for s in range(n_sub):
        rel.append(reliability(Z[sid==s],y[sid==s],rng))
        others=np.mean([R[o] for o in range(n_sub) if o!=s],axis=0)
        cross.append(spearmanr(R[s],others).statistic)
    rel=np.array(rel); cross=np.array(cross)
    frac=cross.mean()/max(rel.mean(),1e-9)
    print("=== shared-structure accounting ===")
    print("  within-subject split-half reliability  %.3f" % rel.mean())
    print("  subject-vs-group RDM correlation       %.3f" % cross.mean())
    print("  shared fraction of reliable structure  %.3f" % frac)
    print("  -> pooling can use %.0f%% of what a per-subject model uses" % (100*frac))
    print()
    K=6
    Zf=np.stack([pcg.transform(z.T).T for z in X],0)
    strat=np.array(["%d_%d"%(a,b) for a,b in zip(sid,y)])
    NAMES=["per_subject","pooled","pooled_JD","pooled_CORAL"]
    RES={k:[] for k in NAMES}
    for sd in range(seeds):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=sd).split(Zf,strat):
            F=feats(Zf,sid,tr,K,tau)
            RES["pooled"].append(clf(F[tr],y[tr],F[te],y[te]))
            Zj=Zf.copy()
            for s in range(n_sub):
                m=sid==s; mtr=sid[tr]==s
                W=jd_filter(Zf[tr][mtr],y[tr][mtr],11)
                Zj[m]=np.einsum("ij,njt->nit",W,Zf[m])
            Fj=feats(Zj,sid,tr,K,tau)
            RES["pooled_JD"].append(clf(Fj[tr],y[tr],Fj[te],y[te]))
            Ft=Zf[tr].transpose(0,2,1).reshape(-1,Zf.shape[1])
            Ct=np.cov(Ft.T)
            Zc=Zf.copy()
            for s in range(n_sub):
                m=sid==s; mtr=tr[sid[tr]==s]
                Fs=Zf[mtr].transpose(0,2,1).reshape(-1,Zf.shape[1])
                Cs=np.cov(Fs.T)+1e-6*np.eye(Zf.shape[1])
                W=np.real(fmp(Cs,-0.5))@np.real(fmp(Ct,0.5))
                Zc[m]=np.einsum("ij,njt->nit",W.T,Zf[m])
            Fc=feats(Zc,sid,tr,K,tau)
            RES["pooled_CORAL"].append(clf(Fc[tr],y[tr],Fc[te],y[te]))
            per=[]
            for s in range(n_sub):
                mtr,mte=sid[tr]==s,sid[te]==s
                if mte.sum()<11 or mtr.sum()<33: continue
                idx=np.concatenate([tr[mtr],te[mte]])
                rt=np.arange(mtr.sum()); rs=np.arange(mtr.sum(),len(idx))
                pcs=PCA(20).fit(Xs[s].transpose(0,2,1).reshape(-1,X.shape[1]))
                Zi=np.stack([pcs.transform(z.T).T for z in X[idx]],0)
                Fi=feats(Zi,np.zeros(len(idx),int),rt,K,tau)
                per.append(clf(Fi[rt],y[idx][rt],Fi[rs],y[idx][rs]))
            RES["per_subject"].append(float(np.mean(per)))
            print("  fold done  " + "  ".join("%s %.3f"%(k[:12],RES[k][-1]) for k in NAMES),flush=True)
    print()
    print("=== ph11 (chance 0.091) ===")
    base=np.array(RES["per_subject"])
    for k in NAMES:
        a=np.array(RES[k]); _,p0=ttest_1samp(a,1/11); _,pb=ttest_rel(a,base)
        print("  %-14s %.3f  vs-chance p=%.7f  vs-per-subject %+.3f (p=%.4f)"%(
            k,a.mean(),p0,a.mean()-base.mean(),pb))
    print()
    print("  predicted pooled ceiling from shared fraction: %.3f x (per-subject - chance) + chance = %.3f"%(
        frac, frac*(base.mean()-1/11)+1/11))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=1)
    ap.add_argument("--tau",type=int,default=20); a=ap.parse_args(); main(a.seeds,a.tau)
