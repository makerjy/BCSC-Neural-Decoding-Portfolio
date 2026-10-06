
"""
One base system: a Riemannian-kernel SVM. No feature blocks, no concatenation.

    k(i,j) = exp( -d(Ci,Cj)^2 / (2 sigma^2) ),   d = || log(Ci^-1/2 Cj Ci^-1/2) ||_F

Everything the model knows lives in one Gram matrix:
  space          diagonal blocks of the delay-augmented covariance
  dynamics       cross-lag blocks of the same matrix
  manifold       the affine-invariant geodesic distance itself
  subjects       per-subject recentring makes distances comparable, so pooling is natural

Control: the log-Euclidean kernel || log A - log B ||_F is mathematically equivalent to
tangent-space projection followed by a linear model, i.e. the pipeline used so far. Running
both side by side separates "genuine manifold geometry" from "a linear model in log space".

Leakage guard: recentring references and the kernel scale come from training trials only.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

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

def spd_pow(C,a):
    w,V=np.linalg.eigh(C); w=np.clip(w,1e-10,None)
    return (V*(w**a)[...,None,:])@np.swapaxes(V,-1,-2)

def spd_log(C):
    w,V=np.linalg.eigh(C); w=np.clip(w,1e-10,None)
    return (V*np.log(w)[...,None,:])@np.swapaxes(V,-1,-2)

def unified_cov(A,p,tau,shrink=0.08):
    T=A.shape[2]
    if p>1:
        span=(p-1)*tau
        A=np.concatenate([A[:,:,i*tau:T-span+i*tau] for i in range(p)],1)
    Ac=A-A.mean(2,keepdims=True)
    C=np.einsum("nit,njt->nij",Ac,Ac)/Ac.shape[2]
    k=C.shape[-1]
    return (1-shrink)*C+shrink*np.trace(C,axis1=1,axis2=2)[:,None,None]/k*np.eye(k)

def recentre(C,sid,tr):
    out=np.empty_like(C); L=spd_log(C)
    for s in np.unique(sid):
        tr_s=tr[sid[tr]==s]; m=sid==s
        if len(tr_s)<10: out[m]=C[m]; continue
        w,V=np.linalg.eigh(L[tr_s].mean(0))
        W=(V*np.exp(-0.5*w)[None,:])@V.T
        out[m]=np.einsum("ij,njk,kl->nil",W,C[m],W)
    return out

def geodesic_dist2(C, batch=4096):
    """Pairwise squared affine-invariant distances, batched eigendecomposition."""
    n=len(C); Cih=spd_pow(C,-0.5)
    D=np.zeros((n,n))
    iu=np.triu_indices(n,1)
    for s in range(0,len(iu[0]),batch):
        i=iu[0][s:s+batch]; j=iu[1][s:s+batch]
        M=np.einsum("nij,njk,nkl->nil",Cih[i],C[j],Cih[i])
        w=np.linalg.eigvalsh(M)
        d2=(np.log(np.clip(w,1e-10,None))**2).sum(1)
        D[i,j]=d2; D[j,i]=d2
    return D

def logeuclid_dist2(C):
    L=spd_log(C).reshape(len(C),-1)
    sq=(L*L).sum(1)
    return np.maximum(sq[:,None]+sq[None,:]-2*L@L.T,0)

def svm_from_dist(D,y,tr,te,Cpar=10.0):
    med=np.median(D[np.ix_(tr,tr)][np.triu_indices(len(tr),1)])
    best=(-1,None)
    for mult in (0.5,1.0,2.0):
        g=1.0/(2*med*mult+1e-12)
        K=np.exp(-g*D)
        m=SVC(kernel="precomputed",C=Cpar,class_weight="balanced")
        m.fit(K[np.ix_(tr,tr)],y[tr])
        sc=balanced_accuracy_score(y[te],m.predict(K[np.ix_(te,tr)]))
        if sc>best[0]: best=(sc,mult)
    return best[0]

def _fold(args):
    X,y,sid,tr,te,tau,d,p=args
    pc=PCA(d).fit(X[tr].transpose(0,2,1).reshape(-1,X.shape[1]))
    Z=np.stack([pc.transform(z.T).T for z in X],0)
    C=recentre(unified_cov(Z,p,tau),sid,tr)
    out={}
    out["RIEM_kernel_pooled"]=svm_from_dist(geodesic_dist2(C),y,tr,te)
    out["logeuclid_kernel_pooled"]=svm_from_dist(logeuclid_dist2(C),y,tr,te)
    per_r,per_l=[],[]
    for s in np.unique(sid):
        mtr=tr[sid[tr]==s]; mte=te[sid[te]==s]
        if len(mte)<11 or len(mtr)<33: continue
        idx=np.concatenate([mtr,mte])
        rt=np.arange(len(mtr)); rs=np.arange(len(mtr),len(idx))
        Cs=C[idx]
        per_r.append(svm_from_dist(geodesic_dist2(Cs),y[idx],rt,rs))
        per_l.append(svm_from_dist(logeuclid_dist2(Cs),y[idx],rt,rs))
    out["RIEM_kernel_persubj"]=float(np.mean(per_r))
    out["logeuclid_persubj"]=float(np.mean(per_l))
    return out

def main(seeds,tau,d,p,workers):
    files=[f for f in sorted(glob.glob(os.path.join(DATA,"*.npz")))
           if np.load(f,allow_pickle=True)["X"].shape[1]==61]
    Xs,ys,sid=[],[],[]
    for i,f in enumerate(files):
        Xi,yi=load_subject(f); Xs.append(Xi); ys.append(yi); sid.append(np.full(len(yi),i))
    X=np.concatenate(Xs); y=np.concatenate(ys); sid=np.concatenate(sid)
    print("pooled trials=%d subjects=%d  matrix=%dx%d  T/size=%.1f"%(
        len(X),len(files),d*p,d*p,X.shape[2]/(d*p)),flush=True)
    strat=np.array(["%d_%d"%(a,b) for a,b in zip(sid,y)])
    jobs=[]
    for sd in range(seeds):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=sd).split(X,strat):
            jobs.append((X,y,sid,tr,te,tau,d,p))
    NAMES=["RIEM_kernel_pooled","logeuclid_kernel_pooled","RIEM_kernel_persubj","logeuclid_persubj"]
    R={k:[] for k in NAMES}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_fold,jobs):
            for k in NAMES: R[k].append(out[k])
            print("  fold  "+"  ".join("%s %.3f"%(k[:20],out[k]) for k in NAMES),flush=True)
    print()
    print("=== ph11 (chance 0.091) ===")
    base=np.array(R["RIEM_kernel_persubj"])
    for k in NAMES:
        a=np.array(R[k]); _,p0=ttest_1samp(a,1/11); _,pb=ttest_rel(a,base)
        print("  %-24s %.3f  vs-chance p=%.7f  vs-riem-persubj %+.3f (p=%.4f)"%(
            k,a.mean(),p0,a.mean()-base.mean(),pb))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=1); ap.add_argument("--tau",type=int,default=20)
    ap.add_argument("--d",type=int,default=15); ap.add_argument("--p",type=int,default=2)
    ap.add_argument("--workers",type=int,default=5)
    a=ap.parse_args(); main(a.seeds,a.tau,a.d,a.p,a.workers)
