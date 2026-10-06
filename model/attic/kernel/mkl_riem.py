
"""
One SVM, one dual problem -- but the kernel decides its own internal mixing.

A single geodesic kernel lost to the block pipeline (0.183 vs 0.270). The reason is not that
everything was in one object; it is that one scalar distance mixes space, dynamics and evoked
deflection at a ratio fixed in advance, while a concatenation lets the classifier weight those
axes independently.

So: keep one model, but let it learn the mixing.

    k = sum_m beta_m k_m ,  beta from centred kernel-target alignment on TRAINING trials only

    k1  geodesic on the spatial covariance          (space)
    k2  geodesic on the delay-augmented covariance  (space + cross-lag dynamics)
    k3  geodesic on per-window covariances          (time-resolved geometry)
    k4  RBF on window means                         (evoked deflection the covariance drops)

Every base kernel is a manifold object; beta is the only thing learned besides the SVM duals.
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

def spd_log(C):
    w,V=np.linalg.eigh(C); w=np.clip(w,1e-10,None)
    return (V*np.log(w)[...,None,:])@np.swapaxes(V,-1,-2)

def cov(A,shrink=0.08):
    Ac=A-A.mean(2,keepdims=True)
    C=np.einsum("nit,njt->nij",Ac,Ac)/Ac.shape[2]
    k=C.shape[-1]
    return (1-shrink)*C+shrink*np.trace(C,axis1=1,axis2=2)[:,None,None]/k*np.eye(k)

def delay_aug(A,p,tau):
    if p==1: return A
    T=A.shape[2]; span=(p-1)*tau
    return np.concatenate([A[:,:,i*tau:T-span+i*tau] for i in range(p)],1)

def recentre_log(C,sid,tr):
    L=spd_log(C); out=L.copy()
    for s in np.unique(sid):
        tr_s=tr[sid[tr]==s]; m=sid==s
        if len(tr_s)<10: continue
        out[m]=L[m]-L[tr_s].mean(0)
    return out

def dist2_from_log(L):
    F=L.reshape(len(L),-1); sq=(F*F).sum(1)
    return np.maximum(sq[:,None]+sq[None,:]-2*F@F.T,0)

def dist2_euclid(F):
    sq=(F*F).sum(1)
    return np.maximum(sq[:,None]+sq[None,:]-2*F@F.T,0)

def rbf(D,tr,mult=1.0):
    med=np.median(D[np.ix_(tr,tr)][np.triu_indices(len(tr),1)])
    return np.exp(-D/(2*med*mult+1e-12))

def centre_K(K):
    n=K.shape[0]; H=np.eye(n)-np.ones((n,n))/n
    return H@K@H

def alignment_weights(Ks,y,tr):
    """Centred kernel-target alignment on the training block; beta >= 0, unit norm."""
    n=len(tr); Y=(y[tr][:,None]==y[tr][None,:]).astype(float)
    Yc=centre_K(Y); nY=np.linalg.norm(Yc)
    b=[]
    for K in Ks:
        Kc=centre_K(K[np.ix_(tr,tr)])
        b.append(max(0.0,float((Kc*Yc).sum()/(np.linalg.norm(Kc)*nY+1e-12))))
    b=np.array(b); s=b.sum()
    return b/s if s>0 else np.ones(len(Ks))/len(Ks)

def svm_eval(K,y,tr,te,C=10.0):
    m=SVC(kernel="precomputed",C=C,class_weight="balanced")
    m.fit(K[np.ix_(tr,tr)],y[tr])
    return balanced_accuracy_score(y[te],m.predict(K[np.ix_(te,tr)]))

def build_kernels(X,sid,tr,d,p,tau,K_win):
    pc=PCA(d).fit(X[tr].transpose(0,2,1).reshape(-1,X.shape[1]))
    Z=np.stack([pc.transform(z.T).T for z in X],0)
    Ks=[]
    Ks.append(rbf(dist2_from_log(recentre_log(cov(Z),sid,tr)),tr))
    Ks.append(rbf(dist2_from_log(recentre_log(cov(delay_aug(Z,p,tau)),sid,tr)),tr))
    ed=np.linspace(0,Z.shape[2],K_win+1).astype(int)
    Lw=[]; Mw=[]
    for i in range(K_win):
        seg=Z[:,:,ed[i]:ed[i+1]]
        Lw.append(recentre_log(cov(seg),sid,tr).reshape(len(Z),-1))
        Mw.append(seg.mean(2))
    Ks.append(rbf(dist2_euclid(np.hstack(Lw)),tr))
    M=np.hstack(Mw)
    for s in np.unique(sid):
        m=sid==s; tr_s=tr[sid[tr]==s]
        if len(tr_s)>=10: M[m]=M[m]-M[tr_s].mean(0)
    Ks.append(rbf(dist2_euclid(M),tr))
    return Ks

def _fold(args):
    X,y,sid,tr,te,d,p,tau,K_win=args
    Ks=build_kernels(X,sid,tr,d,p,tau,K_win)
    out={}
    for i,nm in enumerate(["k_space","k_delay","k_windows","k_means"]):
        out[nm]=svm_eval(Ks[i],y,tr,te)
    out["MKL_uniform"]=svm_eval(np.mean(Ks,axis=0),y,tr,te)
    b=alignment_weights(Ks,y,tr)
    out["MKL_aligned"]=svm_eval(np.tensordot(b,np.stack(Ks),axes=1),y,tr,te)
    out["beta"]=b
    per=[]
    for s in np.unique(sid):
        mtr=tr[sid[tr]==s]; mte=te[sid[te]==s]
        if len(mte)<11 or len(mtr)<33: continue
        idx=np.concatenate([mtr,mte]); rt=np.arange(len(mtr)); rs=np.arange(len(mtr),len(idx))
        Kl=[K[np.ix_(idx,idx)] for K in Ks]
        bb=alignment_weights(Kl,y[idx],rt)
        per.append(svm_eval(np.tensordot(bb,np.stack(Kl),axes=1),y[idx],rt,rs))
    out["MKL_persubj"]=float(np.mean(per))
    return out

def main(seeds,d,p,tau,K_win,workers):
    files=[f for f in sorted(glob.glob(os.path.join(DATA,"*.npz")))
           if np.load(f,allow_pickle=True)["X"].shape[1]==61]
    Xs,ys,sid=[],[],[]
    for i,f in enumerate(files):
        Xi,yi=load_subject(f); Xs.append(Xi); ys.append(yi); sid.append(np.full(len(yi),i))
    X=np.concatenate(Xs); y=np.concatenate(ys); sid=np.concatenate(sid)
    print("pooled trials=%d subjects=%d d=%d p=%d windows=%d"%(len(X),len(files),d,p,K_win),flush=True)
    strat=np.array(["%d_%d"%(a,b) for a,b in zip(sid,y)])
    jobs=[]
    for sd in range(seeds):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=sd).split(X,strat):
            jobs.append((X,y,sid,tr,te,d,p,tau,K_win))
    NAMES=["k_space","k_delay","k_windows","k_means","MKL_uniform","MKL_aligned","MKL_persubj"]
    R={k:[] for k in NAMES}; B=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_fold,jobs):
            for k in NAMES: R[k].append(out[k])
            B.append(out["beta"])
            print("  fold  "+"  ".join("%s %.3f"%(k[:11],out[k]) for k in NAMES),flush=True)
    print()
    print("  learned beta (space, delay, windows, means): "+
          "  ".join("%.3f"%v for v in np.mean(B,axis=0)))
    print()
    print("=== ph11 (chance 0.091) ===")
    base=np.array(R["MKL_aligned"])
    for k in NAMES:
        a=np.array(R[k]); _,p0=ttest_1samp(a,1/11); _,pb=ttest_rel(a,base)
        print("  %-14s %.3f  vs-chance p=%.7f  vs-MKL_aligned %+.3f (p=%.4f)"%(
            k,a.mean(),p0,a.mean()-base.mean(),pb))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=1); ap.add_argument("--d",type=int,default=20)
    ap.add_argument("--p",type=int,default=2); ap.add_argument("--tau",type=int,default=20)
    ap.add_argument("--windows",type=int,default=6); ap.add_argument("--workers",type=int,default=5)
    a=ap.parse_args(); main(a.seeds,a.d,a.p,a.tau,a.windows,a.workers)
