
"""
Where does pooling actually help?

Instead of one more global pooled-vs-per-subject comparison, split the pipeline into three
stages and pool each independently:

    spatial basis (PCA)  ->  tangent reference  ->  classifier

All 8 combinations on identical folds. This measures which parts of the model are shared
across people and which are individual, rather than assuming a factorisation and imposing it.

Also reruns ARD with the earlier bug fixed: targets were one-hot minus 1/11, so their scale was
~0.09 and the noise precision blew up, pinning most alphas at the clip bound. Targets are now
unit-scaled and the bounds widened, so relevance is actually distributed.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
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
    return S[...,iu[0],iu[1]]*np.where(iu[0]==iu[1],1.0,np.sqrt(2.0))

def make_z(X,sid,tr,d,p,tau,K_win,basis_pooled,ref_pooled):
    if basis_pooled:
        pc=PCA(d).fit(X[tr].transpose(0,2,1).reshape(-1,X.shape[1]))
        Z=np.stack([pc.transform(z.T).T for z in X],0)
    else:
        Z=np.empty((len(X),d,X.shape[2]))
        for s in np.unique(sid):
            tr_s=tr[sid[tr]==s]; m=sid==s
            src=tr_s if len(tr_s)>=d+2 else tr
            pc=PCA(d).fit(X[src].transpose(0,2,1).reshape(-1,X.shape[1]))
            Z[m]=np.stack([pc.transform(z.T).T for z in X[m]],0)
    T=Z.shape[2]
    A=Z if p==1 else np.concatenate(
        [Z[:,:,i*tau:T-(p-1)*tau+i*tau] for i in range(p)],1)
    def tangent(C):
        L=spd_log(C)
        if ref_pooled:
            return triu(L-L[tr].mean(0))
        out=L.copy()
        for s in np.unique(sid):
            tr_s=tr[sid[tr]==s]
            if len(tr_s)>=10: out[sid==s]=L[sid==s]-L[tr_s].mean(0)
        return triu(out)
    blocks=[tangent(cov(A))]
    ed=np.linspace(0,T,K_win+1).astype(int)
    for i in range(K_win):
        seg=Z[:,:,ed[i]:ed[i+1]]
        blocks.append(tangent(cov(seg)))
        M=seg.mean(2)
        if ref_pooled: M=M-M[tr].mean(0)
        else:
            for s in np.unique(sid):
                tr_s=tr[sid[tr]==s]
                if len(tr_s)>=10: M[sid==s]=M[sid==s]-M[tr_s].mean(0)
        blocks.append(M)
    Zf=np.hstack(blocks)
    return StandardScaler().fit(Zf[tr]).transform(Zf)

def lda_acc(Z,y,tr,te):
    m=make_pipeline(StandardScaler(),LDA(solver="lsqr",shrinkage="auto"))
    m.fit(Z[tr],y[tr]); return balanced_accuracy_score(y[te],m.predict(Z[te]))

def persubj_acc(Z,y,sid,tr,te):
    per=[]
    for s in np.unique(sid):
        mtr=tr[sid[tr]==s]; mte=te[sid[te]==s]
        if len(mte)<11 or len(mtr)<33: continue
        per.append(lda_acc(Z,y,mtr,mte))
    return float(np.mean(per))

def ard_fit(Phi,Y,n_iter=60):
    n,d=Phi.shape
    Y=Y/ (Y.std()+1e-12)
    alpha=np.ones(d); beta=1.0
    PtP=Phi.T@Phi; PtY=Phi.T@Y; W=np.zeros((d,Y.shape[1]))
    for _ in range(n_iter):
        Sigma=np.linalg.inv(beta*PtP+np.diag(alpha)+1e-10*np.eye(d))
        W=beta*Sigma@PtY
        g=1.0-alpha*np.diag(Sigma)
        alpha=np.clip(g/np.maximum((W**2).mean(1),1e-14),1e-8,1e10)
        resid=((Y-Phi@W)**2).sum()
        beta=float(np.clip((n*Y.shape[1]-g.sum()*Y.shape[1])/max(resid,1e-9),1e-8,1e10))
    return W,alpha

def _fold(args):
    X,y,sid,tr,te,d,p,tau,K_win=args
    n_cls=len(np.unique(y)); out={}
    for bp,rp in product([True,False],[True,False]):
        Z=make_z(X,sid,tr,d,p,tau,K_win,bp,rp)
        tag="basis%s_ref%s"%("P" if bp else "S","P" if rp else "S")
        out[tag+"_clfP"]=lda_acc(Z,y,tr,te)
        out[tag+"_clfS"]=persubj_acc(Z,y,sid,tr,te)
        if bp and rp:
            Y=np.eye(n_cls)[y[tr]]
            W,al=ard_fit(Z[tr],Y)
            out["ARD_fixed_pooled"]=balanced_accuracy_score(
                y[te],np.argmax(Z[te]@W,axis=1))
            out["ard_alpha_spread"]=float(np.log10(al.max()/max(al.min(),1e-12)))
            out["ard_frac_pinned"]=float(np.mean(al>1e9))
    return out

def main(seeds,d,p,tau,K_win,workers):
    files=[f for f in sorted(glob.glob(os.path.join(DATA,"*.npz")))
           if np.load(f,allow_pickle=True)["X"].shape[1]==61]
    Xs,ys,sid=[],[],[]
    for i,f in enumerate(files):
        Xi,yi=load_subject(f); Xs.append(Xi); ys.append(yi); sid.append(np.full(len(yi),i))
    X=np.concatenate(Xs); y=np.concatenate(ys); sid=np.concatenate(sid)
    print("pooled trials=%d subjects=%d"%(len(X),len(files)),flush=True)
    strat=np.array(["%d_%d"%(a,b) for a,b in zip(sid,y)])
    jobs=[]
    for sd in range(seeds):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=sd).split(X,strat):
            jobs.append((X,y,sid,tr,te,d,p,tau,K_win))
    NAMES=[]
    for bp,rp,cp in product("PS","PS","PS"):
        NAMES.append("basis%s_ref%s_clf%s"%(bp,rp,cp))
    NAMES+=["ARD_fixed_pooled"]
    R={k:[] for k in NAMES}; diag=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_fold,jobs):
            for k in NAMES: R[k].append(out[k])
            diag.append((out["ard_alpha_spread"],out["ard_frac_pinned"]))
            print("  fold done",flush=True)
    dg=np.array(diag).mean(0)
    print()
    print("  ARD diagnostic: log10(alpha range) %.1f   fraction pinned at bound %.2f"%(dg[0],dg[1]))
    print()
    print("=== ph11 (chance 0.091)   P=pooled  S=per-subject ===")
    rows=sorted(NAMES,key=lambda k:-np.mean(R[k]))
    best=np.array(R[rows[0]])
    for k in rows:
        a=np.array(R[k]); _,p0=ttest_1samp(a,1/11); _,pb=ttest_rel(a,best)
        print("  %-26s %.3f  vs-chance p=%.7f  vs-best %+.3f (p=%.4f)"%(
            k,a.mean(),p0,a.mean()-best.mean(),pb))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=1); ap.add_argument("--d",type=int,default=14)
    ap.add_argument("--p",type=int,default=2); ap.add_argument("--tau",type=int,default=20)
    ap.add_argument("--windows",type=int,default=4); ap.add_argument("--workers",type=int,default=5)
    a=ap.parse_args(); main(a.seeds,a.d,a.p,a.tau,a.windows,a.workers)
