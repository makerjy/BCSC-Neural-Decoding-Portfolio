
"""
Does electrode position tell us what is individual and what is shared?

Individual differences in EEG are largely spatial: head shape, cortical folding and source
orientation change how the same neural activity projects onto the scalp. So they should be
smooth spatial transformations, not arbitrary channel mixtures. An unconstrained 61x61
alignment can permute distant electrodes, which is physically impossible and fits noise --
which is what the earlier Procrustes attempt did (-0.008).

Position gives a principled axis: the graph Laplacian of the electrode montage has eigenvectors
that are spatial frequencies on the scalp. Low frequencies are broad fields, high frequencies
are local. Hypothesis: broad components are shared across people, local ones are individual.

Per spatial-frequency band this measures
    within-subject decoding      is there phoneme information at this scale?
    cross-subject RDM agreement  is that information the same across people?
and then pools using only the bands that agree, which is a position-derived subject/phoneme
filter rather than a statistical one.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_1samp, ttest_rel
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
    pos=d["channel_positions_xyz"]
    return (Xk-b)[:,:,wm][m], np.array([PHON.index(v) for v in ph[m]]), np.asarray(pos,float)

def laplacian_basis(pos):
    P=pos.copy()
    bad=~np.isfinite(P).all(1)|(np.abs(P).sum(1)==0)
    P[bad]=P[~bad].mean(0)
    D2=((P[:,None,:]-P[None,:,:])**2).sum(-1)
    h2=np.median(D2[D2>0])
    A=np.exp(-D2/(2*h2)); np.fill_diagonal(A,0.0)
    L=np.diag(A.sum(1))-A
    w,V=eigh(L)
    return V, w

def spd_log(C):
    w,V=np.linalg.eigh(C); w=np.clip(w,1e-10,None)
    return (V*np.log(w)[...,None,:])@np.swapaxes(V,-1,-2)

def cov(A,shrink=0.1):
    Ac=A-A.mean(2,keepdims=True)
    C=np.einsum("nit,njt->nij",Ac,Ac)/Ac.shape[2]
    k=C.shape[-1]
    return (1-shrink)*C+shrink*np.trace(C,axis1=1,axis2=2)[:,None,None]/k*np.eye(k)

def triu(S):
    d=S.shape[-1]; iu=np.triu_indices(d)
    return S[...,iu[0],iu[1]]*np.where(iu[0]==iu[1],1.0,np.sqrt(2.0))

def band_features(Xb,K_win=4):
    T=Xb.shape[2]; ed=np.linspace(0,T,K_win+1).astype(int)
    blocks=[]
    for i in range(K_win):
        seg=Xb[:,:,ed[i]:ed[i+1]]
        blocks.append(triu(spd_log(cov(seg))))
        blocks.append(seg.mean(2))
    return np.hstack(blocks)

def rdm(F,y):
    P=np.stack([F[y==c].mean(0) for c in range(11)])
    return squareform(1-np.corrcoef(P),checks=False)

def lda_acc(F,y,tr,te):
    m=make_pipeline(StandardScaler(),LDA(solver="lsqr",shrinkage="auto"))
    m.fit(F[tr],y[tr]); return balanced_accuracy_score(y[te],m.predict(F[te]))

def main(n_bands,seeds):
    files=[f for f in sorted(glob.glob(os.path.join(DATA,"*.npz")))
           if np.load(f,allow_pickle=True)["X"].shape[1]==61]
    Xs,ys,sid,pos=[],[],[],None
    for i,f in enumerate(files):
        Xi,yi,p=load_subject(f); Xs.append(Xi); ys.append(yi); sid.append(np.full(len(yi),i))
        if pos is None: pos=p
    X=np.concatenate(Xs); y=np.concatenate(ys); sid=np.concatenate(sid); n_sub=len(files)
    V,w=laplacian_basis(pos)
    print("electrodes %d  laplacian eigenvalues %.3g .. %.3g"%(len(w),w[0],w[-1]))
    edges=np.linspace(0,61,n_bands+1).astype(int)
    print("pooled trials=%d subjects=%d bands=%d\n"%(len(X),n_sub,n_bands),flush=True)

    print("%-14s %10s %14s %10s"%("band(spatial)","within-subj","cross-subj RDM","shared?"))
    band_stats=[]
    Fb=[]
    for bi in range(n_bands):
        U=V[:,edges[bi]:edges[bi+1]]
        Z=np.einsum("ck,nct->nkt",U,X)
        F=band_features(Z)
        Fb.append(F)
        acc,rds=[],[]
        for s in range(n_sub):
            m=sid==s
            if m.sum()<60: continue
            a=[]
            for tr,te in StratifiedKFold(4,shuffle=True,random_state=0).split(F[m],y[m]):
                a.append(lda_acc(F[m],y[m],tr,te))
            acc.append(np.mean(a)); rds.append(rdm(F[m],y[m]))
        cross=[]
        for s in range(len(rds)):
            oth=np.mean([rds[o] for o in range(len(rds)) if o!=s],axis=0)
            cross.append(spearmanr(rds[s],oth).statistic)
        band_stats.append((np.mean(acc),np.mean(cross)))
        print("%-14s %10.3f %14.3f %10s"%("%d-%d"%(edges[bi],edges[bi+1]),
              np.mean(acc),np.mean(cross),"yes" if np.mean(cross)>0.15 else "no"),flush=True)

    cs=np.array([b[1] for b in band_stats])
    keep=np.where(cs>=np.median(cs))[0]
    print("\n  bands kept as shared: %s"%list(keep))
    Fall=np.hstack(Fb)
    Fshared=np.hstack([Fb[i] for i in keep])
    strat=np.array(["%d_%d"%(a,b) for a,b in zip(sid,y)])
    R={"pooled_allbands":[],"pooled_SHAREDbands":[],"persubj_allbands":[]}
    for sd in range(seeds):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=sd).split(X,strat):
            R["pooled_allbands"].append(lda_acc(Fall,y,tr,te))
            R["pooled_SHAREDbands"].append(lda_acc(Fshared,y,tr,te))
            per=[]
            for s in range(n_sub):
                mtr=tr[sid[tr]==s]; mte=te[sid[te]==s]
                if len(mte)<11 or len(mtr)<33: continue
                per.append(lda_acc(Fall,y,mtr,mte))
            R["persubj_allbands"].append(float(np.mean(per)))
            print("  fold  pooled %.3f  shared-only %.3f  per-subj %.3f"%(
                R["pooled_allbands"][-1],R["pooled_SHAREDbands"][-1],
                R["persubj_allbands"][-1]),flush=True)
    print("\n=== ph11 (chance 0.091) ===")
    base=np.array(R["pooled_allbands"])
    for k,v in R.items():
        a=np.array(v); _,p0=ttest_1samp(a,1/11); _,pb=ttest_rel(a,base)
        print("  %-22s %.3f  vs-chance p=%.7f  vs-pooled-all %+.3f (p=%.4f)"%(
            k,a.mean(),p0,a.mean()-base.mean(),pb))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--bands",type=int,default=6); ap.add_argument("--seeds",type=int,default=1)
    a=ap.parse_args(); main(a.bands,a.seeds)
