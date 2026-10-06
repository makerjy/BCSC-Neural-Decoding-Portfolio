"""
Unified space-time SPD matrix on the npz2 pipeline.

    Z(t) = [x(t); x(t-tau); ...; x(t-(p-1)tau); 1]
    C    = (1/T) sum_t Z(t) Z(t)^T        (no centring, so the constant row keeps the mean)

    diagonal blocks  -> spatial covariance
    off-diagonal     -> cross-lag structure (dynamics)
    last row/col     -> evoked waveform, which a centred covariance discards

One point on one manifold per trial: space, dynamics and evoked deflection interact inside a
single object rather than being computed separately and concatenated.

The binding constraint is timepoints per trial, not trials: C is estimated from T samples for
a (pd+1)-square matrix. npz2 gives T=256, so pd+1 must stay near or below ~40.
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
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

DATA="npz2"; PHON=["a","i","e","u","o","s","z","p","b","t","d"]; STOPS=["b","p","d","t"]
WMIN,WMAX=0.0,1.01; BASE=(-0.2,-0.11); K_PLACE,K_PH11=2,6

def load_subject(path):
    d=np.load(path,allow_pickle=True); X,t=d["X"].astype(float),d["times"]
    meta=pd.DataFrame({c:[str(v) for v in d[c]] for c in ["task","tms_target","phoneme1"]})
    keep=((meta.task=="single_phoneme")&(meta.tms_target=="control")).values
    bm=(t>=BASE[0])&(t<=BASE[1]); wm=(t>=WMIN)&(t<=WMAX); Xk=X[keep]
    base=Xk[:,:,bm].mean(2,keepdims=True) if bm.any() else 0.0
    return (Xk-base)[:,:,wm], meta.phoneme1.values[keep]

def build_task(Xb,ph,task):
    if task=="place":
        m=np.isin(ph,STOPS); return Xb[m],np.where(np.isin(ph[m],["d","t"]),1,0),K_PLACE,0.5
    m=np.isin(ph,PHON); return Xb[m],np.array([PHON.index(v) for v in ph[m]]),K_PH11,1/11

def proj(Xtr,Xte,d):
    pc=PCA(min(d,len(Xtr)-2)).fit(Xtr.transpose(0,2,1).reshape(-1,Xtr.shape[1]))
    f=lambda Z: np.stack([pc.transform(z.T).T for z in Z],0); return f(Xtr),f(Xte)

def unified(A,p,tau,aug,shrink=0.08):
    T=A.shape[2]
    if p>1:
        span=(p-1)*tau; A=np.concatenate([A[:,:,i*tau:T-span+i*tau] for i in range(p)],1)
    n,k,T2=A.shape
    if aug:
        A=np.concatenate([A,np.ones((n,1,T2))],1); k+=1
        C=np.einsum("nit,njt->nij",A,A)/T2
    else:
        Ac=A-A.mean(2,keepdims=True); C=np.einsum("nit,njt->nij",Ac,Ac)/T2
    tr=np.trace(C,axis1=1,axis2=2)[:,None,None]/k
    return (1-shrink)*C+shrink*tr*np.eye(k)

def tang(Ca,Cb):
    ts=TangentSpace(metric="riemann").fit(Ca); return ts.transform(Ca),ts.transform(Cb)

def baseline_parts(Xtr,Xte,K,d=20):
    A,B=proj(Xtr,Xte,d); ed=np.linspace(0,A.shape[2],K+1).astype(int)
    ta,tb,ma,mb=[],[],[],[]
    for i in range(K):
        sa,sb=A[:,:,ed[i]:ed[i+1]],B[:,:,ed[i]:ed[i+1]]
        Ca=Covariances("oas").fit_transform(sa); Cb=Covariances("oas").fit_transform(sb)
        a,b=tang(Ca,Cb); ta.append(a); tb.append(b); ma.append(sa.mean(2)); mb.append(sb.mean(2))
    f=lambda L: np.hstack([x.reshape(len(x),-1) for x in L])
    return f(ta),f(tb),f(ma),f(mb)

def ev(Ftr,ytr,Fte,yte):
    m=make_pipeline(StandardScaler(),LDA(solver="lsqr",shrinkage="auto"))
    m.fit(Ftr,ytr); return balanced_accuracy_score(yte,m.predict(Fte))

def cat(bt,be):
    sc=[StandardScaler().fit(x) for x in bt]
    return (np.hstack([s.transform(x) for s,x in zip(sc,bt)]),
            np.hstack([s.transform(x) for s,x in zip(sc,be)]))

GRID=[(20,2,False),(13,3,False),(20,3,False)]
FIXED=(20,2,False)
VAR=["base_cov+mean"]+["U_d%d_p%d_cen"%(d,p) for d,p,_ in GRID]+["base+U_FIXED"]

def run_subject(args):
    path,task,seeds,tau=args
    Xb,ph=load_subject(path); X,y,K,_=build_task(Xb,ph,task)
    acc={v:[] for v in VAR}
    for sd in range(seeds):
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=sd).split(X,y):
            ca,cb,ma,mb=baseline_parts(X[tr],X[te],K)
            a,b=cat([ca,ma],[cb,mb]); acc["base_cov+mean"].append(ev(a,y[tr],b,y[te]))
            fixed=None
            for (d,p,aug) in GRID:
                A,B=proj(X[tr],X[te],d)
                Ca=unified(A,p,tau,aug); Cb=unified(B,p,tau,aug)
                ua,ub=tang(Ca,Cb)
                acc["U_d%d_p%d_cen"%(d,p)].append(ev(ua,y[tr],ub,y[te]))
                if (d,p,aug)==FIXED: fixed=(ua,ub)
            a,b=cat([ca,ma,fixed[0]],[cb,mb,fixed[1]])
            acc["base+U_FIXED"].append(ev(a,y[tr],b,y[te]))
    return os.path.basename(path).split("_")[0],{v:float(np.mean(acc[v])) for v in VAR}

def main(task,seeds,workers,tau):
    files=[f for f in sorted(glob.glob(os.path.join(DATA,"*.npz")))
           if np.load(f,allow_pickle=True)["X"].shape[1]==61]
    chance=0.5 if task=="place" else 1/11
    print("task=%s subjects=%d tau=%d"%(task,len(files),tau)); print("  "+"  ".join(VAR)+chr(10),flush=True)
    out={v:[] for v in VAR}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nm,sc in ex.map(run_subject,[(f,task,seeds,tau) for f in files]):
            for v in VAR: out[v].append(sc[v])
            print("  %-6s %s"%(nm,"  ".join("%.3f"%sc[v] for v in VAR)),flush=True)
    print(chr(10)+"=== %s (chance %.3f) ==="%(task,chance))
    base=np.array(out["base_cov+mean"])
    for v in VAR:
        a=np.array(out[v]); _,p0=ttest_1samp(a,chance); _,pb=ttest_rel(a,base)
        print("  %-18s %.3f  vs-chance p=%.7f  vs-base %+.3f (p=%.4f) %s"%(
            v,a.mean(),p0,a.mean()-base.mean(),pb,
            "**BEATS**" if pb<0.05 and a.mean()>base.mean() else ""))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",default="ph11"); ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--workers",type=int,default=8); ap.add_argument("--tau",type=int,default=20)
    a=ap.parse_args(); main(a.task,a.seeds,a.workers,a.tau)
