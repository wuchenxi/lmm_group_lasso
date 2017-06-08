"""
GLasso with Proximal Gradient
"""

import scipy as SP
import scipy.linalg as LA
import scipy.optimize as OPT
import pdb
import matplotlib.pylab as PLT
import time
import numpy as NP


def stability_selection(X,K,y,mu,mu2,group,n_reps,f_subset,**kwargs):
    """
    run stability selection

    Input:
    X: SNP matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty

    n_reps:   number of repetitions
    f_subset: fraction of datasets that is used for creating one bootstrap

    output:
    selection frequency for all SNPs: n_f x 1
    """
    time_start = time.time()
    [n_s,n_f] = X.shape
    n_subsample = int(SP.ceil(f_subset * n_s))
    freq = SP.zeros(n_f)
    
    for i in range(n_reps):
        print 'Iteration %d'%i
        idx = SP.random.permutation(n_s)[:n_subsample]
        res = train(X[idx],K[idx][:,idx],y[idx],mu,mu2,group,**kwargs)
        snp_idx = (res['weights']!=0).flatten()
        freq[snp_idx] += 1
        
    time_end = time.time()
    time_diff = time_end - time_start
    print 'time: %.2fs'%(time_diff)
    return freq

def train(X,K,y,mu,mu2,group=[[0,1],[2,3,4]],numintervals=100,ldeltamin=-5,ldeltamax=5,rho=1,alpha=1,debug=False):
    """
    train linear mixed model lasso

    Input:
    X: SNP matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    rho: augmented Lagrangian parameter for Lasso solver
    alpha: over-relatation parameter (typically ranges between 1.0 and 1.8) for Lasso solver

    Output:
    results
    """
    
    time_start = time.time()
    [n_s,n_f] = X.shape
    assert X.shape[0]==y.shape[0], 'dimensions do not match'
    assert K.shape[0]==K.shape[1], 'dimensions do not match'
    assert K.shape[0]==X.shape[0], 'dimensions do not match'
    if y.ndim==1:
        y = SP.reshape(y,(n_s,1))

    # train null model
    S,U,ldelta0,monitor_nm = train_nullmodel(y,K,numintervals,ldeltamin,ldeltamax,debug=debug)
    
    # train lasso on residuals
    delta0 = SP.exp(ldelta0)
    Sdi = 1./(S+delta0)
    Sdi_sqrt = SP.sqrt(Sdi)
    SUX = SP.dot(U.T,X)
    SUX = SUX * SP.tile(Sdi_sqrt,(n_f,1)).T
    SUy = SP.dot(U.T,y)
    SUy = SUy * SP.reshape(Sdi_sqrt,(n_s,1))
    
    w= train_lasso(SUX,SUy,mu,mu2,group,rho,alpha,debug=debug)

    time_end = time.time()
    time_diff = time_end - time_start
    print 'time %.2fs'%(time_diff)

    res = {}
    res['ldelta0'] = ldelta0
    res['weights'] = w
    res['time'] = time_diff
    res['monitor_nm'] = monitor_nm
    return res


def predict(y_t,X_t,X_v,K_tt,K_vt,ldelta,w):
    """
    predict the phenotype

    Input:
    y_t: phenotype: n_train x 1
    X_t: SNP matrix: n_train x n_f
    X_v: SNP matrix: n_val x n_f
    K_tt: kinship matrix: n_train x n_train
    K_vt: kinship matrix: n_val  x n_train
    ldelta: kernel parameter
    w: lasso weights: n_f x 1

    Output:
    y_v: predicted phenotype: n_val x 1
    """
    assert y_t.shape[0]==X_t.shape[0], 'dimensions do not match'
    assert y_t.shape[0]==K_tt.shape[0], 'dimensions do not match'
    assert y_t.shape[0]==K_tt.shape[1], 'dimensions do not match'
    assert y_t.shape[0]==K_vt.shape[1], 'dimensions do not match'
    assert X_v.shape[0]==K_vt.shape[0], 'dimensions do not match'
    assert X_t.shape[1]==X_v.shape[1], 'dimensions do not match'
    assert X_t.shape[1]==w.shape[0], 'dimensions do not match'
    
    [n_train,n_f] = X_t.shape
    n_test = X_v.shape[0]
    
    if y_t.ndim==1:
        y_t = SP.reshape(y_t,(n_train,1))
    if w.ndim==1:
        w = SP.reshape(w,(n_f,1))
    
    delta = SP.exp(ldelta)
    idx = w.nonzero()[0]

    if idx.shape[0]==0:
        return SP.dot(K_vt,LA.solve(K_tt + delta*SP.eye(n_train),y_t))
        
    y_v = SP.dot(X_v[:,idx],w[idx]) + SP.dot(K_vt, LA.solve(K_tt + delta*SP.eye(n_train),y_t-SP.dot(X_t[:,idx],w[idx])))
    return y_v

"""
helper functions
"""

def train_lasso(X,y,mu,mu2,group,rho=1,alpha=1,max_iter=1000,abstol=1E-4,reltol=1E-2,zero_threshold=1E-3,debug=False):
    """
    train lasso via PG:
    min_w  0.5*sum((y-Xw)**2) + mu*|z| + mu2*|z|_2
    
    Input:
    X: design matrix: n_s x n_f
    y: outcome:  n_s x 1
    mu: l1-penalty parameter
    rho: augmented Lagrangian parameter
    alpha: over-relatation parameter (typically ranges between 1.0 and 1.8)
    """
    if debug:
        print '... train lasso'

    # init
    [n_s,n_f] = X.shape
    w = SP.zeros((n_f,1))
    wold = w
    curval=0.5*((SP.dot(X,w)-y)**2).sum()
    t=1
    tt=1
    tto=0
    for i in range(max_iter):
        alf=(tto-1.0)/tt
        v=(1+alf)*w-alf*wold
        
        curval=0.5*((SP.dot(X,v)-y)**2).sum()
        grad=SP.dot(X.transpose(),(y-SP.dot(X,v)))
        wn=v+t*grad
        wn=soft_thresholding(wn,mu*t,mu2*t,group)
        newval=0.5*((SP.dot(X,wn)-y)**2).sum()
        while True:
            testls=SP.dot((t*grad+0.5*(v-wn))[:,0],
                          (v-wn)[:,0])
            if testls+t*(curval-newval)>=0:
                break
            t*=0.9;
            wn=v+t*grad
            wn=soft_thresholding(wn,mu*t,mu2*t,group)
            newval=0.5*((SP.dot(X,wn)-y)**2).sum()
        #print t
        if LA.norm(v-wn)<zero_threshold*t:
            break
        wold=w
        w=wn

        tto=tt
        tt=0.5*(1+(1+4*tt**2)**0.5)
    print i
    
    w[SP.absolute(w)<zero_threshold]=0

    return w


    

def nLLeval(ldelta,Uy,S):
    """
    evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K+deltaI) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.

    Uy: transformed outcome: n_s x 1
    S:  eigenvectors of K: n_s
    ldelta: log-transformed ratio sigma_gg/sigma_ee
    """
    n_s = Uy.shape[0]
    delta = SP.exp(ldelta)
    
    # evaluate log determinant
    Sd = S+delta
    ldet = SP.sum(SP.log(Sd))

    # evaluate the variance    
    Sdi = 1.0/Sd
    Uy = Uy.flatten()
    ss = 1./n_s * (Uy*Uy*Sdi).sum()

    # evalue the negative log likelihood
    nLL=0.5*(n_s*SP.log(2.0*SP.pi)+ldet+n_s+n_s*SP.log(ss));

    return nLL


def train_nullmodel(y,K,numintervals=100,ldeltamin=-5,ldeltamax=5,debug=False):
    """
    train random effects model:
    min_{delta}  1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    
    Input:
    X: SNP matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    """
    if debug:
        print '... train null model'
        
    n_s = y.shape[0]

    # rotate data
    S,U = LA.eigh(K)
    Uy = SP.dot(U.T,y)

    # grid search
    nllgrid=SP.ones(numintervals+1)*SP.inf
    ldeltagrid=SP.arange(numintervals+1)/(numintervals*1.0)*(ldeltamax-ldeltamin)+ldeltamin
    nllmin=SP.inf
    for i in SP.arange(numintervals+1):
        nllgrid[i]=nLLeval(ldeltagrid[i],Uy,S);
        
    # find minimum
    nll_min = nllgrid.min()
    ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

    # more accurate search around the minimum of the grid search
    for i in SP.arange(numintervals-1)+1:
        if (nllgrid[i]<nllgrid[i-1] and nllgrid[i]<nllgrid[i+1]):
            ldeltaopt,nllopt,iter,funcalls = OPT.brent(nLLeval,(Uy,S),(ldeltagrid[i-1],ldeltagrid[i],ldeltagrid[i+1]),full_output=True);
            if nllopt<nllmin:
                nllmin=nllopt;
                ldeltaopt_glob=ldeltaopt;

    monitor = {}
    monitor['nllgrid'] = nllgrid
    monitor['ldeltagrid'] = ldeltagrid
    monitor['ldeltaopt'] = ldeltaopt_glob
    monitor['nllopt'] = nllmin
    
    return S,U,ldeltaopt_glob,monitor
 

def factor(X,rho):
    """
    computes cholesky factorization of the kernel K = 1/rho*XX^T + I

    Input:
    X design matrix: n_s x n_f (we assume n_s << n_f)
    rho: regularizaer

    Output:
    L  lower triangular matrix
    U  upper triangular matrix
    """
    n_s,n_f = X.shape
    K = 1/rho*SP.dot(X,X.T) + SP.eye(n_s)
    U = LA.cholesky(K)
    return U


def soft_thresholding(w,kappa,kappa2,gp):
    """
    Performs elementwise soft thresholding for each entry w_i of the vector w:
    s_i= argmin_{s_i}  rho*abs(s_i) + rho/2*(x_i-s_i) **2
    by using subdifferential calculus

    Input:
    w vector nx1
    kappa regularizer

    Output:
    s vector nx1
    """
    n_f = w.shape[0]
    zeros = SP.zeros((n_f,1))
    s = NP.max(SP.hstack((w-kappa,zeros)),axis=1) - NP.max(SP.hstack((-w-kappa,zeros)),axis=1)
    for i in xrange(len(gp)):
        vgn=SP.dot(s[gp[i][0]:gp[i][1]],s[gp[i][0]:gp[i][1]])**0.5#this is 2 times faster than LA.norm 
        if vgn>kappa2:
            ratio=(vgn-kappa2)/vgn
        else:
            ratio=0
        s[gp[i][0]:gp[i][1]]=ratio*s[gp[i][0]:gp[i][1]]
    s = SP.reshape(s,(n_f,1))
    return s


def lasso_obj(X,y,w,mu,mu2,gp):
    """
    evaluates lasso objective: 0.5*sum((y-Xw)**2) + mu*|z|

    Input:
    X: design matrix: n_s x n_f
    y: outcome:  n_s x 1
    mu: l1-penalty parameter
    w: weights: n_f x 1
    z: slack variables: n_fx1

    Output:
    obj
    """
    r=0.5*((SP.dot(X,w)-y)**2).sum() + mu*SP.absolute(w).sum()
    for group in gp:
        r+=LA.norm(w[group[0]:group[1]])*mu2
    return r


    
    
