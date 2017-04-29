import scipy as SP
import scipy.linalg as LA
import scipy.optimize as OPT
import pdb
import time
import numpy as NP

def nLLeval(ldelta,Uy,S):
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
    n_s,n_f = X.shape
    K = 1/rho*SP.dot(X,X.T) + SP.eye(n_s)
    U = LA.cholesky(K)
    return U



    
    
