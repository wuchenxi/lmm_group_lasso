"""
test.py

Author:		Barbara Rakitsch
Year:		2012
Group:		Machine Learning and Computational Biology Group (http://webdav.tuebingen.mpg.de/u/karsten/group/)
Institutes:	Max Planck Institute for Developmental Biology and Max Planck Institute for Intelligent Systems (72076 Tuebingen, Germany)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import csv
import scipy as SP
import pdb
import lmm_lasso_pg as lmm_lasso
import os

if __name__ == "__main__":

    # data directory
    data_dir = 'data'
    
    # load genotypes
    geno_filename = os.path.join(data_dir,'genotypes.csv')
    X = SP.genfromtxt(geno_filename)
    [n_s,n_f] = X.shape

    # simulate phenotype
    idx=[]
    for i in range(4):
        idxx=int(SP.rand()*19)*50
        for j in range(10):
            idx+=[idxx+int(SP.rand()*50)]
    print idx
    
    ypheno = SP.sum(X[:,idx],axis=1)
    ypheno = SP.reshape(ypheno,(n_s,1))
    ypheno = (ypheno-ypheno.mean())/ypheno.std()
    pheno_filename = os.path.join(data_dir,'poppheno.csv')
    ypop = SP.genfromtxt(pheno_filename)
    ypop = SP.reshape(ypop,(n_s,1))
    y = 0.3*ypop + 0.5*ypheno# + 0.2*SP.random.randn(n_s,1)
    y = (y-y.mean())/y.std()
    
    # init
    debug = False
    n_train = 150
    n_test = n_s - n_train
    n_reps = 5
    f_subset = 0.5
    mu = 5.0
    mu2 = 2.0
    group=[]
    for i in range(20):
        group+=[[i*50,i*50+50]]
        
    # split into training and testing
    train_idx = SP.random.permutation(SP.arange(n_s))
    test_idx = train_idx[n_train:]
    train_idx = train_idx[:n_train]

    # calculate kernel
    K = 1./n_f*SP.dot(X,X.T)

    # Parameter selection
    
    # train
    res = lmm_lasso.train(X[train_idx],K[train_idx][:,train_idx],y[train_idx],mu,mu2,group)
    w=res['weights']
    for i in range(100):
        print w[i*10:i*10+10], i*10+10
        
    # predict
    ldelta0 = res['ldelta0']
    yhat = lmm_lasso.predict(y[train_idx],X[train_idx,:],X[test_idx,:],K[train_idx][:,train_idx],K[test_idx][:,train_idx],ldelta0,w)
    corr = 1./n_test * ((yhat-yhat.mean())*(y[test_idx]-y[test_idx].mean())).sum()/(yhat.std()*y[test_idx].std())
    print corr

    # stability selection
    # group info included
    ss = lmm_lasso.stability_selection(X,K,y,mu,mu2,group,n_reps,f_subset)
    for i in range(100):
        print ss[i*10:i*10+10], i*10+10

    sserr1=0
    sserr2=0
    for i in range(1000):
        if i in idx:
            sserr1+=n_reps-ss[i]
        else:
            sserr2+=ss[i]
    # group not included
    ss2=lmm_lasso.stability_selection(X,K,y,mu,mu2,[],n_reps,f_subset)
    for i in range(100):
        print ss2[i*10:i*10+10], i*10+10    
    
    ss2err1=0
    ss2err2=0
    for i in range(1000):
        if i in idx:
            ss2err1+=n_reps-ss2[i]
        else:
            ss2err2+=ss2[i]
    print sserr1, sserr2, ss2err1, ss2err2
