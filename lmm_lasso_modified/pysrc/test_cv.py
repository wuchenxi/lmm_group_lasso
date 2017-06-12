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
import lmm_lasso_admm_vb as lmm_lasso
import os

if __name__ == "__main__":

    # data directory
    data_dir = 'data'
    
    # load genotypes
    geno_filename = os.path.join(data_dir,'genotypes.csv')
    X = SP.genfromtxt(geno_filename,delimiter=',')
    X=X.T
    print X.shape
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
    print ypop.shape
    ypop = SP.reshape(ypop,(n_s,1))
    y = 0.3*ypop + 0.5*ypheno + 0.2*SP.random.randn(n_s,1)
    y = (y-y.mean())/y.std()
    
    # init
    debug = False
    n_train = 150
    n_test = n_s - n_train
    n_reps = 8
    f_subset = 0.5
    muinit = 0.01
    mu2init = 0.01
    group=[]
    for i in range(43):
        group+=[[i*50,i*50+50]]
    group+=[[2150,2196]]
        
    # split into training and testing
    train_idx = SP.random.permutation(SP.arange(n_s))
    test_idx = train_idx[n_train:]
    train_idx = train_idx[:n_train]

    # calculate kernel
    K = 1./n_f*SP.dot(X,X.T)


    # Glasso Parameter selection by 5 fold cv
    optmu=muinit
    optmu2=mu2init
    optcor=0
    for j1 in range(10):
        for j2 in range(10):
            mu=muinit*(2**j1)
            mu2=mu2init*(2**j2)
            corr1=0

            train1_idx=train_idx[:int(n_train*0.8)]
            train2_idx=train_idx[int(n_train*0.8):n_train]
            res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,mu2,group)
            w1=res1['weights']
            yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
            corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

            train1_idx=SP.concatenate((train_idx[:int(n_train*0.6)],train_idx[int(n_train*0.8):n_train]))
            train2_idx=train_idx[int(n_train*0.6):int(n_train*0.8)]
            res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,mu2,group)
            w1=res1['weights']
            yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
            corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

            train1_idx=SP.concatenate((train_idx[:int(n_train*0.4)],train_idx[int(n_train*0.6):n_train]))
            train2_idx=train_idx[int(n_train*0.4):int(n_train*0.6)]
            res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,mu2,group)
            w1=res1['weights']
            yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
            corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

            train1_idx=SP.concatenate((train_idx[:int(n_train*0.2)],train_idx[int(n_train*0.4):n_train]))
            train2_idx=train_idx[int(n_train*0.2):int(n_train*0.4)]
            res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,mu2,group)
            w1=res1['weights']
            yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
            corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

            train1_idx=train_idx[int(n_train*0.2):n_train]
            train2_idx=train_idx[:int(n_train*0.2)]
            res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,mu2,group)
            w1=res1['weights']
            yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
            corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

            print mu, mu2, corr1/5
            if corr1>optcor:
                optmu=mu
                optmu2=mu2
                
    print optmu, optmu2
    
    # train
    res = lmm_lasso.train(X[train_idx],K[train_idx][:,train_idx],y[train_idx],optmu,optmu2,group)
    w=res['weights']
    for i in range(100):
        print w[i*10:i*10+10], i*10+10
        
    # predict
    ldelta0 = res['ldelta0']
    yhat = lmm_lasso.predict(y[train_idx],X[train_idx,:],X[test_idx,:],K[train_idx][:,train_idx],K[test_idx][:,train_idx],ldelta0,w)
    corr = 1./n_test * ((yhat-yhat.mean())*(y[test_idx]-y[test_idx].mean())).sum()/(yhat.std()*y[test_idx].std())
    print corr

    # lasso parameter selection by 5 fold cv
    optmu0=muinit
    optcor=0
    for j1 in range(10):
        mu=muinit*(2**j1)
        corr1=0

        train1_idx=train_idx[:int(n_train*0.8)]
        train2_idx=train_idx[int(n_train*0.8):n_train]
        res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,0,[])
        w1=res1['weights']
        yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
        corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

        train1_idx=SP.concatenate((train_idx[:int(n_train*0.6)],train_idx[int(n_train*0.8):n_train]))
        train2_idx=train_idx[int(n_train*0.6):int(n_train*0.8)]
        res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,0,[])
        w1=res1['weights']
        yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
        corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

        train1_idx=SP.concatenate((train_idx[:int(n_train*0.4)],train_idx[int(n_train*0.6):n_train]))
        train2_idx=train_idx[int(n_train*0.4):int(n_train*0.6)]
        res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,0,[])
        w1=res1['weights']
        yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
        corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

        train1_idx=SP.concatenate((train_idx[:int(n_train*0.2)],train_idx[int(n_train*0.4):n_train]))
        train2_idx=train_idx[int(n_train*0.2):int(n_train*0.4)]
        res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,0,[])
        w1=res1['weights']
        yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
        corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

        train1_idx=train_idx[int(n_train*0.2):n_train]
        train2_idx=train_idx[:int(n_train*0.2)]
        res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,0,[])
        w1=res1['weights']
        yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
        corr1+=5./n_train*((yhat1-yhat1.mean())*(y[train2_idx]-y[train2_idx].mean())).sum()/(yhat1.std()*y[train2_idx].std())

        print mu, corr1/5
        if corr1>optcor:
            optmu0=mu
                
    print optmu0

    # train
    res = lmm_lasso.train(X[train_idx],K[train_idx][:,train_idx],y[train_idx],optmu0,0,[])
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
    ss = lmm_lasso.stability_selection(X,K,y,optmu,optmu2,group,n_reps,f_subset)

    sserr1=0
    sserr2=0
    for i in range(n_f):
        if i in idx:
            sserr1+=n_reps-ss[i]
        else:
            sserr2+=ss[i]
    # group not included
    ss2=lmm_lasso.stability_selection(X,K,y,optmu0,0,[],n_reps,f_subset)

    
    ss2err1=0
    ss2err2=0
    for i in range(n_f):
        if i in idx:
            ss2err1+=n_reps-ss2[i]
        else:
            ss2err2+=ss2[i]

    # Output
    
    for i in range(n_f):
        print i, (i in idx), ss[i], ss2[i]
    
    print sserr1, sserr2, ss2err1, ss2err2
