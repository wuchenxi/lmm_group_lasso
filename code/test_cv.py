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
import scipy.linalg as LA
import pdb
import lmm_lasso_pg as lmm_lasso
import os

# load genotypes
X = SP.genfromtxt("geno.csv",delimiter=',')
[n_f,n_s] = X.shape
for i in xrange(n_f):
    X[i]=(X[i]-(X[i]).mean())/(X[i]).std()
X=X.T
print X.shape

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
ypop = SP.genfromtxt('pheno.csv')
print ypop.shape

ypop = SP.reshape(ypop,(n_s,1))
y = 0.3*ypop + 0.5*ypheno + 0.2*SP.random.randn(n_s,1)
y = (y-y.mean())/y.std()

# init
debug = False
n_train = 150
n_test = n_s - n_train
n_reps = 5
f_subset = 0.5

muinit = 0.1
mu2init = 0.1
ps_step = 1.5

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
opterr=500*n_s
for j1 in range(10):
    for j2 in range(10):
        mu=muinit*(ps_step**j1)
        mu2=mu2init*(ps_step**j2)
        err=0
        for k in range(5):
            train1_idx=SP.concatenate((train_idx[:int(n_train*k*0.2)],train_idx[int(n_train*(k+1)*0.8):n_train]))
            train2_idx=train_idx[int(n_train*k*0.2):int(n_train*(k+1)*0.2)]
            res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,mu2,group)
            w1=res1['weights']
            yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
            err+=LA.norm(yhat1-y[train2_idx])**2
        
        print mu, mu2, err
        if err<opterr:
            opterr=err
            optmu=mu
            optmu2=mu2
            
print optmu, optmu2, opterr

# train
res = lmm_lasso.train(X[train_idx],K[train_idx][:,train_idx],y[train_idx],optmu,optmu2,group)
w=res['weights']
    
# predict
ldelta0 = res['ldelta0']
yhat = lmm_lasso.predict(y[train_idx],X[train_idx,:],X[test_idx,:],K[train_idx][:,train_idx],K[test_idx][:,train_idx],ldelta0,w)
corr = 1./n_test * ((yhat-yhat.mean())*(y[test_idx]-y[test_idx].mean())).sum()/(yhat.std()*y[test_idx].std())
print corr

# lasso parameter selection by 5 fold cv
optmu0=muinit
optcor=0
for j1 in range(10):
    mu=muinit*(1.5**j1)
    err=0
    for k in range(5):
        train1_idx=SP.concatenate((train_idx[:int(n_train*k*0.2)],train_idx[int(n_train*(k+1)*0.8):n_train]))
        train2_idx=train_idx[int(n_train*k*0.2):int(n_train*(k+1)*0.2)]
        res1=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,0,[])
        w1=res1['weights']
        yhat1 = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[train2_idx,:],K[train1_idx][:,train1_idx],K[train2_idx][:,train1_idx],res1['ldelta0'],w1)
        err+=LA.norm(yhat1-y[train2_idx])**2
    
    print mu, err/5
    if err<opterr:
        opterr=err
        optmu0=mu
            
print optmu0

# train
res = lmm_lasso.train(X[train_idx],K[train_idx][:,train_idx],y[train_idx],optmu0,0,[])
w=res['weights']
   
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
        
# group info not included
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
