"""
"""

import csv
import scipy as SP
import scipy.linalg as LA
import pdb
import lmm_lasso_pg_vb as lmm_lasso
import os

# load genotypes
X = SP.array(list(csv.reader(open('geno.csv','rb'),
                             delimiter=','))).astype(float)
[n_f,n_s] = X.shape
for i in xrange(n_f):
    m=X[i].mean()
    std=X[i].std()
    X[i]=(X[i]-m)/std
X = X.T
print X.shape

# simulate phenotype
idx=[]
for i in range(4):
    idxx=int(SP.rand()*20)*50
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
n_train = int(n_s*0.7)
n_test = n_s - n_train
n_reps = 5
f_subset = 0.7

muinit = 10
mu2init = 10

    
# split into training and testing
train_idx = SP.random.permutation(SP.arange(n_s))
test_idx = train_idx[n_train:]
train_idx = train_idx[:n_train]

# calculate kernel
# the first 2622 SNP are in the first chromosome which we are testing
chro=2622 
XO = X[:,2622:]
X = X[:,:2622]
K = 1./(n_f-chro)*SP.dot(XO,XO.T)
n_f = chro
group=[]
for i in range(20):
    group+=[[i*50,i*50+50]]
group+=[[2000,n_f]]


# Glasso Parameter selection by 5 fold cv
optmu=muinit
optmu2=mu2init

# train
res = lmm_lasso.train(X[train_idx],K[train_idx][:,train_idx],y[train_idx],optmu,optmu2,group)
w = res['weights']
    
# predict
ldelta0 = res['ldelta0']
yhat = lmm_lasso.predict(y[train_idx],X[train_idx,:],X[test_idx,:],K[train_idx][:,train_idx],K[test_idx][:,train_idx],ldelta0,w)
corr = 1./n_test * SP.dot(yhat.T-yhat.mean(),y[test_idx]-y[test_idx].mean())/(yhat.std()*y[test_idx].std())
print corr[0,0]

# lasso parameter selection by 5 fold cv
optmu0=muinit

# train
res = lmm_lasso.train(X[train_idx],K[train_idx][:,train_idx],y[train_idx],optmu0,0,[])
w=res['weights']
   
# predict
ldelta0 = res['ldelta0']
yhat = lmm_lasso.predict(y[train_idx],X[train_idx,:],X[test_idx,:],K[train_idx][:,train_idx],K[test_idx][:,train_idx],ldelta0,w)
corr = 1./n_test * SP.dot(yhat.T-yhat.mean(),y[test_idx]-y[test_idx].mean())/(yhat.std()*y[test_idx].std())
print corr[0,0]    


# stability selection
# group info included
ss = lmm_lasso.stability_selection(X,K,y,optmu,optmu2,group,n_reps,f_subset)

sserr1=0
sserr2=0
for i in range(n_f):
    if i in idx:
        if ss[i]<n_reps*0.7:
            sserr1+=1
    else:
        if ss[i]>=n_reps*0.7:
            sserr2+=1
            
# group info not included
ss2=lmm_lasso.stability_selection(X,K,y,optmu0,0,[],n_reps,f_subset)

ss2err1=0
ss2err2=0
for i in range(n_f):
    if i in idx:
        if ss2[i]<n_reps*0.7:
            ss2err1+=1
    else:
        if ss2[i]>=n_reps*0.7:
            ss2err2+=1

# Output

for i in range(n_f):
    print i, (i in idx), ss[i], ss2[i]

print optmu,optmu2,optmu0
print sserr1, sserr2, ss2err1, ss2err2