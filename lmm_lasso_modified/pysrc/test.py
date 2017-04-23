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
import lmm_lasso
import os

if __name__ == "__main__":

    # data directory
    data_dir = 'data'
    
    # load genotypes
    geno_filename = os.path.join(data_dir,'genotypes.csv')
    X = SP.genfromtxt(geno_filename)
    [n_s,n_f] = X.shape

    # simulate phenotype
    SP.random.seed(1)
    n_c = 5
    idx = SP.random.randint(0,n_f,n_c)
    w = 1./n_c * SP.ones((n_c,1))
    ypheno = SP.dot(X[:,idx],w)
    ypheno = (ypheno-ypheno.mean())/ypheno.std()
    ypop2 = SP.sum(X[:,[]+range(2,6)+range(8,14)+range(67,79)+range(201,204)],axis=1)
    ypop2 = SP.reshape(ypop2,(n_s,1))
    y = 0.3*ypop2 + 0.5*ypheno + 0.2*SP.random.randn(n_s,1)
    y = (y-y.mean())/y.std()
    
    # init
    debug = False
    n_train = 150
    n_test = n_s - n_train
    n_reps = 20
    f_subset = 0.5
    mu = 8
    mu2 = 2
    group=[]
    for i in range(100):
        group+=[([]+range(i*10,i*10+10))]
        
    # split into training and testing
    train_idx = SP.random.permutation(SP.arange(n_s))
    test_idx = train_idx[n_train:]
    train_idx = train_idx[:n_train]

    # calculate kernel
    K = 1./n_f*SP.dot(X,X.T)
    
    # train
    res = lmm_lasso.train(X[train_idx],K[train_idx][:,train_idx],y[train_idx],mu,mu2,group,debug=debug)
    w=res['weights']
    wp=[ele for row in w for ele in row]
    for i in range(100):
        print wp[i*10:i*10+10], i*10+10
        
    # predict
    ldelta0 = res['ldelta0']
    yhat = lmm_lasso.predict(y[train_idx],X[train_idx,:],X[test_idx,:],K[train_idx][:,train_idx],K[test_idx][:,train_idx],ldelta0,w)
    corr = 1./n_test * ((yhat-yhat.mean())*(y[test_idx]-y[test_idx].mean())).sum()/(yhat.std()*y[test_idx].std())
    print corr

    # stability selection
    ss = lmm_lasso.stability_selection(X,K,y,mu,mu2,group,n_reps,f_subset)
    ssp = [int(f*5) for f in ss]
    for i in range(100):
        print ssp[i*10:i*10+10], i*10+10    

