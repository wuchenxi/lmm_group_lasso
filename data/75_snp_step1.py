#Picking SNP

import csv
import scipy as SP
import pdb
import os

if __name__ == "__main__":

    # load genotypes
    X = SP.genfromtxt('call_method_75_TAIR9.csv',delimiter=',',dtype=None)
    Y = (X[0:2]).tolist()
    SP.random.seed(84)
    for i in range(2,X.shape[0]):
        if SP.rand()<0.01 and not('-' in X[i].tolist()):
            Y.append(X[i])
    with open("75.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(Y)
    print len(Y)
    
