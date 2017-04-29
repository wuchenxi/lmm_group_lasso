#Picking SNP

import csv
import scipy as SP
import pdb
import os

if __name__ == "__main__":

    # load genotypes
    geno_filename = 'call_method_32.csv'
    X = SP.genfromtxt(geno_filename,delimiter=',',dtype=None)
    Y = (X[0:2]).tolist()
    for i in range(2,X.shape[0]):
        if SP.rand()<0.01 and not('-' in X[i].tolist()):
            Y.append(X[i])
    with open("32.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(Y)
    print len(Y)
    
