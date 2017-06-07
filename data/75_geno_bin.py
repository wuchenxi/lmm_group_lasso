#Parse genotypes

import csv
import scipy as SP
import pdb
import os

def normalize(l):
    count={'A':0,'T':0,'G':0,'C':0}
    for c in l:
        count[c]+=1
    dft=max(count,key=count.get)
    r=[]
    for c in l:
        if c==dft:
            r.append(0)
        else:
            r.append(1)
    arr=SP.array(r)
    return arr#(arr-arr.mean())/arr.std()

if __name__ == "__main__":

    # load genotypes
    X = SP.genfromtxt('75.csv',delimiter=',',dtype=None)
        
    #parse and output genotypes
    Y=[]
    for i in range(2,X.shape[0]):
        Y.append(normalize(X[i][2:]))
    with open("genotype75_bin.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(Y[:])
    nf=len(Y)
    print nf
