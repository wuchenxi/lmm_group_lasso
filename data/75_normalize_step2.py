#Normalize genotype and combine it with fake phenotype

import csv
import scipy as SP
import pdb
import os
import lmm_lasso

def normalize(l):
    count={'A':0,'T':0,'G':0,'C':0}
    for c in l:
        count[c]+=1
    dft=max(count,key=count.get)
    r=[]
    for c in l:
        if c==dft:
            r.append(0.0)
        else:
            r.append(1.0)
    arr=SP.array(r)
    return (arr-arr.mean())/arr.std()

if __name__ == "__main__":

    # load genotypes
    X = SP.genfromtxt('75.csv',delimiter=',',dtype=None)
    # load leaf number phenotype
    X1 = SP.genfromtxt('ln10.tsv', delimiter='\t',dtype=None)
    pheno=(X[1]).tolist()
    for c in range(2,len(pheno)):
        for r in X1:
            if int(pheno[c])==int(r[0]):
                pheno[c]=r[1]
        if type(pheno[c])==str:
            pheno[c]=0
        
    #normalize and output phenotype            
    Y=[pheno[2:]]
    for i in range(2,X.shape[0]):
        Y.append(normalize(X[i][2:]))
    with open("genotype75.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(Y[1:])
    nf=len(Y)-1
    print nf

    #obtain genotype & phenotype for samples with complete phenotype
    MY=SP.array(Y).transpose()
    RMY=MY[MY[:,0]>0]
    RY=RMY[:,0]
    RY=(RY-RY.mean())/RY.std()
    RX=RMY[:,1:]

    #train null model for these samples
    COR=1./nf*SP.dot(RX,RX.transpose())
    res=lmm_lasso.train_nullmodel(RY,COR)
    delta=SP.exp(res[2])
    print delta

    #get fake phenotype
    FX=MY[:,1:]
    FCOR=1./nf*SP.dot(FX,FX.transpose())
    D=SP.diag(SP.array([delta]*len(Y[1])))
    FY=SP.random.multivariate_normal(SP.array([0]*len(Y[1])),SP.add(FCOR,D))
    FY=(FY-FY.mean())/FY.std()
    FY=SP.array([FY])
    with open("phenotype75.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(FY.transpose())

    #validate fake phenotype, that it has similar delta as we start with
    res=lmm_lasso.train_nullmodel(FY.transpose(),FCOR)
    delta=SP.exp(res[2])
    print delta
