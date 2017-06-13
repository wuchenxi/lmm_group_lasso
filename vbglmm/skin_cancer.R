library(MASS)
library(glmnet)
library(Matrix)

#remove(list=ls())
skin_cancer = read.table('skin_cancer.txt',header=TRUE)
n = 7081;
ni = 5;
u = 1;
p = 5;
m = 1683;
y = as.null();
X = as.null();
Z = 0;
yi = skin_cancer[1,7];
Xi = as.numeric(c(skin_cancer[1,3:6],skin_cancer[1,8:9]));
m = 1;
for (i in 2:n) {
    if (skin_cancer[i,9]>skin_cancer[i-1,9]) {
        yi = c(yi,skin_cancer[i,7]);
        Xi = rbind(Xi,as.numeric(c(skin_cancer[i,3:6],skin_cancer[i,8:9])));
    } else {
        y = c(y,yi);
        X = rbind(X,Xi);
	Zi = rep(1,length(yi));
	Z = bdiag(Z,Zi)
        m = m+1;
        yi = skin_cancer[i,7];
	Xi = as.numeric(c(skin_cancer[i,3:6],skin_cancer[i,8:9]));
    }
}
y = c(y,yi);
X = rbind(X,Xi);
X = as.matrix(X);
Zi = rep(1,length(yi));
Z = bdiag(Z,Zi)
Z = Z[2:7082,2:1684]
#%======== end importing data ============%

# glmmvb
ptm <- proc.time()
out = balasso_glmm(y,X,Z,m,family='poisson')
time2 = proc.time() - ptm
out$beta


