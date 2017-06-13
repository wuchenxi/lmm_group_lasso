%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% six city data
library(MASS)
library(glmnet)
library(Matrix)

#remove(list = ls())
data = read.table('sixcity.txt');
N = 2148;
u = 2;
p = 3;
m = 536;
y = as.null();
X = as.null();
Z = 0;
yi = data[1,2];
Xi = as.numeric(data[1,4:5]);
m = 1;
for (i in 2:N) {
    if (data[i,3]==data[i-1,3]) {
        yi = c(yi,data[i,2])
        Xi = rbind(Xi,data[i,4:5]);
    } else {
        y = c(y,yi);
        X = rbind(X,Xi);
	  Zi = cbind(rep(1,length(yi)),Xi[,1])	
        Z = bdiag(Z,Zi);
        m = m+1;
        yi = data[i,2];
        Xi = data[i,4:5];
    }
}
y = c(y,yi);
X = rbind(X,Xi);
Zi = cbind(rep(1,length(yi)),Xi[,1])	
Z = bdiag(Z,Zi);
X = as.matrix(X)
Z = as.matrix(Z)
Z = Z[2:2149,2:1075]


# glmmvb
ptm <- proc.time()
out = balasso_glmm(y,X,Z,m,family='binomial')
time2 = proc.time() - ptm
out$beta
