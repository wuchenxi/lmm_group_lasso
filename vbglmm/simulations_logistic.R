########## simulation 1 ##########
library(MASS)
library(Matrix)
library(glmnet)

#========== generate data ==============#

#remove(list=ls())
p = 5 		# number of fixed effects
m = 50		# number of clusters 
sigma2 = 0.5 

ni = 5
n = m*ni 
u = 1 
beta_true = c(3,-2.5,0,0,-2.5,rep(0,p-5))
Sig_true = sigma2*diag(u) 
	y = as.null()
	X = as.null()
	Z = matrix(rep(0,n*u*m),n) 
	id = c(rep(1:m, rep(ni, m)))
	for (i in 1:m) {
		if (u==1) {
			Zi = bdiag(rep(1,ni))
		} else {
			Zi = rep(1,ni) 		
			for (uu in 1:(u-1)) {Zi = cbind(Zi,runif(ni,0,1))}
		}
		Xi = rep(1,ni)
		for (ii in 1:(p-1)) {Xi = cbind(Xi,runif(ni,0,1))}
		bi = mvrnorm(1,rep(0,u),Sig_true) 
        	if (u==1) {
		  prob = exp(Xi%*%beta_true+bi)/(1+exp(Xi%*%beta_true+bi))	
		  yi = rbinom(ni,1, prob)
		} else {
		  prob = exp(Xi%*%beta_true+Zi%*%bi)/(1+exp(Xi%*%beta_true+Zi%*%bi))	
		  yi = rbinom(ni,1, prob)
		}
		X = rbind(X,Xi)
		Z[((i-1)*ni+1):(i*ni),((i-1)*u+1):(i*u)] = as.matrix(Zi) 	
		y = c(y,yi) 
	} 
	family = 'binomial'
	id = as.character(c(rep(1:m, rep(ni, m))))
	X = matrix(as.numeric(X[,-1]),n)
	data = data.frame(y,X,id)
#========== END generate data ==============#

# run vbglmm
out = balasso_glmm(y,X,Z,m,family='binomial')
out$beta
beta_true
