balasso_glmm = function(y,X,Z,m,family='poisson') {
# X must not contain the column 1
n = dim(X)[1]
X = cbind(rep(1,n),X)
p = dim(X)[2]
u = dim(Z)[2]/m

N = 100
alpha_q_r = 5; beta_q_r = 1; eta_r = c(alpha_q_r,beta_q_r)
alpha_0_r = 1; beta_0_r = 0;
nu0 = u+1
um = u*m
s = 1e-5
# initialize beta_q
if (n>p) {
	fit = glm(y~X[,2:p],family)
	beta_q = coef(fit)
} else {
	out = cv.glmnet(X[,2:p],y,intercept=T)
	fit.glm = glmnet(X[,2:p],y,family,intercept=T)
	beta_q = as.numeric(coef(fit.glm,s=out$lambda.min))
}

nu_q = nu0+m;
S_q = 0.001*diag(u)
mub = rep(0,um)
Sigb = 0.001*diag(um)

alpha_lamj = alpha_q_r/beta_q_r+1
beta_lamj = rep(sqrt(sum(beta_q^2)),p)

stop = F
count = 0
c = 1/sqrt(N)
while (!stop) {
    oldbeta_q = beta_q

       # update mub and Sigb	
    meanQ = nu_q*S_q; 
    meanQ = kronecker(diag(m),meanQ);
    Xbeta = X%*%beta_q
    b_old = mub
    stopb = F
    tol = 1e-4
    countb = 0
    while (!stopb) {
        eta = Xbeta + Z%*%b_old
        if (family == 'poisson') {zeta1 = exp(eta); zeta2 = zeta1}
	  if (family == 'binomial') {
		zeta1 = exp(eta)/(1+exp(eta));
		zeta2 = zeta1/(1+exp(eta))
	  }
        ub = t(Z)%*%(y - zeta1) - meanQ%*%b_old
        Hb = -t(Z)%*%diag(as.numeric(zeta2))%*%Z - meanQ  
        b_new = b_old - solve(Hb)%*%ub
        countb = countb + 1    
        if ((countb > 500)|(sqrt(sum((b_new - b_old)^2)) < tol)) { bstar = b_new; stopb = T} else b_old = b_new
    }
    etastar = Xbeta + Z%*%bstar
    if (family == 'poisson') zeta2 = exp(etastar)
    if (family == 'binomial') zeta2 = exp(etastar)/(1+exp(etastar))^2
    mub = bstar
    Sigb = solve(t(Z)%*%diag(as.numeric(zeta2))%*%Z + meanQ)
    
    # update S_q

    S_q = 0.001*diag(u)
    for (q in 1:m) {
        indexx = ((q-1)*u+1):(q*u)
        S_q = S_q + mub[indexx]%*%t(mub[indexx]) + Sigb[indexx,indexx]
    }
    S_q = solve(S_q)

    # update beta_q
    lamj = alpha_lamj/beta_lamj
    lamj[1] = 0				#not penalize beta_0	
    beta = beta_q	
    stop_CGD = F
    count_CGD = 0;	
    Zmub = Z%*%mub
    ZSigbZ = 1/2*diag(Z%*%Sigb%*%t(Z))
   
    while (!stop_CGD) {
	beta_old = beta
	for (j in 1:p) {
      	eta = X%*%beta+Zmub
		if (family == 'poisson') {
			aux = eta+ZSigbZ
			fdotj = sum(X[,j]*(exp(aux)-y))
			f2dotj = sum(X[,j]^2*exp(aux))
		}
		if (family == 'binomial') {
			exp_eta = exp(eta)
			aux = -1/(1+exp_eta)+3/(1+exp_eta)^2-2/(1+exp_eta)^3
			aux = ZSigbZ*aux+exp_eta/(1+exp_eta)
			fdotj = sum(X[,j]*(aux-y))
			aux = exp_eta/(1+exp_eta)^2-6*exp_eta/(1+exp_eta)^3+6*exp_eta/(1+exp_eta)^4
			aux = exp_eta/(1+exp_eta)^2+ZSigbZ*aux
			f2dotj = sum(X[,j]^2*aux)
		}
    		aux = sort(c((lamj[j]-fdotj)/f2dotj,-beta[j],(-lamj[j]-fdotj)/f2dotj))
            dj = aux[2]
		Delta = dj*fdotj+lamj[j]*(abs(beta[j]+dj)-abs(beta[j]))

            if (family == 'poisson') target_beta = sum(exp(eta+ZSigbZ))-sum(y*eta)
		if (family == 'binomial') target_beta = sum(log(1+exp_eta)+ZSigbZ*exp_eta/(1+exp_eta)^2)-sum(y*eta)
		l = 0
		stop_l = F
		while (!stop_l) {
			aj = 0.5^l
			
                  eta_new = eta+aj*dj*X[,j]
			if (family == 'poisson') {
				target_beta_new = sum(exp(eta_new+ZSigbZ))-sum(y*eta_new)
			}
			if (family == 'binomial') {
				exp_eta_new = exp(eta_new)
				target_beta_new = sum(log(1+exp_eta_new)+ZSigbZ*exp_eta_new/(1+exp_eta_new)^2)-sum(y*eta_new)
			}
			aux = target_beta_new-target_beta
			aux = aux+lamj[j]*(abs(beta[j]+aj*dj)-abs(beta[j]))
			aux = as.numeric(aux-aj*0.1*Delta)

			if ((is.na(aux))|((aux<=0)|(l>100))) stop_l = T else l=l+1
		}

            beta[j] = beta[j]+aj*dj 
	}
      count_CGD = count_CGD+1 
   	if ((!is.na(match(NA,beta)))|((count_CGD>50)|(sqrt(sum((beta_old-beta)^2))<1e-3))) stop_CGD = T
    }
    if (is.na(match(NA,beta))) beta_q = beta 
	
    if (count %% 2 == 0) {
	   
	    S = 100;
	    C = matrix(c(trigamma(alpha_q_r),-1/beta_q_r,-1/beta_q_r,alpha_q_r/beta_q_r/beta_q_r),2)
	    p_x_y = function(x) {(p*log(s)-beta_0_r+sum(digamma(alpha_lamj)-log(beta_lamj)))*x-p*lgamma(x)+(alpha_0_r-1)*log(x)}
	    T_x = function(x) {matrix(c(log(x),-x),2,byrow=T)}
	    r_star1 =  rgamma(S,shape=alpha_q_r,scale=1/beta_q_r)
          r_star2 =  rgamma(S,shape=alpha_q_r,scale=1/beta_q_r)
          r_star1[r_star1 <= 0] = 1e-20
          r_star2[r_star2 <= 0] = 1e-20
          aux1 = T_x(r_star1)-T_x(r_star2)
	    aux2 = (p_x_y(r_star1)-p_x_y(r_star2))
          aux2 = rbind(aux2,aux2) 
	    aux = aux1*aux2
	    g = rowMeans(aux)/2
	    Cbar = 0;
	    gbar = 0;
	    for (t in 1:N) {
            #Ctilde = C+Lam_guard; g_tilde = g+Lam_guard%*%eta_r
            #eta_r = solve(Ctilde,g_tilde)
            eta_r = solve(C,g)
            alpha_q_r = max(1e-2,eta_r[1])
            beta_q_r = max(1e-2,eta_r[2])
       	C_t_hat = matrix(c(trigamma(alpha_q_r),-1/beta_q_r,-1/beta_q_r,alpha_q_r/beta_q_r/beta_q_r),2)
	      r_star1 =  rgamma(S,shape=alpha_q_r,scale=1/beta_q_r)
            r_star2 =  rgamma(S,shape=alpha_q_r,scale=1/beta_q_r)
            r_star1[r_star1 <= 0] = 1e-20
            r_star2[r_star2 <= 0] = 1e-20

            aux1 = T_x(r_star1)-T_x(r_star2)
    		aux2 = (p_x_y(r_star1)-p_x_y(r_star2))
            aux2 = rbind(aux2,aux2) 
    		aux = aux1*aux2
    		g_t_hat = rowMeans(aux)/2
            g = (1-c)*g+c*g_t_hat
            C = (1-c)*C+c*C_t_hat
            if (t>N/2) {gbar = gbar+g_t_hat; Cbar = Cbar+C_t_hat}
  	    }

          eta_r = solve(Cbar,gbar)
 	    alpha_q_r = max(1e-2,eta_r[1])
	    beta_q_r = max(1e-2,eta_r[2])

    }
    alpha_lamj = alpha_q_r/beta_q_r+1
    beta_lamj = abs(beta_q)+s
    count = count+1
    if ((count>500) | (sqrt(sum((oldbeta_q-beta_q)^2))<1e-4)) stop = T
}
fit = list(beta=beta_q,Sigma=solve(S_q)/(nu_q - u - 1),lambda=lamj[2:p],iter=count)
}


