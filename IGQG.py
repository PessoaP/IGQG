from mpmath import mp,polylog,re,gamma,det,power,zeta
from numpy import zeros,array,ndarray
import warnings

mp.dps = 35; mp.pretty = True

#Adaptation of mp functions
def fpolylog(phi,x):
    if isinstance(x, ndarray):
        return array([fpolylog(phi,xx) for xx in x])
    return re(polylog(phi,x))

def fgamma(phi):
    return re(gamma(phi))

def polyexp(phi,x,eps=1e-8):
    if isinstance(x, ndarray):
        return array([polyexp(phi,xx,eps) for xx in x])
    if phi<=1:
        warnings.warn('Warning: order smaller than 1, not necessarily unique ')
    est,delta=x,1

    if x == 0:
        return x
    elif 0<x<zeta(phi):
        est = x/zeta(phi)
    elif x>zeta(phi):
        warnings.warn('Warning: value not in function domain')
        return None
    
    err = (fpolylog(phi,est)-x) 
    while abs(err)>min(eps/2,eps*(1-est)/2):
        dpl = fpolylog(phi-1,est)/est
        term=-err/dpl
        if (est+term>1):
            est = (est+1)/2
        else:
            est += term
        err = (fpolylog(phi,est)-x) 
        
    return est

#matrix functions
def A(phi,xi):
    #matA = [[fgamma(phi+3)*fpolylog(phi+2,xi),fgamma(phi+2)*fpolylog(phi+1,xi)],
    #        [fgamma(phi+2)*fpolylog(phi+1,xi),fgamma(phi+1)*fpolylog(phi,xi)]]
    #return det(matA)
    return fgamma(phi+3)*fpolylog(phi+2,xi)*fgamma(phi+1)*fpolylog(phi,xi)-(fgamma(phi+2)*fpolylog(phi+1,xi))**2

def B(phi,xi):
    li2,li1,li0,lin1=fpolylog(phi+2,xi),fpolylog(phi+1,xi),fpolylog(phi,xi),fpolylog(phi-1,xi)
    #matB = [[fgamma(phi+3)*li2,fgamma(phi+2)*li1,fgamma(phi+1)*li0],
    #        [fgamma(phi+4)*li2,fgamma(phi+3)*li1,fgamma(phi+2)*li0],
    #        [fgamma(phi+3)*li1,fgamma(phi+2)*li0,fgamma(phi+1)*lin1]]
    #return det(matB)
    return(fgamma(phi+3)*li2*fgamma(phi+3)*li1*fgamma(phi+1)*lin1
           +fgamma(phi+4)*li2*fgamma(phi+2)*li0*fgamma(phi+1)*li0
           +fgamma(phi+3)*li1*fgamma(phi+2)*li1*fgamma(phi+2)*li0
           -fgamma(phi+1)*li0*fgamma(phi+3)*li1*fgamma(phi+3)*li1
           -fgamma(phi+2)*li0*fgamma(phi+2)*li0*fgamma(phi+3)*li2
           -fgamma(phi+1)*lin1*fgamma(phi+2)*li1*fgamma(phi+4)*li2)

def Ac(phi,xi):
    #matA = [[fgamma(phi+3)*fpolylog(phi+2,xi),0],
    #        [fgamma(phi+2)*fpolylog(phi+1,xi),xi*power((1-xi),-2)]]
    #return det(matA)
    return fgamma(phi+3)*fpolylog(phi+2,xi)*xi*(1-xi)**(-2)

def Bc(phi,xi):
    li2,li1,li0,lin1=fpolylog(phi+2,xi),fpolylog(phi+1,xi),fpolylog(phi,xi),fpolylog(phi-1,xi)
    #matB = [[fgamma(phi+3)*fpolylog(phi+2,xi),fgamma(phi+2)*fpolylog(phi+1,xi),xi*power((1-xi),-2)],
    #        [fgamma(phi+4)*fpolylog(phi+2,xi),fgamma(phi+3)*fpolylog(phi+1,xi),0],
    #        [fgamma(phi+3)*fpolylog(phi+1,xi),fgamma(phi+2)*fpolylog(phi,xi),xi*(xi+1)*power((1-xi),-3)]]
    #return det(matB)
    return(fgamma(phi+3)*li2*fgamma(phi+3)*li1*xi*(xi+1)*((1-xi)**(-3))
          +fgamma(phi+4)*li2*fgamma(phi+2)*li0*xi*((1-xi)**(-2))
          -xi*((1-xi)**(-2))*fgamma(phi+3)*li1*fgamma(phi+3)*li1
          -xi*(xi+1)*((1-xi)**(-3))*fgamma(phi+2)*li1*fgamma(phi+4)*li2)
    
#Curvatures
def bose_Determinant(a,ac,beta,eta): #Calculates the *unitless* metric g_f for bosons (41)
    return a + ac*(beta**(eta+1))#ac*power(beta,(eta+1))

def fermi_Determinant(a): #somewhat pointless function, calculates *unitless* metric g_f for fermions (32)
    return a

def bose_Curvature(a,ac,b,bc,beta,eta): #Calculates the *unitless* curvature R_f for bosons (44)
    be= beta**(eta+1)#power(beta,(1+eta))
    num =  (b+be*bc)
    den = bose_Determinant(a,ac,beta,eta)**2
    return -num/den

def fermi_Curvature(a,b): #Calculates the *unitless* curvature R_f for fermions (32)
    return b/(a**2)

#xi inversion functions
def number(x,beta,eta,branch='be'):
    if branch == 'fd':
        return -gamma(eta+1)*power(beta,-eta-1)*fpolylog(eta+1,-x)
    if branch == 'beexcited':
        return gamma(eta+1)*power(beta,-eta-1)*fpolylog(eta+1,x)
    Ne = gamma(eta+1)*power(beta,-eta-1)*fpolylog(eta+1,x)
    N0 = x/(1-x)
    return N0+Ne

def dnumber(x,beta,eta,branch='be'):
    if branch == 'fd':
        return -gamma(eta+1)*power(beta,-eta-1)*fpolylog(eta,-x)/(x)
    if branch == 'beexcited':
        return gamma(eta+1)*power(beta,-eta-1)*fpolylog(eta,x)/x
    tNe = gamma(eta+1)*power(beta,-eta-1)*fpolylog(eta,x)/x
    tN0 = power(1-x,-2)
    return tN0+tNe

def xi(beta,eta,N,branch='be',est=1,epsilon=.1,method='Newton'):
    if isinstance(beta, ndarray):
        xiarray = [xi(beta[0],eta,N,branch,est,epsilon,method)]
        for i in range(1,len(beta)):
            xiarray.append(xi(beta[i],eta,N,branch,xiarray[-1],epsilon,method))
        return array(xiarray)
    
    if method == 'Bissec':
        return xi_bissec(beta,eta,N,branch,est,epsilon,method)
    
    if est==1:
        if branch == 'be':
            mino,majo=0,1
            est = 1-1/N
            
        elif branch =='fd':
            mino,majo = 0,-abs(mp.floor(abs(est)+1))
            while(N>number(majo,beta,eta,branch)):
                mino = majo
                majo*=2
                est = (majo+mino)/2

    while(abs(N-number(est,beta,eta,branch))>epsilon):
        term = -(number(est,beta,eta,branch)-N)/dnumber(est,beta,eta,branch)
        if (est+term>1):
            est = (est+1)/2
        else:
            est+=term
    return est

def xi_bissec(beta,eta,N,branch='be',est=1,epsilon=.1):
    if isinstance(beta, ndarray):
        xiarray = [xi(beta[0],eta,N,branch,est,epsilon)]
        for i in range(1,len(beta)):
            xiarray.append(xi(beta[i],eta,N,branch,xiarray[-1],epsilon))
        return array(xiarray)
    
    i = 0
    if branch == 'be':
        mino,majo = 0,1
        if est==1:
            est = 1-1/N
    elif branch =='fd':
        mino,majo = 0,mp.floor(est+1)
        while(N>number(majo,beta,eta,branch)):
            i+=1
            mino = majo
            majo*=2
            est = (majo+mino)/2

    while(abs(N-number(est,beta,eta,branch))>epsilon):
        i+=1
        if(N>number(est,beta,eta,branch)):
            mino = est
        else:
            majo = est
        est = (majo+mino)/2
    #print('Inversion for N={} and beta= {} -- attempts:{}'.format(int(N),beta,i))
    return est

def xiTL(betaratio,eta,branch='be',eps=1e-8):
    if isinstance(betaratio, ndarray):
        return array([xiTL(br,eta,branch,eps) for br in betaratio])
    if branch == 'fd':
        warnings.warn('Warning: Fermi-Dirac branch not implemented')
        return None
    if betaratio>=1:
        return 1
    return(polyexp(eta+1,zeta(eta+1)*betaratio**(eta+1),eps=eps))


#dxi dbeta
def dxidb(xiarray,beta,eta,branch='be'):
    num = gamma(eta+2)*(beta**(-eta-2))*fpolylog(eta+1,xiarray)
    den = gamma(eta+1)*(beta**(-eta-1))*fpolylog(eta,xiarray)/xiarray + (1-xiarray)**-2
    return num/den

def dxidbTL(xiarray,beta,eta,branch='be'):
    s= (branch == 'be') - (branch =='fd')
    return ((eta+1)/beta)*xiarray*(fpolylog(eta+1,s*xiarray)/fpolylog(eta,s*xiarray))

#BE finisher
def betaBE(eta,N,g):
    betacritical = power(gamma(eta+1)*(1/N)*zeta(eta+1),1/(eta+1))
    betas = (g+1)*betacritical
    return betas

def NgammaBEcurvature(eta,N,gam,xi_est=1,xi_epsilon=.1,return_xi=False):
    beta= betaBE(eta,N,gam)
    xiarray = xi(beta,eta,N,est=xi_est,epsilon=xi_epsilon)
    if(not isinstance(xiarray,ndarray)):
        xiarray=array([xiarray])
    
    aarray = A(eta,xiarray) 
    barray = B(eta,xiarray)
    acarray = Ac(eta,xiarray) 
    bcarray = Bc(eta,xiarray)
    if(isinstance(beta,ndarray)):
        prefactor = [power(bs,eta+1)/2 for bs in beta]
    else:
        #prefactor = (.5*(beta**(eta+1)))
        prefactor = power(beta,eta+1)/2
    curv = prefactor*bose_Curvature(aarray,acarray,barray,bcarray,beta,eta)
    if return_xi:
        return curv,xiarray[0]
    return curv

def estimategammamax(eta,N,mino,majo=0,delta=1e-8,xi_eps=.1,observe=False):
    if mino>majo:
        mino,majo=majo,mino
    golden = 1*mp.phi
    s,b = majo - ((majo - mino)/golden),mino + ((majo - mino)/golden)
    zs,xi_est = NgammaBEcurvature(eta,N,s,1,xi_eps,return_xi=True)
    zb = NgammaBEcurvature(eta,N,b,xi_est,xi_eps)
    gunc,est=1,1
    while (abs(gunc/est)>delta/2):
        if zs>zb:
            majo,b,est = b,s,s
            s = majo - ((majo - mino)/golden)
            #zb,zs= zs,NgammaBEcurvature(eta,N,s)
            zb=zs
            zs,xi_est = NgammaBEcurvature(eta,N,s,xi_est,xi_eps,return_xi=True)
        else:
            mino,s,est=s,b,b
            b=mino + ((majo - mino)/golden)
            #zb,zs= NgammaBEcurvature(eta,N,b),zb
            zs=zb
            zb,xi_est = NgammaBEcurvature(eta,N,b,xi_est,xi_eps,return_xi=True)
        gunc= abs(mino-majo)
        if observe:
            print(float(est),float(gunc))
    gunc= max(abs(est-majo),abs(est-mino))
    if observe:
        print(float(est),float(gunc))
    return est,NgammaBEcurvature(eta,N,est,xi_epsilon=xi_eps),float(gunc)

