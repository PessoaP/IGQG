from mpmath import mp,polylog,re,gamma,det,power,zeta
from numpy import zeros,array,ndarray
mp.dps = 35; mp.pretty = True

#Adaptation of mp functions
def fpolylog(phi,x):
    if isinstance(x, ndarray):
        return array([fpolylog(phi,xx) for xx in x])
    return re(polylog(phi,x))

def fgamma(phi):
    return re(gamma(phi))

#matrix functions
def A(phi,xi):
    #print('here')
    matA = [[fgamma(phi+3)*fpolylog(phi+2,xi),fgamma(phi+2)*fpolylog(phi+1,xi)],
            [fgamma(phi+2)*fpolylog(phi+1,xi),fgamma(phi+1)*fpolylog(phi,xi)]]
    return det(matA)

def B(phi,xi):
    matB = [[fgamma(phi+3)*fpolylog(phi+2,xi),fgamma(phi+2)*fpolylog(phi+1,xi),fgamma(phi+1)*fpolylog(phi,xi)],
            [fgamma(phi+4)*fpolylog(phi+2,xi),fgamma(phi+3)*fpolylog(phi+1,xi),fgamma(phi+2)*fpolylog(phi,xi)],
            [fgamma(phi+3)*fpolylog(phi+1,xi),fgamma(phi+2)*fpolylog(phi,xi),fgamma(phi+1)*fpolylog(phi-1,xi)]]
    #print(matB)
    return det(matB)

def Ac(phi,xi):
    #matA = [[fgamma(phi+3)*fpolylog(phi+2,xi),0],
    #        [fgamma(phi+2)*fpolylog(phi+1,xi),xi*power((1-xi),-2)]]
    #return linalg.det(matA)
    return fgamma(phi+3)*fpolylog(phi+2,xi)*xi*power((1-xi),-2)

def Bc(phi,xi):
    matB = [[fgamma(phi+3)*fpolylog(phi+2,xi),fgamma(phi+2)*fpolylog(phi+1,xi),xi*power((1-xi),-2)],
            [fgamma(phi+4)*fpolylog(phi+2,xi),fgamma(phi+3)*fpolylog(phi+1,xi),0],
            [fgamma(phi+3)*fpolylog(phi+1,xi),fgamma(phi+2)*fpolylog(phi,xi),xi*(xi+1)*power((1-xi),-3)]]
    #print(matB)
    return det(matB)

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

def xi(beta,eta,N,branch='be',est=1,epsilon=.1):
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

def xiTL(betaratio,eta,branch='be',epsilon=1e-6):
    if isinstance(betaratio, ndarray):
        return array([xiTL(br,eta,branch,epsilon) for br in betaratio])
    if betaratio>=1:
        return 1
    mino,majo = 0,1
    i=0
    est = mp.mpf(betaratio)
    while (abs(betaratio**(eta+1)-fpolylog(eta+1,est)/zeta(eta+1))>epsilon):
        if(betaratio**(eta+1)>fpolylog(eta+1,est)/zeta(eta+1)):
            mino = est
        else:
            majo = est
        est = (majo+mino)/2
        i+=1
    #print('Inversion for TL and beta= {} -- attempts:{}'.format(betaratio,i))
    return est

