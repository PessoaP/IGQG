from mpmath import mp,polylog,re,gamma,det,power
from numpy import zeros
mp.dps = 35; mp.pretty = True

def fpolylog(phi,x):
	return re(polylog(phi,x))
def fgamma(phi):
	return re(gamma(phi))

def A(phi,xi):
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