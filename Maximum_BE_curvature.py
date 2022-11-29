from IGQG import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

mp.dps=200

def estimate_Rmax(mults):
    Na = 602214076*power(10,-8)*power(10,23)
    guess = -power(10,-8)
    df = pd.DataFrame(columns=['N/Na','gamma*','R*','gamma*_uncertainty'])
    for m in mults:
        a=estimategammamax(eta,m*Na,guess,-guess)
        guess = 10*a[0]
        df.loc[len(df)] = [m,float(a[0]),float(a[1]),float(a[2])]
    return df

mults = np.concatenate([mp.mpf(2)**np.arange(-10,11),mp.mpf(10)**np.arange(6,61,3)])

eta=mp.mpf(2)
df2 = estimate_Rmax(mults)
df2.to_csv('curvature_max-eta{}_orderNa.csv'.format(eta),index=False)

mults = np.concatenate([mp.mpf(2)**np.arange(0,21),mp.mpf(10)**np.arange(9,61,3)])

eta = power(2,-1)
df1 = estimate_Rmax(mults)
df1.to_csv('curvature_max-eta{}_orderNa.csv'.format(eta),index=False)