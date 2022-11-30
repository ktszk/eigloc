import numpy as np
import json
import scipy.linalg as sclin
import matplotlib.pyplot as plt

Ry2eV=13.6056931
data=[f for f in open('lsocmat','r').readlines() if ('hammso real' in f)]
no=int(np.sqrt(len(data)))
tmp2=[[float(d) for d in np.array(dt.split())[[7,9,12,14,17,19]]] for dt in data]
hso_tmp=np.array([[dp[0]+1j*dp[1],dp[2]+1j*dp[3],dp[4]+1j*dp[5]] for dp in tmp2]).T*Ry2eV

lorb=int((no-1)*.5)
uni0=np.zeros((no,no),dtype=complex) #unitary complex2real harmonics
for i in range(no):
    if i==lorb:
        uni0[i,i]=1.
    elif i<lorb:
        uni0[i,i]=-1j/np.sqrt(2.)
        uni0[i,no-i-1]=1/np.sqrt(2.)
    else:
        uni0[i,i]=(-1.)**((i-1)%2)/np.sqrt(2.)
        uni0[i,no-1-i]=(-1)**((i-1)%2)*1j/np.sqrt(2.)
hammso_up=uni0.dot((hso_tmp[0].reshape(no,no)).dot(uni0.T.conjugate()))
hammso_dn=uni0.dot((hso_tmp[1].reshape(no,no)).dot(uni0.T.conjugate()))
hammso_pm=uni0.dot((hso_tmp[2].reshape(no,no)).dot(uni0.T.conjugate()))
#hammso_up=hso_tmp[0].reshape(no,no)
#hammso_dn=hso_tmp[1].reshape(no,no)
#hammso_pm=np.zeros((no,no))
#hammso_pm=hso_tmp[2].reshape(no,no)
#print(hammso_up.round(3))
#print(hammso_dn.round(3))
#print(hammso_pm.round(3))
hammso=np.hstack((np.vstack((hammso_up,hammso_pm)),np.vstack((hammso_pm.T.conjugate(),hammso_dn))))
(eig,uni)=sclin.eigh(hammso)
az=(eig**2).sum()
#xi=np.sqrt(az/14)
xi=np.sqrt(az/42)
print(eig)
print(np.diag(hammso))
print(xi)
for hso in hammso:
    for hs in hso.round(3):
        print('%6.3f %6.3fj,'%(hs.real,hs.imag),end='')
    else:
        print('')
for hso in abs(uni)**2:
    for hs in hso.round(3):
        print('%6.3f,'%hs,end='')
    else:
        print('')

out_dic={"hammsoc":{"real":hammso.real.tolist(),"imag":hammso.imag.tolist()}}
with open("dipole.json",'w') as files:
    json.dump(out_dic,files,indent=2)

