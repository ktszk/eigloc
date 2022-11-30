import numpy as np
import json
import scipy.linalg as sclin
import matplotlib.pyplot as plt

Ry2eV=13.6056931
data=[np.array(f.split()) for f in open('lmaxloc2','r').readlines() if ('<wannier i|r|wannier j>' in f)]
tmp0=np.array([[int(d) for d in dt[[5,6]]] for dt in data])
tmp=np.array([[float(d) for d in dt[[8,10,13,15,18,20]]] for dt in data])
no=tmp0.max()
lenup=int(len(tmp0)*.5)

data=[f.split() for f in open('Hopping.up','r').readlines() if ('0     0     0    0.000000    0.000000    0.000000' in f)]
hamu=np.array([float(d[8])+1j*float(d[9]) for d in data]).reshape(no,no)
data=[f.split() for f in open('Hopping.dn','r').readlines() if ('0     0     0    0.000000    0.000000    0.000000' in f)]
hamd=np.array([float(d[8])+1j*float(d[9]) for d in data]).reshape(no,no)

def gen_uni(no):
    lorb=int((no-1)*.5)
    uni0=np.zeros((no,no),dtype=complex)
    for i in range(no):
        if i==lorb:
            uni0[i,i]=1.
        elif i<lorb:
            uni0[i,i]=-1j/np.sqrt(2.)
            uni0[i,no-i-1]=1/np.sqrt(2.)
        else:
            uni0[i,i]=(-1.)**((i-1)%2)/np.sqrt(2.)
            uni0[i,no-1-i]=(-1)**((i-1)%2)*1j/np.sqrt(2.)
    return uni0

#rxu=np.zeros((no,no),dtype=np.complex)
#ryu=np.zeros((no,no),dtype=np.complex)
#rzu=np.zeros((no,no),dtype=np.complex)
unid=gen_uni(5)
unif=gen_uni(7)

ru=np.zeros((no,no),dtype=np.complex)
for ((i,j),tp) in zip(tmp0[:lenup],tmp):
    ru[i-1,j-1]=tp[0]+1j*tp[1]+tp[2]+1j*tp[3]+tp[4]+1j*tp[5]
    #rxu[i-1,j-1]=tp[0]+1j*tp[1]
    #ryu[i-1,j-1]=tp[2]+1j*tp[3]
    #rzu[i-1,j-1]=tp[4]+1j*tp[5]
#rxd=np.zeros((no,no),dtype=np.complex)
#ryd=np.zeros((no,no),dtype=np.complex)
#rzd=np.zeros((no,no),dtype=np.complex)
ruorb=unid.dot((ru[:5,5:]).dot(unif.T.conjugate()))

rd=np.zeros((no,no),dtype=np.complex)
for ((i,j),tp) in zip(tmp0[lenup:],tmp[lenup:]):
    rd[i-1,j-1]=tp[0]+1j*tp[1]+tp[2]+1j*tp[3]+tp[4]+1j*tp[5]
    #rxd[i-1,j-1]=tp[0]+1j*tp[1]
    #ryd[i-1,j-1]=tp[2]+1j*tp[3]
    #rzd[i-1,j-1]=tp[4]+1j*tp[5]
rdorb=unid.dot((rd[:5,5:]).dot(unif.T.conjugate()))

hamdf_u=unid.dot((hamu[:5,5:]).dot(unif.T.conjugate()))
hamdf_d=unid.dot((hamd[:5,5:]).dot(unif.T.conjugate()))


rdf=np.concatenate([ruorb.sum(axis=0),rdorb.mean(axis=0)])
hdf=np.concatenate([hamdf_u.sum(axis=0),hamdf_d.mean(axis=0)])
print(rdf)
print(hdf)

out_dic_av={"rdf":{"real":rdf.real.tolist(),"imag":rdf.imag.tolist()},
            "tdf":{"real":hdf.real.tolist(),"imag":hdf.imag.tolist()}}

out_dic={"rdfu":{"real":ruorb.real.tolist(),"imag":ruorb.imag.tolist()},
         "rdfd":{"real":rdorb.real.tolist(),"imag":rdorb.imag.tolist()},
         "tdfu":{"real":hamdf_u.real.tolist(),"imag":hamdf_u.imag.tolist()},
         "tdfd":{"real":hamdf_d.real.tolist(),"imag":hamdf_d.imag.tolist()}}

with open("rt.json",'w') as files:
    json.dump(out_dic,files,indent=2)
with open("rta.json",'w') as files:
    json.dump(out_dic_av,files,indent=2)
