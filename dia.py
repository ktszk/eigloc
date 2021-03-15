import numpy as np
import scipy as sc
import scipy.linalg as sl
import scipy.special as scsp
import scipy.optimize as scopt
import itertools as itts
import matplotlib.pyplot as plt
import get_ham
#from numba import jit

ns=14 #f-orbitals
ne=6 #filling

zeta= 0.191651

F0=12.55229
Up= 5.30021e-2
B40=1.92436e-3
B60=3.91589e-5
#Eu3+ ofelt
#Up  = 0.04972
#zeta= 0.16366
#Tb3+ ofelt
#Up  = 0.05381
#zeta= 0.21139
#init_n=[1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.]
init_n=[0.9784,0.9796,0.9801,0.9805,0.9805,0.9802,0.0000,
        0.    ,0.    ,0.0001,0.0001,0.0001,0.    ,0.0000]
#init_n=[.5,1.,1.,1.,1.,1.,.5,0.,0.,0.,0.,0.,0.,0.]
sub_num=111

cf_type=1
sw_F_type=0

if sw_F_type in {0,1}:
    Up2=0
    Up3=0
else:
    try:
        Up2
        Up3
    except NameError:
        Up2=0
        Up3=0
try:
    B40
    B60
except NameError:
    print('B40,B60 set zero')
    B40=0.0
    B60=0.0
try:
    B20
    B66
except NameError:
    print('B20,B66 set zero')
    B20=0.0
    B66=0.0

if ne==6:
    n_fcs=49
else:
    n_fcs=2

sw_conv=False
sw_conv_cf=True
sw_full=True
sw_spec=False
sp=np.array(['-3,1','-2,1','-1,1','0,1','1,1','2,1','3,1','-3,-1','-2,-1','-1,-1','0,-1','1,-1','2,-1','3,-1'])
sp1=np.array([[-3,1],[-2,1],[-1,1],[0,1],[1,1],[2,1],[3,1],[-3,-1],[-2,-1],[-1,-1],[0,-1],[1,-1],[2,-1],[3,-1]])
erange=4.0
#temp=1.0e02
temp=2.6e-2 #~300K
idelta=1.e-4

def get_F(F_type,E0,E1,E2,E3):
    if F_type==0:
        F=np.array([0.,1.*225,0.138*1089,0.0151*7361.64])*abs(E1)
        F[0]=abs(E0)
    elif F_type==1:
        F=np.array([abs(E0)+9.*abs(E1)/7.,
                    75.*(abs(E1)+143.*abs(E2)+11.*abs(E3))/14.,
                    99.*(abs(E1)-130.*abs(E2)+4.*abs(E3))/7.,
                    5577.*(abs(E1)+35.*abs(E2)-7.*abs(E3))/350.])
    elif F_type==2:
        F=abs(np.array([E0,1.*225*E1,0.138*1089*E2,0.0151*7361.64*E3]))
    else:
        F=abs(np.array([E0,E1,E2,E3]))
    return F

def gen_hop():
    def import_Hopping(fname):
        tmp=[f.split() for f in open(fname,'r')]
        tmp=sc.array([complex(float(tp[8]),float(tp[9])) for tp in tmp])
        ham_r=sc.reshape(tmp,(ns//2,ns//2))
        return(ham_r)

    hopu=import_Hopping('hopu.dat')
    hopd=import_Hopping('hopd.dat')
    hoppm=np.zeros((ns//2,ns//2))
    tmp=np.hstack((hopu,hoppm))
    tmp2=np.hstack((hoppm,hopd))
    hop=np.vstack((tmp,tmp2))
    return(hop)

def gen_hop_free(zeta,Blm,sw_ls=True):
    #soc
    (B40,B60,B20,B66)=Blm
    mmax=.5*(ns//2-1)
    lsdiag=np.diag(np.array([(l-mmax) for l in range(ns//2)]))*.5
    lspm=np.zeros((ns//2,ns//2))
    if sw_ls: #if False, consider only lzsz
        for l in range(ns//2-1):
            m=l+1-mmax
            #print(np.sqrt((mmax+m)*(mmax-m+1))*.5,mmax,l-mmax)
            lspm[l,l+1]=np.sqrt((mmax+m)*(mmax-m+1))*.5
    tmp=np.hstack((lsdiag,lspm))
    tmp2=np.hstack((lspm.T,-lsdiag))
    hopsoc=np.vstack((tmp,tmp2))*zeta    
    #print(np.round(lspm,4))
    #print(lsdiag)
    #cf
    if cf_type!=0:
        #make unitary matrix j2lm
        uni=np.zeros((14,14))
        for i in range(6):
            uni[i,13-i]=np.sqrt(6.-i)
            uni[i,5-i]=-np.sqrt(i+1.)
        for i in range(7):
            uni[i+7,13-i]=np.sqrt(i+1.)
            uni[i+6,6-i]=np.sqrt(7.-i)
        uni=uni/np.sqrt(7)
        #make Hcf from j basis stevens op.
        if cf_type in {1,2}:
            O4j=np.diag([ 1.,-3., 2., 2.,-3., 1., 7.,-13.,-3.,  9.,  9.,-3.,-13., 7.])
            O6j=np.diag([ 0., 0., 0., 0., 0., 0.,1.,-5.,9.,-5.,-5.,9.,-5.,1.])
        if cf_type==1:
            #O44,j5/2
            O4j[0,4]=np.sqrt(5.)
            O4j[4,0]=O4j[0,4]
            O4j[1,5]=O4j[0,4]
            O4j[5,1]=O4j[1,5]
            #O44,j7/2
            O4j[6,10]=np.sqrt(35.)
            O4j[10,6]=O4j[6,10]
            O4j[13,9]=O4j[6,10]
            O4j[9,13]=O4j[13,9]
            O4j[7,11]=5.*np.sqrt(3)
            O4j[11,7]=O4j[7,11]
            O4j[8,12]=O4j[7,11]
            O4j[12,8]=O4j[8,12]
            #O64,j7/2
            O6j[6,10]=-3.*np.sqrt(35.)
            O6j[10,6]=O6j[6,10]
            O6j[13,9]=O6j[6,10]
            O6j[9,13]=O6j[13,9]
            O6j[7,11]=7.*np.sqrt(3)
            O6j[11,7]=O6j[7,11]
            O6j[8,12]=O6j[7,11]
            O6j[12,8]=O6j[8,12]
            hopcf=uni.T.conjugate().dot((B40*O4j+21.*B60*O6j).dot(uni))
        elif cf_type==2:
            O2j=np.diag([10.,-2.,-8.,-8.,-2.,10.,21.,3.,-9.,-15.,-15.,-9.,3.,21.])
            O66j=np.zeros((ns,ns))
            O66j[6,12]=np.sqrt(7.)
            O66j[12,6]=O66j[6,12]
            O66j[7,13]=O66j[6,12]
            O66j[13,7]=O66j[7,13]
            hopcf=uni.T.conjugate().dot((B20*O2j/60.+B40*O4j+21.*B60*O6j+4.*B66*O66j).dot(uni))
        hop=hopsoc+hopcf
        #print(hopcf.round(3))
    else:
        hop=hopsoc
    return(hop)

def get_HF(ham0,U,J,temp=1.0e-9,eps=1.0e-6,itemax=1000,switch=True):
    ini_n=np.array(init_n)
    n1=np.diag(ini_n)
    for k in range(itemax):
        ham_hub=np.zeros((ns,ns))
        for i in range(ns//2):
            ham_hub[i,i]=((U[i,:]*n1.diagonal()[ns//2:]).sum()
                          +np.delete((U[i,:]-J[i,:])*n1.diagonal()[:ns//2],i).sum())
            ham_hub[i+ns//2,i+ns//2]=((U[i,:]*n1.diagonal()[:ns//2]).sum()
                                      +np.delete((U[i,:]-J[i,:])*n1.diagonal()[ns//2:],i).sum())
        for i in range(ns//2):
            for j in range(i+1,ns//2):
                ham_hub[i,j]=J[i,j]*n1[j+ns//2,i+ns//2]*.5
                ham_hub[i+ns//2,j+ns//2]=J[i,j]*n1[j,i]*.5
                ham_hub[j,i]=ham_hub[i,j].conjugate()
                ham_hub[j+ns//2,i+ns//2]=ham_hub[i+ns//2,j+ns//2].conjugate()
                ham_hub[i,i+ns//2]=ham_hub[i,i+ns//2]-J[i,j]*n1[j+ns//2,j]*.5
                ham_hub[i+ns//2,i]=ham_hub[i,i+ns//2]-J[i,j]*n1[j,j+ns//2]*.5
        ham=ham0+ham_hub
        (eig,uni)=sl.eigh(ham)
        f=lambda mu: ne+.5*(np.tanh(0.5*(eig-mu)/temp)-1.).sum()
        mu=scopt.brentq(f,eig.min(),eig.max())
        #mu=scopt.newton(f,0.5*(eig.min()+eig.max()))
        n0=.5-.5*np.tanh(0.5*(eig-mu)/temp)
        new_n=uni.dot(np.diag(n0).dot(uni.T.conjugate()))
        dn=abs(new_n-n1).sum()/abs(new_n).sum()
        if dn<eps:
            L,S=0,0
            for i,j in enumerate(new_n.diagonal()):
                L=L+(i-3)*j
                S=S+(.5 if i<ns//2 else -.5)*j
            if switch:
                print('converged loop %d'%k)
                print(L.round(4),S.round(4),(L+S).round(4))
                np.set_printoptions(linewidth=500)
                #print(new_n.round(3))
            break
        else:
            n1=new_n
    else:
        if switch:
            print('no converged')
            np.set_printoptions(linewidth=500)
            print(new_n.round(3))
    return(ham-mu*np.identity(ns))

def plot_hamHF(hop,U,J,F,temp=1.0e-9):
    if sw_full:
        ham=get_ham.get_HF_full(ns,ne,init_n,hop,U,J,F,temp)
    else:
        ham=get_HF(hop,U,J,temp)
    (eig,uni)=sl.eigh(ham)
    #print(ham.round(3))
    print((eig).round(3))
    hop2=gen_hop()
    #print(hop2.real.round(2))
    (eig2,uni)=sl.eigh(hop2)
    f2=lambda mu: ne+.5*(np.tanh(0.5*(eig2-mu)/temp)-1.).sum()
    mu2=scopt.brentq(f2,eig2.min(),eig2.max())
    print((eig2-mu2).round(3))
    #print((uni**2).round(3))
    plt.scatter([0]*ns,eig,marker='_')
    plt.scatter([0]*ns,eig2-mu2,marker='_',color='red')
    plt.show()

def ham_conv(F0,Up,Up2,Up3,zeta,B40,B60,B20,B66):
    def func(x):
        (F0,Up,Up2,Up3,B4,B6,B2,B62)=tuple(x)
        if sw_conv_cf:
            Blm=(B4,B6,B2,B62)
        else:
            Blm=(0,0,0,0)    
        hop=gen_hop_free(zeta,Blm,False)
        F=get_F(sw_F_type,F0,Up,Up2,Up3)
        U,J=get_ham.UJ(F)
        if sw_full:
            ham=get_ham.get_HF_full(ns,ne,init_n,hop,U,J,F,switch=False)
        else:
            ham=get_HF(hop,U,J,switch=False)
        hop2=gen_hop()
        (eig,uni)=sl.eigh(ham)
        (eig2,uni)=sl.eigh(hop2)
        de=abs((eig-eig[0])-(eig2-eig2[0])).sum()
        return de
    x=[F0,Up,Up2,Up3,B40,B60,B20,B66]
    #minmethod='Nelder-Mead'
    minmethod='Powell'
    #minmethod='BFGS'
    optsol=scopt.minimize(func,x,method=minmethod)
    print(optsol)
    (F01,Up1,Up21,Up31,B4,B6,B2,B62)=optsol.get('x')
    print(F01.round(4),Up1.round(4),Up21.round(4),Up31.round(4),
          B4.round(4),B6.round(4),B2.round(4),B62.round(4))
    if sw_conv_cf:
        Blm=(B4,B6,B2,B62)
    else:
        Blm=(0,0,0,0)
    F=get_F(sw_F_type,F01,Up1,Up21,Up31)
    return(F,Blm)

def main():
    eV2cm=8.06554 #ev to 10e3cm^-1
    #eV2cm=1.
    if sw_conv:
        (F,Blm)=ham_conv(F0,Up,Up2,Up3,zeta,B40,B60,B20,B66)
    else:
        F=get_F(sw_F_type,F0,Up,Up2,Up3)
        if sw_conv_cf:
            Blm=(B40,B60,B20,B66)
        else:
            Blm=(0,0,0,0)
    hop=gen_hop_free(zeta,Blm)
    print(F)
    print(Blm)
    #check hoppings eig ned to devide 6:8 with soc
    (eig,eigf)=sl.eigh(hop)
    np.set_printoptions(linewidth=500)
    print(hop.round(4))
    #print(eig.round(4))
    U,J=get_ham.UJ(F)
    plot_hamHF(hop,U,J,F)
    #exit()
    nwf=scsp.comb(ns,ne,exact=True)
    instates=np.array(list(itts.combinations(range(ns),ne)))
    wf=np.zeros((nwf,ns),dtype=int)
    for i,ist in enumerate(instates):
        wf[i][ist]=1
    np.set_printoptions(linewidth=300)
    #print(wf)

    ham=get_ham.get_ham(wf,hop,nwf,U,J,ns,F)
    #plt.spy(abs(ham))
    #plt.show()
    #print(ham)
    (eig,eigf)=sl.eigh(ham)
    eigmax=(np.where(eig<=erange+eig[0])[0]).size
    if sw_spec:
        wlen,chi,chi2=get_ham.get_spectrum(nwf,wf,eig,eigf,instates,sp1,erange,temp,idelta)
        figs=plt.figure()
        ax1=figs.add_subplot(211)
        ax1.plot(wlen,chi)
        ax1.scatter(eig[n_fcs]-eig[:eigmax],[0]*eigmax,c='red',marker='o')
        ax1.scatter(eig[:eigmax]-eig[1],[0]*eigmax,c='green',marker='+')
        ax1.scatter(eig[:eigmax]-eig[0],[0]*eigmax,c='cyan',marker='|')
        ax1.set_xlim(0,erange)
        ax2=figs.add_subplot(212)
        ax2.plot(wlen,chi2)
        ax2.scatter(eig[n_fcs]-eig[:eigmax],[0]*eigmax,c='red',marker='o')
        ax2.scatter(eig[:eigmax]-eig[1],[0]*eigmax,c='green',marker='+')
        ax2.scatter(eig[:eigmax]-eig[0],[0]*eigmax,c='cyan',marker='|')
        ax2.set_xlim(0,erange)
        figs.savefig('spectrum.pdf')
        #G=np.array([-(1./(complex(iw,id)-eig0)).sum().imag for iw in wlen])
        #plt.plot(wlen,G)
        plt.show()

    #exit()
    eig=(eig-eig[0])*eV2cm
    f=open('output.txt','w')
    for i,ef in enumerate(eigf.T[:eigmax]):
        wfw=np.where(abs(ef)**2>5.0e-3)[0]
        LS=[[l[:,0].sum(),l[:,1].sum()/2] for l in sp1[instates[wfw]]]
        weight=abs(ef[wfw])**2
        f.write('%d %5.2f\n['%(i,eig.round(4)[i]))
        for wg in weight:
            f.write('%4.2f, '%wg)
        f.write(']%4.2f,%d\n'%(weight.sum(),weight.size))
        for sps,LSJ in zip(sp[instates[wfw]],LS):
            f.write('[')
            for spss in sps:
                f.write("%s "%spss)
            f.write('](%4.2f %4.2f %4.2f)\n'%(abs(LSJ[0]),abs(LSJ[1]),abs(LSJ[0]+LSJ[1])))
    f.close()
    print(U.round(4))
    print(J.round(4))
    #print(eig.round(4)[:eigmax])
    de=np.array([j-i for i,j in zip(eig,eig[1:])]).round(4)[:eigmax-1]
    eig2=eig[np.where(de!=0)[0]+1].round(4)
    degenerate=np.array([np.where(eig.round(4)==i)[0].size for i in eig2])
    for ide,ideg in zip(eig2.round(4),degenerate):
        print('%6.3f(%d,J=%3.1f)'%(ide,ideg,(ideg-1)*.5),end=', ')
    else:
        print('')
    plt.scatter([0]*eigmax,eig.round(4)[:eigmax],marker='_')
    if ne==6: #Eu3+
        eig_ofelt=np.array([0.,.374,1.036,1.888,2.866,3.921,5.022, #7FJ (J:0>6)
                            17.374,18.945,21.508,24.456,27.747, #5DJ (J:0>4)
                            24.489,25.340,26.220,26.959,27.386, #5LJ (J:6>10)
                            26.564,26.600,26.725,26.733,27.065, #5GJ (J:2>6)
                            30.483,30.729,30.941,30.964,31.248]) #5HJ (J:3,7,4,5,6)
    elif ne==8: #Tb3+
        eig_ofelt=np.array([0.,2.02,3.279,4.258,4.927,5.405,5.632, #7FJ (J:6>0)
                            20.455,26.216,27.982,30.400,31.649, #5DJ (J:4>0)
                            25.760,27.312,28.183,28.720,28.920, #5LJ (j:10>6)
                            27.263,27.659,28.365,28.960,29.411, #5GJ (J:6>2)
                            30.953,32.713,32.995,34.107,34.414]) #5HJ (J:7>3)
    else:
        eig_ofelt=np.array([0])
    print(eig_ofelt.round(4))
    plt.scatter(eig_ofelt*0,eig_ofelt,marker='_',color='red')
    plt.xlim(-0.05,0.05)
    plt.ylim(0,erange*eV2cm)
    plt.show()

#import time
#t1=time.time()
main()
#t2=time.time()
#print(t2-t1)