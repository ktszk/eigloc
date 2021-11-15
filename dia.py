import numpy as np
import scipy as sc
import scipy.linalg as sl
import scipy.special as scsp
import scipy.optimize as scopt
import itertools as itts
import matplotlib.pyplot as plt
import get_ham
#from numba import jit

lorb=2
ne=3 #electron filling

#zeta= 0.191651

#F0p=0.5693
#F0= 14.7508
#Up= 5.7901e-2
#B40= 1.92436e-3
#B60= 3.91589e-5

#check_d
zeta=0.0
F0=14.3220999
F0p=0.5931263
Up=0.0 #6480558
B40=0.002

#Pr3+
#Up=0.0379
#zeta=0.0905
#Pm3+
#Up=0.0429
#zeta=0.1327
#Eu3+ ofelt
#Up  = 0.04972
#zeta= 0.16366
#Up  = 0.0494
#zeta= 0.1635
#Up  = 0.0503
#zeta= 0.1960
#Tb3+ ofelt
#Up  = 0.05381
#zeta= 0.21139
#Ho3+
#Up=0.0558
#zeta=0.2682

#init_n=[0.9784,0.9796,0.9801,0.9805,0.9805,0.9802,0.0000,
#        0.    ,0.    ,0.0001,0.0001,0.0001,0.    ,0.0000]
#init_n=[.5,1.,1.,1.,1.,1.,.5,0.,0.,0.,0.,0.,0.,0.]
init_n=[.5,1.,0.,1.,.5,0.,0.,0.,0.,0.]
JRPG=np.array([[0,3,3],[6,3,3],[0,2,2]])

cf_type=1
erange=2.0 #plot energy range
idelta=1.e-4
temp=2.6e-2 #~300K

#for arrows plotting
iemax=3.          #max initial energy value
demax=3.5         #max transition energy to plot arrows
demin=1.e-3       #min tansition energy to plot arrows

compair_ham=True  #switch compair MF and QSGW hamiltonian or not
sw_conv=False     #switch calc parameters or not
sw_conv_cf=True   #switch consider crystal field or not
sw_full=True      #switch consider full-interaction or not if obtain MF hamiltonian
sw_spec=False     #switch calc spectrum and grotrian diagram or not
sw_F_type=0       #switch set F setting 
sw_unit=False      #True cm^-1 False eV
sw_TSplot=False   #switch calc Tanabe-Sugano diagram or not
sw_cfsoc=False    #switch crystal field basis j or l,s
sw_arrows=True    #switch plot arrows correspond to transtion (<l|r or 2s+l|m>^2<1e-3)
sw_add_pdata=False
sw_dd=False
if sw_F_type==0: #no use Up2,Up3
    Up2=0
    Up3=0
else:
    try: #if Up2,Up3 are not defined. generate these.
        Up2
        Up3
    except NameError:
        Up2=0
        Up3=0
try:
    B40
except NameError:
    print('B40 set zero')
    B40=0.0
try:
    B60
except NameError:
    print('B60 set zero')
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

ns=4*lorb+2 #number of states
eV2cm=8.06554 #ev to 10e3cm^-1

if ne>ns:
    print('too many electrons')
    exit()

sp1=np.array([[-lorb+l,1] for l in range(2*lorb+1)]
             +[[-lorb+l,-1] for l in range(2*lorb+1)])
sp=np.array(['%d,%d'%tuple(l) for l in sp1])
try:
    init_n
except NameError:
    print('generate init_n')
    init_n=[1 if i<ne else 0 for i in range(ns)]

def get_F(F_type,E0,E1,E2,E3):
    """
    generate Slater-Condon parameteres
    """
    if F_type==0:
        F=np.array([0.,1.*225,0.138*1089,0.0151*7361.64])*abs(E1)
        F[0]=abs(E0)
    elif F_type==1:
        #if Fn>=0, E2 and E3 have upper limit.
        E2max=abs(E1)/70.
        E3max=3.*abs(E1)/14.
        if abs(E2)>E2max:
            E2=E2max
        if abs(E3)>E3max:
            E3=E3max
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
    """
    import hopping parameters from QSGW
    """
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

def gen_hop_free(zeta,Blm,sw_ls=True,wsoc_cf=False):
    """
    generate H_soc and H_cf
    """
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
        if wsoc_cf: #make Hcf from j basis stevens op.
            #make unitary matrix j2lm
            uni=np.zeros((ns,ns))
            for i in range(2*lorb):
                uni[i,4*lorb+1-i]=np.sqrt(2*lorb-i)
                uni[i,2*lorb-1-i]=-np.sqrt(i+1.)
            for i in range(ns//2):
                uni[i+ns//2,ns-1-i]=np.sqrt(i+1.)
                uni[i+2*lorb,2*lorb-i]=np.sqrt(ns/2-i)
            uni=uni/np.sqrt(ns/2)
            if cf_type in {1,2}:
                if lorb==3:
                    O4j=np.diag([ 1.,-3., 2., 2.,-3., 1., 7.,-13.,-3.,  9.,  9.,-3.,-13., 7.])
                    O6j=np.diag([ 0., 0., 0., 0., 0., 0.,1.,-5.,9.,-5.,-5.,9.,-5.,1.])
                    if cf_type==1: #Cube
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
                        O6j[7,11]=7.*O4j[7,11]/5
                        O6j[11,7]=O6j[7,11]
                        O6j[8,12]=O6j[7,11]
                        O6j[12,8]=O6j[8,12]
                        hopcf=uni.T.conjugate().dot((B40*O4j+21.*B60*O6j).dot(uni))
                    elif cf_type==2: #hexagonal
                        O2j=np.diag([10.,-2.,-8.,-8.,-2.,10.,21.,3.,-9.,-15.,-15.,-9.,3.,21.])
                        O66j=np.zeros((ns,ns))
                        O66j[6,12]=np.sqrt(7.)
                        O66j[12,6]=O66j[6,12]
                        O66j[7,13]=O66j[6,12]
                        O66j[13,7]=O66j[7,13]
                        hopcf=uni.T.conjugate().dot((B20*O2j/60.+B40*O4j+21.*B60*O6j+4.*B66*O66j).dot(uni))
                    else:
                        print('There is no crystal field in this symmetry please add')
                        exit()
                elif lorb==2:
                    pass
                elif lorb==1:
                    pass
                else:
                    print('consider only l=1~3')
                    exit()
            else:
                pass
        else: #l basis (wosoc cf)
            if cf_type in {1,2}:
                if lorb==3:
                    O4=np.diag([ 3., -7.,  1., 6., 1., -7., 3., 3.,-7., 1., 6., 1., -7., 3.])
                    O6=np.diag([ 1., -6., 15., -20., 15., -6., 1., 1., -6., 15., -20., 15.,-6., 1.])
                    if cf_type==1:
                        #O44,l=3
                        O4[0,4]=np.sqrt(15.)
                        O4[4,0]=O4[0,4]
                        O4[2,6]=O4[0,4]
                        O4[6,2]=O4[0,4]
                        O4[7,11]=O4[0,4]
                        O4[11,7]=O4[0,4]
                        O4[9,13]=O4[0,4]
                        O4[13,9]=O4[0,4]
                        O4[1,5]=5.
                        O4[5,1]=O4[1,5]
                        O4[8,12]=O4[1,5]
                        O4[12,8]=O4[1,5]
                        #O64,l=3
                        O6[0,4]=-7*np.sqrt(15)
                        O6[4,0]=O6[0,4]
                        O6[2,6]=O6[0,4]
                        O6[6,2]=O6[6,2]
                        O6[7,11]=O6[0,4]
                        O6[11,7]=O6[7,11]
                        O6[9,13]=O6[0,4]
                        O6[13,9]=O6[9,13]
                        O6[1,5]=21.
                        O6[5,1]=O6[1,5]
                        O6[8,12]=O6[1,5]
                        O6[12,8]=O6[8,12]
                        hopcf=B40*O4+3.*B60*O6
                    elif cf_type==2:
                        O2=np.diag([5.,0.,-3,-4,-3.,0.,5.,5.,0.,-3.,-4.,-3.,0.,5.])
                        O66=np.zeros((ns,ns))
                        O66[0,6]=1.
                        O66[6,0]=O66[0,6]
                        O66[7,13]=O66[0,6]
                        O66[13,7]=O66[7,13]
                        hopcf=.05*B20*O2+B40*O4+3.*B60*O6+6.*B66*O66
                elif lorb==2:
                    O4=np.diag([1.,-4.,6.,-4.,1.,1.,-4,6.,-4.,1.])
                    if cf_type==1:
                        O4[0,4]=5.
                        O4[4,0]=O4[0,4]
                        O4[5,9]=O4[0,4]
                        O4[9,5]=O4[5,9]
                        hopcf=B40*O4
                    elif cf_type==2:
                        O2=np.diag([2.,-1.,-2.,-1.,2,2.,-1.,-2.,-1.,2])
                        hopcf=.25*B20*O2+B40*O4
                    else:
                        print('There is no crystal field in this symmetry please add')
                        exit()
                elif lorb==1:
                    pass
                else:
                    print('consider only l=1~3')
                    exit()                    
        #print(hopcf.round(3))
        hop=hopsoc+hopcf
    else:
        hop=hopsoc
    return(hop)

def get_HF(ham0,U,J,temp=1.0e-9,eps=1.0e-6,itemax=1000,switch=True):
    """
    obtain H_HF corresponding to many body hamiltonian
    """
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
                L+=(i-3)*j
                S+=(.5 if i<ns//2 else -.5)*j
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

def plot_hamHF(hop,U,J,dU,F,temp=1.0e-9):
    """
    plot H_HF and compare H_HF and H_import
    """
    if sw_full:
        ham=get_ham.get_HF_full(ns,ne,init_n,hop,U,J,dU,F,temp,lorb=lorb)
    else:
        ham=get_HF(hop,U,J,temp)
    print(ham.round(3))
    (eig,uni)=sl.eigh(ham)
    #print(ham.round(3))
    print((eig).round(3))
    print((abs(uni)**2).round(2))
    plt.scatter([0]*ns,eig,marker='_')
    if False:
        hop2=gen_hop()
        print(hop2.real.round(2))
        uni0=np.zeros((ns,ns),dtype=complex)
        for i in range(ns//2):
            if i==lorb:
                uni0[i,i]=1.
                uni0[i+ns//2,i+ns//2]=1.
            elif i<lorb:
                uni0[i,i]=-1j/np.sqrt(2.)
                uni0[i,2*lorb-i]=1/np.sqrt(2.)
                uni0[i+ns//2,i+ns//2]=-1j/np.sqrt(2.)
                uni0[i+ns//2,2*lorb-i+ns//2]=1/np.sqrt(2.)
            else:
                uni0[i,i]=(-1.)**((i-1)%2)/np.sqrt(2.)
                uni0[i,2*lorb-i]=(-1)**((i-1)%2)*1j/np.sqrt(2.)
                uni0[i+ns//2,i+ns//2]=(-1)**(i%2)/np.sqrt(2.)
                uni0[i+ns//2,2*lorb-i+ns//2]=(-1)**(i%2)*1j/np.sqrt(2.)
        (eig2,uni2)=sl.eigh(hop2)
        uni=uni0.conjugate().dot(uni2)
        f2=lambda mu: ne+.5*(np.tanh(0.5*(eig2-mu)/temp)-1.).sum()
        mu2=scopt.brentq(f2,eig2.min(),eig2.max())
        print((eig2-mu2).round(3))
        print((uni**2).round(3))
        plt.scatter([0]*ns,eig2-mu2,marker='_',color='red')
    plt.show()

def ham_conv(F0,F0p,Up,Up2,Up3,zeta,B40,B60,B20,B66):
    """
    self-consistent cycle to define parameters
    """
    def func(x):
        (F0,F0p,Up,Up2,Up3,B4,B6,B2,B62)=tuple(x)
        if sw_conv_cf:
            Blm=(B4,B6,B2,B62)
        else:
            Blm=(0,0,0,0)    
        hop=gen_hop_free(zeta,Blm,False,sw_cfsoc)
        F=get_F(sw_F_type,F0,Up,Up2,Up3)
        U,J=get_ham.UJ(F,lorb)
        dU=get_ham.get_dU(F0p,lorb)
        if sw_full:
            ham=get_ham.get_HF_full(ns,ne,init_n,hop,U,J,dU,F,switch=False,lorb=lorb)
        else:
            ham=get_HF(hop,U,J,switch=False)
        hop2=gen_hop()
        (eig,uni)=sl.eigh(ham)
        (eig2,uni)=sl.eigh(hop2)
        de=(abs((eig-eig[0])-(eig2-eig2[0]))**2).sum()
        return de
    x=[F0,F0p,Up,Up2,Up3,B40,B60,B20,B66]
    #minmethod='Nelder-Mead'
    minmethod='Powell'
    #minmethod='BFGS'
    optsol=scopt.minimize(func,x,method=minmethod)
    print(optsol)
    (F01,F0p1,Up1,Up21,Up31,B4,B6,B2,B62)=optsol.get('x')
    print(F0p1.round(4),F01.round(4),Up1.round(4),Up21.round(4),Up31.round(4),
          B4.round(4),B6.round(4),B2.round(4),B62.round(4))
    if sw_conv_cf:
        Blm=(B4,B6,B2,B62)
    else:
        Blm=(0,0,0,0)
    F=get_F(sw_F_type,F01,Up1,Up21,Up31)
    return(F,F0p1,Blm)

def plot_TS(U,J,F,nwf,wf,dqmax=5,dqlen=100,mem_enough=False):
    if sw_F_type==0:
        RB=5.*F[2]/63.
        RC=(9.*F[1]-5*F[2])/441.
    elif sw_F_type==3:
        RB=918/eV2cm
        RC=4133/eV2cm
    dq=np.linspace(0,dqmax,dqlen)
    hop=np.array([gen_hop_free(zeta,(dqq,0,0,0),wsoc_cf=sw_cfsoc) for dqq in dq])
    if mem_enough:
        eg=[sl.eigvalsh(get_ham.get_ham(wf,hp,nwf,U,J,ns,F,l=lorb)) for hp in hop]
        eig=np.array([egg-egg[0] for egg in eg])
    else:
        eig=[]
        for hp in hop:
            ham=get_ham.get_ham(wf,hp,nwf,U,J,ns,F,l=lorb)
            eg=sl.eigvalsh(ham)
            eig.append(eg-eg[0])
        eig=np.array(eig)
    xlist=dq/RB
    ylist=eig/RB
    fig=plt.figure()
    ax=fig.add_subplot(111,xlabel='Dq/B',ylabel='Energy/B',title='Tanabe-Sugano Diagram',xlim=(0,3),ylim=(0,50))
    ax.plot(xlist,ylist,c='black',lw=1.)
    fig.savefig("TSdiagram.png")
    plt.show()

def plot_dd():
    def plotdd(ax,z_list,F2list,xtlabs,marker,color):
        for j,(zl,f2) in enumerate(zip(z_list,F2list)):
            ne=j+1
            nwf=scsp.comb(ns,ne,exact=True)
            instates=np.array(list(itts.combinations(range(ns),ne)))
            wf=np.zeros((nwf,ns),dtype=int)
            for i,ist in enumerate(instates):
                wf[i][ist]=1
            F=get_F(sw_F_type,10,f2,0,0)
            U,J=get_ham.UJ(F,lorb)
            hop=gen_hop_free(zl,Blm,wsoc_cf=sw_cfsoc)
            ham=get_ham.get_ham(wf,hop,nwf,U,J,ns,F,l=lorb)
            (eig,eigf)=sl.eigh(ham)
            eigmax=(np.where(eig<=erange+eig[0])[0]).size
            eig=(eig-eig[0])*unit
            ax.scatter([j]*eigmax,eig.round(4)[:eigmax],marker=marker,c=color)

    Blm=(0,0,0,0)
    cunit='cm$^{-1}$' if sw_unit else 'eV'
    unit=eV2cm if sw_unit else 1.
    xtlabs= [ 'Ce' , 'Pr' , 'Nd' , 'Pm' , 'Sm' , 'Eu' , 'Gd' , 'Tb' , 'Dy' , 'Ho' , 'Er' , 'Tm' , 'Yb' ]
    #dieke
    z_list= [0.0794,0.0905,0.1086,0.1327,0.1488,0.1637,0.1960,0.2114,0.2356,0.2682,0.3031,0.3293,0.3574]
    F2list= [0.0000,0.0379,0.0418,0.0429,0.0459,0.0497,0.0503,0.0538,0.0521,0.0558,0.0532,0.0559,0.0000]
    #our results
    z_list2=[0.0741,0.1102,0.1346,0.1532,0.1590,0.1917,0.2074,0.2391,0.2920,0.3548,0.4456,0.5136,0.4180]
    F2list2=[0.0236,0.0414,0.0441,0.0521,0.0521,0.0569,0.0579,0.0626,0.0637,0.0646,0.0678,0.0703,0.0740]
    plt.plot(np.arange(len(F2list))+1,F2list)
    plt.plot(np.arange(len(F2list2))+1,F2list2)
    plt.show()
    exit()
    fig=plt.figure()
    ax=fig.add_subplot(111,ylim=(0,erange*unit),xticks=list(range(len(z_list))),xticklabels=xtlabs)
    ax.set_ylabel(r'Energy (%s)'%cunit)
    plotdd(ax,z_list,F2list,xtlabs,'.','black')
    plotdd(ax,z_list2,F2list2,xtlabs,'_','red')
    plt.show()

def main():
    """
    main program of dia.py
    """
    eV2cm=8.06554 #ev to 10e3cm^-1
    #eV2cm=1.
    if sw_dd:
        pass
    else:
        if sw_conv:
            (F,Fp,Blm)=ham_conv(F0,F0p,Up,Up2,Up3,zeta,B40,B60,B20,B66)
        else:
            Fp=F0p
            F=get_F(sw_F_type,F0,Up,Up2,Up3)
            if sw_conv_cf:
                Blm=(B40,B60,B20,B66)
            else:
                Blm=(0,0,0,0)
        nwf=scsp.comb(ns,ne,exact=True)
        instates=np.array(list(itts.combinations(range(ns),ne)))
        wf=np.zeros((nwf,ns),dtype=int)
        for i,ist in enumerate(instates):
            wf[i][ist]=1
        U,J=get_ham.UJ(F,lorb)
        dU=get_ham.get_dU(Fp,lorb)

    if sw_TSplot:
        plot_TS(U,J,F,nwf,wf)
    elif sw_dd:
        plot_dd()
    else:
        hop=gen_hop_free(zeta,Blm,wsoc_cf=sw_cfsoc)
        print(F)
        print(Blm)
        #check hoppings eig ned to devide 6:8 with soc
        (eig,eigf)=sl.eigh(hop)
        np.set_printoptions(linewidth=500)
        #print(hop.round(4))
        #print(eig.round(4))
        if compair_ham:
            plot_hamHF(hop,U,J,dU,F)
        np.set_printoptions(linewidth=300)
        #print(wf)
        ham=get_ham.get_ham(wf,hop,nwf,U,J,ns,F,l=lorb)
        #plt.spy(abs(ham))
        #plt.show()
        #print(ham)
        (eig,eigf)=sl.eigh(ham)
        eigmax=(np.where(eig<=erange+eig[0])[0]).size
        if sw_spec:
            """
            if sw_spec true, calc and plot absorption spectrum
            """
            unit=eV2cm if sw_unit else 1.
            wlen,chi,chi2,Jeig,Jcolor,arrows,arrows_mag=get_ham.get_spectrum(
                nwf,wf,eig,eigf,instates,sp1,erange,temp,lorb,JRPG,iemax,demax,demin,idelta)
            figs=plt.figure()
            ax1=figs.add_subplot(211,xlim=(0,erange*unit))
            ax1.plot(wlen*unit,chi)
            ax1.scatter((eig[n_fcs]-eig[:eigmax])*unit,[0]*eigmax,c='red',marker='o')
            ax1.scatter((eig[:eigmax]-eig[1])*unit,[0]*eigmax,c='green',marker='+')
            ax1.scatter((eig[:eigmax]-eig[0])*unit,[0]*eigmax,c='cyan',marker='|')
            ax2=figs.add_subplot(212,xlim=(0,erange*unit))
            ax2.plot(wlen*unit,chi2)
            ax2.scatter((eig[n_fcs]-eig[:eigmax])*unit,[0]*eigmax,c='red',marker='o')
            ax2.scatter((eig[:eigmax]-eig[1])*unit,[0]*eigmax,c='green',marker='+')
            ax2.scatter((eig[:eigmax]-eig[0])*unit,[0]*eigmax,c='cyan',marker='|')
            figs.savefig('spectrum.pdf')
            plt.show()

            #print(arrows)
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.scatter(Jeig[:eigmax],(eig-eig[0])[:eigmax],c=Jcolor[:eigmax])
            if sw_arrows:
                for ar in arrows: #electric dipole lightgray
                    ax.arrow(x=Jeig[ar[1]],y=eig[ar[1]]-eig[0],dx=Jeig[ar[0]]-Jeig[ar[1]],dy=eig[ar[0]]-eig[ar[1]],
                             width=0.01,head_width=0.05,head_length=0.2,length_includes_head=True,color='lightgray')
                for ar in arrows_mag: #magnetic dipole darkgray
                    ax.arrow(x=Jeig[ar[1]],y=eig[ar[1]]-eig[0],dx=Jeig[ar[0]]-Jeig[ar[1]],dy=eig[ar[0]]-eig[ar[1]],
                             width=0.01,head_width=0.05,head_length=0.2,length_includes_head=True,color='darkgray')
            fig.savefig('Grotrian.pdf')
            plt.show()

        #exit()
        eig=(eig-eig[0])*eV2cm
        f=open('output.txt','w')
        for i,ef in enumerate(eigf.T[:eigmax]):
            wfw=np.where(abs(ef)**2>5.0e-3)[0]
            LS=[[l[:,0].sum(),l[:,1].sum()/2] for l in sp1[instates[wfw]]]
            weight=abs(ef[wfw])**2
            f.write('%d %5.2f\n['%(i,eig.round(3)[i]))
            for wg in weight:
                f.write('%4.2f, '%wg)
            f.write(']%4.2f,%d\n'%(weight.sum(),weight.size))
            for sps,LSJ in zip(sp[instates[wfw]],LS):
                f.write('[')
                for spss in sps:
                    f.write("%s "%spss)
                f.write('](%4.2f %4.2f %4.2f)\n'%(abs(LSJ[0]),abs(LSJ[1]),abs(LSJ[0]+LSJ[1])))
        f.close()

        f=open('eig_diff.txt','w')
        for i,est in enumerate(eig[:eigmax]):
            for j,een in enumerate(eig[i+1:eigmax]):
                diff_e=een-est
                if diff_e<erange*eV2cm:
                    f.write('%6.3f, %d, %d\n'%(diff_e,j+i+1,i))
        f.close()
        print(U.round(4))
        print(J.round(4))
        print(dU.round(4))
        eig2=np.unique(eig[:eigmax].round(3))
        degenerate=np.array([np.where(eig.round(3)==i)[0].size for i in eig2])
        for ide,ideg in zip(eig2.round(3),degenerate):
            print('%6.3f(%d,J=%3.1f)'%(ide,ideg,(ideg-1)*.5),end=', ')
        else:
            print('')
        plt.scatter([0]*eigmax,eig.round(4)[:eigmax],marker='_')
        if lorb==3 and sw_add_pdata:
            if ne==1: #Ce3+
                pass
            elif ne==2: #pr3+
                eig_previous=np.array([0.,2.026,4.167,4.735,6.088,6.774, #3HJ(J:4>6),3FJ(J:2>4)
                                       10.002,16.973,20.896,50.673, #1G4,1D2,1I6,1S0
                                       20.407,20.984,22.211]) #3PJ(J:0>2)
            elif ne==3: #Nd3+
                pass
            elif ne==4: #Pm3+
                eig_previous=np.array([0.,1.586,3.305,5.101,6.934, #5IJ(J:4>8)
                                       12.288,12.731,13.650,14.602,16.016, #5FJ(J:1>5)
                                       14.369,15.915,17.186,18.756, #5S2, 3KJ(J:6>8)
                                       17.473,19.881,24.369, #3HJ(J:4>6)
                                       17.719,18.095,20.274,22.204,22.632, #5GJ(J:2>6)
                                       21.187,24.252,26.736,22.030,23.154,24.765, #3GJ(J:3>5),3DJ(J:2,1,3)
                                       22.208,23.353,24.253,24.254,27.838,28.429, #3LJ(J:7>9),3MJ(J:8>10)
                                       24.849,27.120,25.373,26.439,28.094,28.654,29.128, #3PJ(J:0>1),3FJ(J:4,2,3)
                                       25.373,27.514,28.719,29.221,29.319,32.290, #1D2,3IJ(J:5>7),1L8,1G4
                                       29.788,30.318,31.161,31.989,33.019,32.864, #5DJ(J:0>4),1D2
                                       32.552,35.979,32.930,34.442,36.962,38.443, #3FJ(J:3,4),3P0,1I6,1K7,1D2
                                       35.337,35.983,38.013,36.362,39.684,39.743]) #3HJ(J:4>6),1H5,3GJ(J:4,3)
            elif ne==5: #Sm3+
                pass
            elif ne==6: #Eu3+
                eig_previous=np.array([0.,.374,1.036,1.888,2.866,3.921,5.022, #7FJ(J:0>6)
                                       17.374,18.945,21.508,24.456,27.747, #5DJ(J:0>4)
                                       24.489,25.340,26.220,26.959,27.386, #5LJ(J:6>10)
                                       26.564,26.600,26.725,26.733,27.065, #5GJ(J:2>6)
                                       30.483,30.729,30.941,30.964,31.248, #5HJ(J:3,7,4,5,6)
                                       33.616,33.870,34.805,34.919,34.947, #5IJ(J:5,4,8,6,7)
                                       33.871,33.955,34.085,34.440,34.932, #5FJ(J:2,3,1,4)
                                       34.457,37.040,                      #3PJ(J:0,1)
                                       36.179,37.573,38.809,39.508])       #5KJ(J:5>8)
            elif ne==7: #Gd3+
                pass
            elif ne==8: #Tb3+
                eig_previous=np.array([0.,2.02,3.279,4.258,4.927,5.405,5.632, #7FJ(J:6>0)
                                       20.455,26.216,27.982,30.400,31.649, #5DJ(J:4>0)
                                       25.760,27.312,28.183,28.720,28.920, #5LJ(j:10>6)
                                       27.263,27.659,28.365,28.960,29.411, #5GJ(J:6>2)
                                       30.953,32.713,32.995,34.107,34.414]) #5HJ(J:7>3)
            elif ne==9: #Dy3+
                pass
            elif ne==10: #Ho3+
                eig_previous=np.array([0.,5.087,8.647,11.255,13.349,18.211, #5IJ(J:8>4),5S2
                                       15.403,18.387,20.568,21.113,22.365, #5FJ(J:5>1)
                                       21.321,26.195,30.261,28.391,36.745,37.444, #3KJ(J:8>6),3HJ(J:6,4,5)
                                       22.192,24.104,26.070,28.620,28.953, #5GJ(J:6,5,4,2,3)
                                       31.094,30.726,35.859,36.802,33.296,38.584, #5G2',3FJ(J:4>2),3MJ(J:10,9)
                                       28.421,33.862,37.673,33.921,35.125,36.267, #3LJ(J:9>7),3D3,5D4,1L8
                                       33.973,36.700,38.521,38.214,39.215,39.654]) #3PJ(J:1,0,2),3IJ(J:7,5,6)
            elif ne==11: #Er3+
                pass
            elif ne==12: #Tm3+
                pass
            elif ne==13: #Yb3+
                pass
            else:
                pass
        try:
            print(np.sort(eig_previous.round(3)))
            plt.scatter(eig_previous*0+0.01,eig_previous,marker='_',color='red')
        except NameError:
            pass
        plt.xlim(-0.05,0.05)
        plt.ylim(0,erange*eV2cm)
        plt.show()

#import time
#t1=time.time()
main()
#t2=time.time()
#print(t2-t1)
