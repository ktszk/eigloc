import numpy as np
cimport cython
cimport numpy as cnp
from sympy.physics.wigner import gaunt
import scipy.linalg as sl
import scipy.optimize as scopt
import scipy.constants as scconst
import matplotlib.pyplot as plt

@cython.boundscheck(False)
@cython.wraparound(False)
def gencp(int p,int l,int m):
    """
    generate Gaunt coefficient
    """
    cdef cnp.ndarray[cnp.float64_t,ndim=3] cp
    def get_gaunt(p,m,l):
        return float(gaunt(3,p,3,-l,l-m,m))*np.sqrt(4.*np.pi/(2.*p+1.))*(-1)**l
    cp=np.array([[[get_gaunt(2*pp,ll-l,mm-l) for pp in range(p)] 
                  for mm in range(2*m+1)] for ll in range(2*l+1)])
    return cp

def UJ(F,int l=3):
    """
    generate onsite interaction U,J
    """
    cdef long m1,m2,i,j,lmax=2*l+1
    cdef cnp.ndarray[cnp.float64_t,ndim=2] U,J

    cp=gencp(l+1,l,l)
    Ulm=lambda m1,m2: (F*cp[m1,m1]*cp[m2,m2]).sum()
    Jlm=lambda m1,m2: (F*cp[m1,m2]**2).sum()
    U=np.array([[float(Ulm(i,j)) for i in range(lmax)] for j in range(lmax)])
    J=np.array([[float(Jlm(i,j)) for i in range(lmax)] for j in range(lmax)])
    J=J-np.diag(J.diagonal())
    return U,J

def get_dU(F0,int l=3):
    cdef long m1,m2,i,j,lmax=2*l+1
    cdef cnp.ndarray[cnp.float64_t,ndim=2] dU

    cp=gencp(l+1,l,l)
    dulm= lambda m1,m2: (F0*cp[m1,m1,0]*cp[m2,m2,0])
    dU=np.array([[float(dulm(i,j)) for i in range(lmax)] for j in range(lmax)])
    return dU

def get_J(cnp.ndarray[cnp.int64_t,ndim=2] wf,int nwf,cnp.ndarray[cnp.complex128_t,ndim=2] uni,
          cnp.ndarray[cnp.int64_t,ndim=2] instates,cnp.ndarray[cnp.int64_t,ndim=2] sp1,int eigmax):
    """
    calculate total/orbital/spin angular momentum J,L,S and generate magnetic dipole
    """
    cdef long i,j,k,spp_flag,spm_flag,lp_flag,lm_flag,lz,tmp
    cdef double lpnum,lmnum,upsign,dnsign
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] Jx,Jy,Jz
    cdef cnp.ndarray[cnp.float64_t,ndim=2] mag_specm,Lsq,Ssq
    cdef cnp.ndarray[cnp.int64_t] ist,jst,lflipp,lflipm,sflipp,sflipm
    cdef cnp.ndarray[cnp.int64_t,ndim=1] Lz0,Sz0

    cdef cnp.ndarray[cnp.float64_t,ndim=2] Lp0=np.zeros((nwf,nwf)),Lm0=np.zeros((nwf,nwf))
    cdef cnp.ndarray[cnp.float64_t,ndim=2] Sp0=np.zeros((nwf,nwf)),Sm0=np.zeros((nwf,nwf))
    cdef lmax=3, nhf=int(len(sp1)//2)

    #J+,J-
    for i,ist in enumerate(wf):
        for k in np.where(ist==1)[0]:
            lz=sp1[k][0]
            if lz!=3 and ist[k+1]==0:
                lpnum=np.sqrt((lmax-lz)*(lmax+lz+1.))
                lflipp=ist.copy()
                lflipp[k]=0
                lflipp[k+1]=1
                lp_flag=1
            else:
                lp_flag=0
            if lz!=-3 and ist[k-1]==0:
                lmnum=np.sqrt((lmax+lz)*(lmax-lz+1.))
                lflipm=ist.copy()
                lflipm[k]=0
                lflipm[k-1]=1
                lm_flag=1
            else:
                lm_flag=0
            if k<nhf:
                if ist[k+nhf]==0:
                    sflipm=ist.copy()
                    sflipm[k]=0
                    sflipm[k+nhf]=1
                    upsign=(-1.)**ist[k+1:k+nhf].sum()
                    spm_flag=1
                else:
                    spm_flag=0
                spp_flag=0
            else:
                if ist[k-nhf]==0:
                    sflipp=ist.copy()
                    sflipp[k]=0
                    sflipp[k-nhf]=1
                    dnsign=(-1.)**ist[k-nhf+1:k].sum()
                    spp_flag=1
                else:
                    spp_flag=0
                spm_flag=0
            for j,jst in enumerate(wf):
                if i!=j:
                    if lp_flag!=0:
                        tmp=abs(lflipp-jst).sum()
                        if tmp==0:
                            Lp0[i,j]=lpnum
                            lp_flag=0
                    if lm_flag!=0:
                        tmp=abs(lflipm-jst).sum()
                        if tmp==0:
                            Lm0[i,j]=lmnum
                            lm_flag=0
                    if spp_flag!=0:
                        tmp=abs(sflipp-jst).sum()
                        if tmp==0:
                            Sp0[i,j]=dnsign
                            spp_flag=0
                    if spm_flag!=0:
                        tmp=abs(sflipm-jst).sum()
                        if tmp==0:
                            Sm0[i,j]=upsign
                            spm_flag=0
                    if (spp_flag+spm_flag+lp_flag+lm_flag)==0:
                        break
    Jx=uni[:,:eigmax].T.conjugate().dot((.5*(Lp0+Lm0+2.*(Sp0+Sm0))).dot(uni[:,:eigmax]))
    Jy=uni[:,:eigmax].T.conjugate().dot((-.5j*(Lp0-Lm0+2.*(Sp0-Sm0))).dot(uni[:,:eigmax]))
    #Jz
    Lz0=np.array([a[:,0].sum() for a in sp1[instates]])
    Sz0=np.array([a[:,1].sum() for a in sp1[instates]])
    Jz=uni[:,:eigmax].T.conjugate().dot(np.diag(Lz0+Sz0).dot(uni[:,:eigmax]))
    #magnetic dipole
    m0=1e10*scconst.hbar/(2.*scconst.c*scconst.m_e) #e\AA is unity
    mag_spec=m0*m0*(abs(Jx)**2+abs(Jy)**2+abs(Jz)**2)
    Ssq=.25*(Sp0+Sm0).dot(Sp0+Sm0)-.25*(Sp0-Sm0).dot(Sp0-Sm0)+.25*np.diag(Sz0)**2
    eig,uni_s=sl.eigh(Ssq)
    S_delta=uni_s.dot(uni_s.conjugate().T)
    Lsq=.25*(Lp0+Lm0).dot(Lp0+Lm0)-.25*(Lp0-Lm0).dot(Lp0-Lm0)+np.diag(Lz0)**2 #L^2
    return mag_spec,Lsq,Lz0,Ssq

def gen_spec(cnp.ndarray[cnp.int64_t,ndim=2] wf,int nwf,cnp.ndarray[cnp.complex128_t,ndim=2] uni,
          cnp.ndarray[cnp.int64_t,ndim=2] instates,cnp.ndarray[cnp.int64_t,ndim=2] sp1,int eigmax):
    """
    calculate electric and magnetic dipole
    """
    #electric dipole_check
    cdef long i,j,l,l1,li,lj,j0,J,miz,mjz
    cdef double lzi,lzj
    cdef cnp.ndarray[cnp.int64_t] ist,jst
    cdef cnp.ndarray[cnp.int64_t,ndim=1] Lz0
    cdef cnp.ndarray[cnp.float64_t,ndim=2] r_spec,Lsq, Ssq
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] rp_mat,rm_mat,rz_mat,rx0,ry0,rz0,rx,ry,rz

    mag_spec,Lsq,Lz0,Ssq=get_J(wf,nwf,uni,instates,sp1,eigmax)
    print('calc mag dipole')
    eig,uni_l=sl.eigh(Lsq) #eigval of l^2(= l(l+1))
    Lz=uni_l.T.conjugate().dot(np.diag(Lz0).dot(uni_l))
    LL=np.array([i*(i+1) for i in range(20)])
    l1=0
    JJ=[]
    lmat=[]
    for i in range(20):
        l=np.where(eig.round(2)==eig.round(2)[l1])[0].size
        l1=l1+l
        lmat.append(l)
        e0=int(eig[l1-1])
        for j,j2 in enumerate(LL):
            if e0==j2:
                JJ.append(j)
        if l1==eig.size:
            break
    rot_mat=np.zeros((nwf,nwf),dtype='c16')
    l1=0
    Lz_size=[]
    L_size=[]
    for l,J in zip(lmat,JJ):
        blz=Lz[l1:l1+l,l1:l1+l].copy()
        elz,uni_lz=sl.eigh(blz)
        rot_mat[l1:l1+l,l1:l1+l]=uni_lz.copy()
        #print(elz.round(6),l)
        l1=l1+l
        Lz_size.extend(elz)
        L_size.extend((np.zeros(elz.size,dtype='i4')+J))
    Lz_size=np.array(Lz_size)
    L_size=np.array(L_size)
    uni_ll=uni_l.dot(rot_mat)
    Ssq2=uni_ll.T.conjugate().dot(Ssq.dot(uni_ll))
    l1=0
    smat=[]
    for l in lmat:
        lz=Lz_size[l1:l1+l].round(2)
        s1=0
        for i in range(20):
            s=np.where(lz==lz[s1])[0].size
            s1=s1+s
            smat.append(s)
            if s1==lz.size:
                break
        l1=l1+l
    rot_mat=np.zeros((nwf,nwf),dtype='c16')
    s1=0
    Ssq_size=[]
    for s in smat:
        bs=Ssq2[s1:s1+s,s1:s1+s].copy()
        eig_s,uni_s=sl.eigh(bs)
        rot_mat[s1:s1+s,s1:s1+s]=uni_s.copy()
        #print(eig_s.round(6),s)
        s1=s1+s
        Ssq_size.extend(eig_s)
    Ssq_size=np.array(Ssq_size)
    uni_lls=uni_ll.dot(rot_mat)

    ckuni=abs(uni[:,0].dot(uni_lls))**2
    ck_phi=np.where(ckuni>1.e-3)[0]
    f=open('ck_LS.txt','w')
    for w,s,l in zip(ckuni[ck_phi],Ssq_size[ck_phi],L_size[ck_phi]):
        f.write('weight=%5.3f,L=%4.1f,S(S+1)=%4.1f\n'%(w,l,s))
    f.write('eig_200\n')
    ckuni=abs(uni[:,200].dot(uni_lls))**2
    ck_phi=np.where(ckuni>1.e-3)[0]
    for w,s,l in zip(ckuni[ck_phi],Ssq_size[ck_phi],L_size[ck_phi]):
        f.write('weight=%5.3f,L=%4.1f,S(S+1)=%4.1f\n'%(w,l,s))
    f.close()
    rp_mat=np.zeros((nwf,nwf),dtype='c16')
    rm_mat=np.zeros((nwf,nwf),dtype='c16')
    rz_mat=np.zeros((nwf,nwf),dtype='c16')
    for i,(li,lzj,si) in enumerate(zip(L_size,Lz_size,Ssq_size)):
        miz=int(lzj)
        for j0, (lj,lzj,sj) in enumerate(zip(L_size[i:],Lz_size[i:],Ssq_size[i:])):
            if si==sj:
                j=j0+i
                mjz=int(lzj)
                rp_mat[i,j]=complex(gaunt(lj,1,li,miz,1,mjz))
                rm_mat[i,j]=complex(gaunt(lj,1,li,miz,-1,mjz))
                rz_mat[i,j]=complex(gaunt(lj,1,li,miz,0,mjz))
                rp_mat[j,i]=rp_mat[i,j].conjugate()
                rm_mat[j,i]=rm_mat[i,j].conjugate()
                rz_mat[j,i]=rz_mat[i,j].conjugate()
    rx0=uni_lls.dot((rm_mat-rp_mat).dot(uni_lls.T.conjugate()))/np.sqrt(2.)
    ry0=1j*uni_lls.dot((rp_mat+rm_mat).dot(uni_lls.T.conjugate()))/np.sqrt(2.)
    rz0=uni_lls.dot(rz_mat.dot(uni_lls.T.conjugate()))
    rx=uni[:,:eigmax].T.conjugate().dot(rx0.dot(uni[:,:eigmax]))
    ry=uni[:,:eigmax].T.conjugate().dot(ry0.dot(uni[:,:eigmax]))
    rz=uni[:,:eigmax].T.conjugate().dot(rz0.dot(uni[:,:eigmax]))
    r_spec=abs(rx)**2+abs(ry)**2+abs(rz)**2
    print('calc elec dipole')
    return mag_spec,r_spec

def get_spectrum(int nwf,cnp.ndarray[cnp.int64_t,ndim=2] wf,cnp.ndarray[cnp.float64_t,ndim=1] eig,
                 cnp.ndarray[cnp.complex128_t,ndim=2] eigf,cnp.ndarray[cnp.int64_t,ndim=2] instates,
                 cnp.ndarray[cnp.int64_t,ndim=2] sp1,int erange,double temp,double id=1.0e-3,int wmesh=2000):
    """
    generate spectrum
    """
    cdef long eig_int_max=(np.where(eig<=2.*erange+eig[0])[0]).size
    cdef cnp.ndarray[cnp.float64_t,ndim=1] chi,chi2,dfunc,deig,wlen=np.linspace(0,erange,wmesh)
    mnn2,mnn=gen_spec(wf,nwf,eigf,instates,sp1,eig_int_max)
    fig=plt.figure()
    ax1=fig.add_subplot(211)
    maps=ax1.imshow(mnn.round(3),cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(maps,ax=ax1)
    fig.savefig('mnn_map.png')
    ax2=fig.add_subplot(212)
    ax2.plot(range(eig_int_max),mnn[0,:])
    mnn=mnn.flatten()
    mnn2=mnn2.flatten()
    eig0=(eig-eig[0])[:eig_int_max]
    func=np.exp(-eig0/temp)
    func=func/func.sum()
    eigf0=eigf.T[:eig_int_max]
    deig=np.array([[e1-e2 for e1 in eig0] for e2 in eig0]).flatten()
    dfunc=np.array([[e2-e1 for e1 in func] for e2 in func]).flatten()
    chi=np.array([(mnn*dfunc/(complex(iw,id)+deig)).sum().imag for iw in wlen])
    chi2=np.array([(mnn2*dfunc/(complex(iw,id)+deig)).sum().imag for iw in wlen])
    return wlen,chi,chi2

def get_HF_full(int ns,int ne,init_n,ham0,cnp.ndarray[cnp.float64_t,ndim=2] U,
                cnp.ndarray[cnp.float64_t,ndim=2] J, cnp.ndarray[cnp.float64_t,ndim=2] dU, F,
                double temp=1.0e-9,double eps=1.0e-6,int itemax=1000,switch=True):
    """
    calculate MF hamiltonian with full Coulomb interactions
    """
    cdef long i,j,k,l,m
    cdef double mu
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] ham, ham_I=np.zeros((ns,ns),dtype='c16')
    cp=gencp(4,3,3)
    G=lambda m1,m2,m3,m4:(-1)**abs(m1-m3)*(F*cp[m1,m3]*cp[m2,m4]).sum()
    ini_n=np.array(init_n)
    n1=np.diag(ini_n)
    for k in range(itemax):
        ham_I=ham_I*0.
        for i in range(ns//2):
            # onsite
            ham_I[i,i]=((U[i,:]*n1.diagonal()[ns//2:]).sum()
                          +np.delete((U[i,:]-J[i,:]+dU[i,:])*n1.diagonal()[:ns//2],i).sum())
            ham_I[i+ns//2,i+ns//2]=((U[i,:]*n1.diagonal()[:ns//2]).sum()
                                      +np.delete((U[i,:]-J[i,:]+dU[i,:])*n1.diagonal()[ns//2:],i).sum())
            for j in range(i+1,ns//2): #offsite
                for l in range(ns//2):
                    m=l+i-j
                    ham_I[i,j]=ham_I[i,j]+G(i,l,j,m)*n1[l+ns//2,m+ns//2]
                    ham_I[i,j]=ham_I[i,j]+(G(i,l,j,m)-G(i,l,m,j))*n1[l,m]
                    ham_I[i+ns//2,j+ns//2]=ham_I[i+ns//2,j+ns//2]+G(i,l,j,m)*n1[l,m]
                    ham_I[i+ns//2,j+ns//2]=ham_I[i+ns//2,j+ns//2]+(G(i,l,j,m)-G(i,l,m,j))*n1[l+ns//2,m+ns//2]
                    #ham_I[i,j+ns//2]=ham_I[i,j+ns//2]+G(i,l,m,j)*n1[l+ns//2,m]
                    #ham_I[i+ns//2,j]=ham_I[i+ns//2,j]+G(i,l,m,j)*n1[l,m+ns//2]
                ham_I[j,i]=ham_I[i,j].conjugate()
                ham_I[j+ns//2,i+ns//2]=ham_I[i+ns//2,j+ns//2].conjugate()
                #ham_I[j+ns//2,i]=ham_I[i,j+ns//2].conjugate()
                #ham_I[j,i+ns//2]=ham_I[i+ns//2,j].conjugate()
        ham=ham0+ham_I
        (eig,uni)=sl.eigh(ham)
        f=lambda mu: ne+.5*(np.tanh(0.5*(eig-mu)/temp)-1.).sum()
        mu=scopt.brentq(f,eig.min(),eig.max())
        n0=.5-.5*np.tanh(0.5*(eig-mu)/temp)
        new_n=uni.dot(np.diag(n0).dot(uni.T.conjugate()))
        dn=abs(new_n-n1).sum()/abs(new_n).sum()
        if dn<eps:
            L,S=0.,0.
            for i,j in enumerate(new_n.diagonal()):
                L=L+(i-3)*j
                S=S+(.5 if i<ns//2 else -.5)*j
            if switch:
                print('converged loop %d'%k)
                print(L,S,L+S)
                np.set_printoptions(linewidth=500)
            break
        else:
            n1=new_n
    else:
        if switch:
            print('no converged')
            np.set_printoptions(linewidth=500)
            print(new_n.round(3))
    return(ham-mu*np.identity(ns))

def get_ham(cnp.ndarray[cnp.int64_t,ndim=2] wf,hop,int nwf,cnp.ndarray[cnp.float64_t,ndim=2] U,
            cnp.ndarray[cnp.float64_t,ndim=2] J, int ns,cnp.ndarray[cnp.float64_t,ndim=1] F,
            int l=3,sw_all_g=True,sw_ph=False):
    """
    get many-body hamiltonian
    """
    cdef long i,j,j0,k,tmp,i0,i2,isgn,j2,jsgn,m1,m2,m3,m4
    cdef cnp.ndarray[cnp.int64_t,ndim=1] ist,jst,tmp1
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] ham=np.zeros((nwf,nwf),dtype='c16')
    cdef cnp.ndarray[cnp.float64_t,ndim=3] cp

    cp=gencp(l+1,l,l)
    for i,ist in enumerate(wf):
        for j0,jst in enumerate(wf[i:]):
            j=j0+i
            tmp=abs(ist-jst).sum()
            if(tmp==0): #on-site energy and interactions
                tmp1=np.where(ist==1)[0] #take occupy states
                #print(tmp1)
                ham[i,i]=hop[tmp1,tmp1].sum() #sum of on-site energy
                for k, i0 in enumerate(tmp1): #consider U and U'
                    if(i0<ns//2): #up spin
                        i2=i0
                        isgn=1
                    else: #down spin
                        i2=i0-ns//2
                        isgn=-1
                    for j0 in tmp1[k+1:]: #consider only en0>st0
                        if(j0<ns//2): #up spin
                            j2=j0
                            jsgn=1
                        else: #down spin
                            j2=j0-ns//2
                            jsgn=-1
                        if isgn*jsgn==1: #spin parallel
                            ham[i,i]=ham[i,i]+U[j2,i2]-J[j2,i2]
                        else:
                            ham[i,i]=ham[i,i]+U[j2,i2]
            elif(tmp==2): #hoppings
                tmp1=ist-jst
                j2=np.where(tmp1==-1)[0][0]
                i2=np.where(tmp1==1)[0][0]
                sgn=(-1)**(jst[:j2].sum()+ist[:i2].sum())
                ham[i,j]=sgn*hop[i2,j2]
            elif(tmp==4): #four operators
                tmp1=ist-jst
                if(tmp1[:ns//2].sum()==0): #spin conservation
                    m3=np.where(tmp1==-1)[0][0] #1st one
                    m4=np.where(tmp1==-1)[0][1] #2nd one
                    m1=np.where(tmp1==1)[0][0]  #1st one
                    m2=np.where(tmp1==1)[0][1]  #2nd one
                    nst=jst[:m3].sum()+jst[:m4].sum()-1
                    nen=ist[:m1].sum()+ist[:m2].sum()-1
                    sgn=(-1)**(nst+nen)
                    if(abs(tmp1[:ns//2]+tmp1[ns//2:]).sum()==0): #Hund's couplings m1=m4,m2=m3
                        ham[i,j]=J[m1,m3]*sgn
                    elif(sw_ph and (abs(tmp1[:ns//2]-tmp1[ns//2:]).sum()==0)): #pair hoppings m1=m2,m3=m4
                        ham[i,j]=J[m1,m3]*sgn*(-1)**abs(m1-m3)
                    elif(sw_all_g):
                        if(m3>=ns//2):
                            m3=m3-ns//2
                        if(m4>=ns//2):
                            m4=m4-ns//2
                        if(m1>=ns//2):
                            m1=m1-ns//2
                        if(m2>=ns//2):
                            m2=m2-ns//2
                        if((m1+m2-(m3+m4))==0): #delta(m1+m2,m3+m4)
                            if(abs(tmp1[:ns//2]).sum()==2): #spin anti-parallel
                                G=(-1)**abs(m1-m3)*(F*cp[m1,m3]*cp[m2,m4]).sum()
                            else: #spin parallel
                                G=((-1)**abs(m1-m3)*(F*cp[m1,m3]*cp[m2,m4]).sum()
                                   -(-1)**abs(m2-m3)*(F*cp[m2,m3]*cp[m1,m4]).sum())
                                #print(m1,m2,m4,m3,G.round(4))
                            ham[i,j]=G*sgn
            ham[j,i]=ham[i,j].conjugate()
    return(ham)
