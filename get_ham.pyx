# -*^ coding: utf-8 -*-
# cython: profile=False
import numpy as np
cimport cython
cimport numpy as cnp
import sympy as syp
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
    cdef long lmax=p-1, ll, mm, pp
    def get_gaunt(int p,int m,int l):
        return float(gaunt(lmax,p,lmax,-l,l-m,m)*2.*syp.sqrt(syp.pi/(2.*p+1.)))*(-1)**l
    cp=np.array([[[get_gaunt(2*pp,ll-l,mm-l) for pp in range(p)] 
                  for mm in range(2*m+1)] for ll in range(2*l+1)])
    return cp

def UJ(cnp.ndarray[cnp.float64_t,ndim=1] F,int l=3):
    """
    generate onsite interaction U,J
    """
    cdef long m1,m2,i,j,lmax=2*l+1
    cdef cnp.ndarray[cnp.float64_t,ndim=2] U,J
    cdef cnp.ndarray[cnp.float64_t,ndim=3] cp

    cp=gencp(l+1,l,l)
    Ulm=lambda m1,m2,cp,F: (F[:l+1]*cp[m1,m1]*cp[m2,m2]).sum()
    Jlm=lambda m1,m2,cp,F: (F[:l+1]*cp[m1,m2]**2).sum()
    U=np.array([[float(Ulm(i,j,cp,F)) for i in range(lmax)] for j in range(lmax)])
    J=np.array([[float(Jlm(i,j,cp,F)) for i in range(lmax)] for j in range(lmax)])
    J=J-np.diag(J.diagonal())
    return U,J

def get_dU(double F0,int l=3):
    cdef long m1,m2,i,j,lmax=2*l+1
    cdef cnp.ndarray[cnp.float64_t,ndim=2] dU
    cdef cnp.ndarray[cnp.float64_t,ndim=3] cp

    cp=gencp(l+1,l,l)
    dulm= lambda m1,m2,cp,F0: (F0*cp[m1,m1,0]*cp[m2,m2,0])
    dU=np.array([[float(dulm(i,j,cp,F0)) for i in range(lmax)] for j in range(lmax)])
    return dU

def get_J(cnp.ndarray[cnp.int64_t,ndim=2] wf,int nwf,cnp.ndarray[cnp.complex128_t,ndim=2] uni,
          cnp.ndarray[cnp.int64_t,ndim=2] instates, cnp.ndarray[cnp.int64_t,ndim=2] sp1, int eigmax, int lmax=3):
    """
    calculate total/orbital/spin angular momentum J,L,S and generate magnetic dipole
    details of argments
    ---------------------
    wf:initial basis of wave functions
    nwf: number of basis
    uni: eigen functions of multibody hamiltonian
    instates: details of wf (setting of electrons in wf)
    sp1: instates label
    eigmax: maximum value of eigenvalues to be considered
    lmax: maximum angular momentum of one body electron
    """
    cdef long i,j,k,spp_flag,spm_flag,lp_flag,lm_flag,lz,tmp
    cdef double lpnum,lmnum,upsign,dnsign
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] Jx,Jy,Jz
    cdef cnp.ndarray[cnp.float64_t,ndim=2] mag_specm,Lsq,Ssq
    cdef cnp.ndarray[cnp.int64_t] ist,jst,lflipp,lflipm,sflipp,sflipm
    cdef cnp.ndarray[cnp.int64_t,ndim=1] Lz0,Sz0

    cdef cnp.ndarray[cnp.float64_t,ndim=2] Lp0=np.zeros((nwf,nwf)),Lm0=np.zeros((nwf,nwf))
    cdef cnp.ndarray[cnp.float64_t,ndim=2] Sp0=np.zeros((nwf,nwf)),Sm0=np.zeros((nwf,nwf))
    cdef nhf=int(len(sp1)//2)

    #J+,J-
    for i,ist in enumerate(wf):
        for k in np.where(ist==1)[0]:
            lz=sp1[k][0]
            if lz!=lmax and ist[k+1]==0:
                #lz>lz+1 setting
                lpnum=np.sqrt((lmax-lz)*(lmax+lz+1.))
                lflipp=ist.copy()
                lflipp[k]=0
                lflipp[k+1]=1
                lp_flag=1
            else:
                lp_flag=0
            if lz!=-lmax and ist[k-1]==0:
                #lz>lz-1 setting
                lmnum=np.sqrt((lmax+lz)*(lmax-lz+1.))
                lflipm=ist.copy()
                lflipm[k]=0
                lflipm[k-1]=1
                lm_flag=1
            else:
                lm_flag=0
            if k<nhf:
                #spin flip u>d setting
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
                #spin flip d>u setting
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
    Lz0=np.array([a[:,0].sum() for a in sp1[instates]])
    Sz0=np.array([a[:,1].sum() for a in sp1[instates]])

    #obtain J of eigenvalues
    Jsq=.25*(Lp0+Lm0+(Sp0+Sm0)).dot(Lp0+Lm0+(Sp0+Sm0))-.25*(Lp0-Lm0+(Sp0-Sm0)).dot(Lp0-Lm0+(Sp0-Sm0))+np.diag(Lz0+.5*Sz0)**2
    Jeig=np.diagonal(uni[:,:eigmax].T.conjugate().dot(Jsq.dot(uni[:,:eigmax]))).real

    Jx=uni[:,:eigmax].T.conjugate().dot((.5*(Lp0+Lm0+2.*(Sp0+Sm0))).dot(uni[:,:eigmax]))
    Jy=uni[:,:eigmax].T.conjugate().dot((-.5j*(Lp0-Lm0+2.*(Sp0-Sm0))).dot(uni[:,:eigmax]))
    Jz=uni[:,:eigmax].T.conjugate().dot(np.diag(Lz0+Sz0).dot(uni[:,:eigmax]))
    #magnetic dipole
    m0=1e10*scconst.hbar/(2.*scconst.c*scconst.m_e) #e\AA is unity
    mag_spec=m0*m0*(abs(Jx)**2+abs(Jy)**2+abs(Jz)**2)
    Ssq=.25*(Sp0+Sm0).dot(Sp0+Sm0)-.25*(Sp0-Sm0).dot(Sp0-Sm0)+.25*np.diag(Sz0)**2
    eig,uni_s=sl.eigh(Ssq)
    S_delta=uni_s.dot(uni_s.conjugate().T)
    Lsq=.25*(Lp0+Lm0).dot(Lp0+Lm0)-.25*(Lp0-Lm0).dot(Lp0-Lm0)+np.diag(Lz0)**2 #L^2
    return mag_spec,Lsq,Lz0,Ssq,Jsq,.5*(np.sqrt(1.+4.*Jeig)-1.).round(2)

def gen_spec(cnp.ndarray[cnp.int64_t,ndim=2] wf,int nwf,cnp.ndarray[cnp.complex128_t,ndim=2] uni,
             cnp.ndarray[cnp.int64_t,ndim=2] instates,cnp.ndarray[cnp.int64_t,ndim=2] sp1,int eigmax,int lorb,
             cnp.ndarray[cnp.int64_t,ndim=2] JRGB):
    """
    calculate electric and magnetic dipole
    details of argments
    ---------------------
    wf:initial basis of wave functions
    nwf: number of basis
    uni: eigen functions of multibody hamiltonian
    instates: details of wf (setting of electrons in wf)
    sp1: instates label
    eigmax: maximum value of eigenvalues to be considered
    lorb: maximum angular momentum of one body electron     
    """
    #electric dipole_check
    cdef long i,j,l,l1,li,lj,j0,J,miz,mjz
    cdef cnp.ndarray[cnp.int64_t] ist,jst
    cdef cnp.ndarray[cnp.int64_t,ndim=1] Lz0
    cdef cnp.ndarray[cnp.float64_t,ndim=2] r_spec,Lsq, Ssq
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] rp_mat,rm_mat,rz_mat,rx0,ry0,rz0,rx,ry,rz

    #obtain J,L,S
    mag_spec,Lsq,Lz0,Ssq,Jsq,Jeig=get_J(wf,nwf,uni,instates,sp1,eigmax,lorb)
    print('calc mag dipole')
    # L       0   1   2   3   4   5   6   7   8   9  10  11  12
    L_label=['S','P','D','F','G','H','I','K','L','M','N','O','Q'] #wo J
    eig,uni_l=sl.eigh(Lsq) #eigval of l^2(= l(l+1))
    L_size=(0.5*(np.sqrt(1.+4.*eig)-1.)).round(2).astype(np.int64)
    Ssq2=uni_l.T.conjugate().dot(Ssq.dot(uni_l))

    #obtain <i|r|j>
    l1=0
    lmat=[]
    for i in range(200):
        l=np.where(eig.round(2)==eig.round(2)[l1])[0].size
        l1=l1+l
        lmat.append(l)
        if l1==eig.size:
            break
    rot_mat=np.zeros((nwf,nwf),dtype='c16')

    l1=0
    S_size=[]
    for l in lmat:
        bs=Ssq2[l1:l1+l,l1:l1+l].copy()
        essq,uni_ssq=sl.eigh(bs)
        rot_mat[l1:l1+l,l1:l1+l]=uni_ssq.copy()
        l1=l1+l
        S_size.extend(.5*(np.sqrt(1.+4.*essq)-1.))
    S_size=np.array(S_size).round(2)
    uni_ls=uni_l.dot(rot_mat)
    Jsq2=uni_ls.T.conjugate().dot(Jsq.dot(uni_ls))
    Lz=uni_ls.T.conjugate().dot(np.diag(Lz0).dot(uni_ls))

    l1=0
    smat=[]
    for l in lmat:
        SS=S_size[l1:l1+l].round(2)
        s1=0
        for i in range(200):
            s=np.where(SS==SS[s1])[0].size
            s1=s1+s
            smat.append(s)
            if s1==SS.size:
                break
        l1=l1+l

    rot_mat=np.zeros((nwf,nwf),dtype='c16')
    rot_J=np.zeros((nwf,nwf),dtype='c16')
    s1=0
    J_size=[]
    Lz_size=[]
    for s in smat:
        blz=Lz[s1:s1+s,s1:s1+s].copy()
        eig_lz,uni_lz=sl.eigh(blz)
        rot_mat[s1:s1+s,s1:s1+s]=uni_lz.copy()
        bj=Jsq2[s1:s1+s,s1:s1+s].copy()
        eig_j,uni_j=sl.eigh(bj)
        rot_J[s1:s1+s,s1:s1+s]=uni_j.copy()
        #print(eig_lz.round(6),s)
        #print(eig_j.round(6),s)
        s1=s1+s
        Lz_size.extend(eig_lz.round(2))
        J_size.extend(.5*(np.sqrt(1.+4.*eig_j)-1.))
    Lz_size=np.array(Lz_size,dtype=np.int64)
    J_size=np.array(J_size).round(2)
    uni_lls=uni_ls.dot(rot_mat)
    uni_J=uni_ls.dot(rot_J)

    RSuni=abs(uni[:,:eigmax].T.dot(uni_J))**2
    Lu=np.unique(L_size)
    Su=np.unique(S_size)
    Ju=np.unique(J_size)
    f=open('ck_LS.txt','w')
    Jcolor=np.zeros((eigmax,3))
    for i,(ckuni,Je) in enumerate(zip(RSuni,Jeig)):
        LSW=[]
        for Li in Lu:
            for Si in Su:
                for Ji in Ju:
                    Lcu=ckuni[np.where(L_size==Li)[0]]
                    Sl=S_size[np.where(L_size==Li)[0]]
                    Jl=J_size[np.where(L_size==Li)[0]]
                    Scu=Lcu[np.where(Sl==Si)[0]]
                    Jls=Jl[np.where(Sl==Si)[0]]
                    LS_Weight=Scu[np.where(Jls==Ji)[0]].sum().round(3)
                    for i2,Jc in enumerate(JRGB):
                        if abs(Jc[0]-Ji)<1.e-2 and abs(Jc[1]-Si)<1.e-2 and Jc[2]==Li:
                            Jcolor[i,i2]=LS_Weight
                        else:
                            pass
                    if LS_Weight>1.e-2:
                        RS=('^(%d)%s_%3.1f'%(2*Si+1,L_label[Li],abs(Ji)) if (2*Ji%2!=0) 
                            else '^(%d)%s_%d'%(2*Si+1,L_label[Li],abs(Ji)))
                        LSW.append([RS,LS_Weight])
        f.write('eig_number %d,J=%4.1f\n'%(i,Je))
        for lw in LSW:
            f.write('%s: %4.2f, '%tuple(lw))
        f.write('\n')
    f.close()
    rp_mat=np.zeros((nwf,nwf),dtype='c16')
    rm_mat=np.zeros((nwf,nwf),dtype='c16')
    rz_mat=np.zeros((nwf,nwf),dtype='c16')
    for i,(li,miz,si) in enumerate(zip(L_size,Lz_size,S_size)):
        for j0, (lj,mjz,sj) in enumerate(zip(L_size[i:],Lz_size[i:],S_size[i:])):
            if abs(si-sj)<1e-3 and abs(li-lj)==1:
                j=j0+i
                rp_mat[i,j]=complex(gaunt(lj,1,li,miz,1,mjz))
                rm_mat[i,j]=complex(gaunt(lj,1,li,miz,-1,mjz))
                rz_mat[i,j]=complex(gaunt(lj,1,li,miz,0,mjz))
                rp_mat[j,i]=-rm_mat[i,j].conjugate()
                rm_mat[j,i]=-rp_mat[i,j].conjugate()
                rz_mat[j,i]=rz_mat[i,j].conjugate()
    rx0=uni_lls.dot((rm_mat-rp_mat).dot(uni_lls.T.conjugate()))/np.sqrt(2.)
    ry0=1j*uni_lls.dot((rp_mat+rm_mat).dot(uni_lls.T.conjugate()))/np.sqrt(2.)
    rz0=uni_lls.dot(rz_mat.dot(uni_lls.T.conjugate()))
    rx=uni[:,:eigmax].T.conjugate().dot(rx0.dot(uni[:,:eigmax]))
    ry=uni[:,:eigmax].T.conjugate().dot(ry0.dot(uni[:,:eigmax]))
    rz=uni[:,:eigmax].T.conjugate().dot(rz0.dot(uni[:,:eigmax]))
    r_spec=abs(rx)**2+abs(ry)**2+abs(rz)**2
    r_spec2=abs(rx.dot(rx))+abs(ry.dot(ry))+abs(rz.dot(rz))
    print('calc elec dipole')
    return mag_spec,r_spec,r_spec2,Jeig,Jcolor

def get_spectrum(int nwf,cnp.ndarray[cnp.int64_t,ndim=2] wf,cnp.ndarray[cnp.float64_t,ndim=1] eig,
                 cnp.ndarray[cnp.complex128_t,ndim=2] eigf,cnp.ndarray[cnp.int64_t,ndim=2] instates,
                 cnp.ndarray[cnp.int64_t,ndim=2] sp1,double erange, double temp, int lorb, 
                 cnp.ndarray[cnp.int64_t,ndim=2] JRGB,double ie_max=2.0, double de_max=3.0,
                 double de_min=1.e-3, double id=1.0e-3, int wmesh=2000):
    """
    generate spectrum
    """
    cdef long eig_int_max=(np.where(eig<=2.*erange+eig[0])[0]).size
    cdef cnp.ndarray[cnp.float64_t,ndim=1] chi,chi2,dfunc,deig,wlen=np.linspace(0,erange,wmesh)

    rsq=0.4376**2
    mnn2,mnn,mnn3,Jeig,Jcolor=gen_spec(wf,nwf,eigf,instates,sp1,eig_int_max,lorb,JRGB)
    arrows=[]
    for i,mn in enumerate(mnn):
        for j0,m in enumerate(mn[i+1:]):
            j=i+j0+1
            if (abs(eig[i]-eig[0])<ie_max and abs(eig[j]-eig[0])<erange and de_min<abs(eig[j]-eig[i])<de_max) and m>1.e-3:
                print(eig[i]-eig[0],eig[i]-eig[j],i,j)
                arrows.append([i,j])
            else:
                pass
    arrows_mag=[]
    for i,mn in enumerate(mnn2):
        for j0,m in enumerate(mn[i+1:]):
            j=i+j0+1
            if (abs(eig[i]-eig[0])<ie_max and abs(eig[j]-eig[0])<erange and de_min<abs(eig[j]-eig[i])<de_max) and m>1.e-5:
                print(eig[i]-eig[0],eig[i]-eig[j],i,j)
                arrows_mag.append([i,j])
            else:
                pass

    fig=plt.figure()
    ax11=fig.add_subplot(321)
    maps=ax11.imshow(mnn.round(3),cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(maps,ax=ax11)
    ax12=fig.add_subplot(322)
    maps=ax12.imshow(mnn3.round(3),cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(maps,ax=ax12)

    ax21=fig.add_subplot(323)
    warray=np.linspace(0,2*erange,eig_int_max)
    wl,el=np.meshgrid(warray,eig[:eig_int_max]-eig[0])
    we_uni=id/((wl-el)**2+id**2)
    pmap=we_uni.T.dot((mnn+mnn2).dot(we_uni))
    maps=ax21.contourf(wl,wl.T,pmap,levels=100,cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(maps,ax=ax21)
    ax22=fig.add_subplot(324)
    pmap=we_uni.T.dot((mnn3+mnn2).dot(we_uni))
    maps=ax22.contourf(wl,wl.T,pmap,levels=100,cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(maps,ax=ax22)

    ax3=fig.add_subplot(313)
    ax3.plot(range(eig_int_max),mnn[0,:])
    fig.savefig('mnn_map.png')

    mnn3=mnn3.flatten()
    mnn2=mnn2.flatten()
    eig0=(eig-eig[0])[:eig_int_max]
    func=np.exp(-eig0/temp)
    func=func/func.sum()
    eigf0=eigf.T[:eig_int_max]
    deig=np.array([[e1-e2 for e1 in eig0] for e2 in eig0]).flatten()
    dfunc=np.array([[e2-e1 for e1 in func] for e2 in func]).flatten()
    chi=rsq*np.array([(mnn3*dfunc/(complex(iw,id)+deig)).sum().imag for iw in wlen])
    chi2=np.array([(mnn2*dfunc/(complex(iw,id)+deig)).sum().imag for iw in wlen])
    return wlen,chi,chi2,Jeig,Jcolor,arrows,arrows_mag

def get_HF_full(int ns, int ne, init_n, ham0, cnp.ndarray[cnp.float64_t,ndim=2] U,
                cnp.ndarray[cnp.float64_t,ndim=2] J, cnp.ndarray[cnp.float64_t,ndim=2] dU,
                cnp.ndarray[cnp.float64_t,ndim=1] F, double temp=1.0e-9,double eps=1.0e-6,
                int itemax=1000,switch=True,lorb=3):
    """
    calculate MF hamiltonian with full Coulomb interactions
    """
    cdef long i,j,k,l,m
    cdef double mu
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] ham, ham_I=np.zeros((ns,ns),dtype='c16')
    cdef cnp.ndarray[cnp.float64_t,ndim=3] cp
    #original interaction g_m1m2m3m4c^+_m1c^+_m2c_m4c_m3
    cp=gencp(lorb+1,lorb,lorb)
    G=lambda m1,m2,m3,m4,cp,F:(-1)**abs(m1-m3)*(F[:lorb+1]*cp[m1,m3]*cp[m2,m4]).sum()
    if len(init_n)<ns:
        n1=np.zeros((ns,ns),dtype='c16')
        i=0
        flags=True
        for f in open('dmat_QSGW','r'):
            if f.find('spin')==-1:
                tmp=f.split()
                if len(tmp)>1:
                    for j,tp in enumerate(tmp):
                        if flags:
                            if i<ns//2:
                                n1[i,j]=float(tp)
                            else:
                                n1[i-ns//2,j]=n1[i-ns//2,j]+1j*float(tp)
                        else:
                            if i<ns//2:
                                n1[i+ns//2,j+ns//2]=float(tp)
                            else:
                                n1[i,j+ns//2]=n1[i,j+ns//2]+1j*float(tp)
                    if i<ns-1:
                        i+=1
                    else:
                        i=0
                        flags=False
    else:
        n1=np.diag(np.array(init_n))
    for k in range(itemax):
        ham_I*=0.
        #i,m,j,l>m1,m2,m3,m4
        for i in range(ns//2):
            # intra orb (consider i==j and m==l (m1==m3 and m2==m4))
            ham_I[i,i]=((U[i,:]*n1.diagonal()[ns//2:]).sum()
                          +np.delete((U[i,:]-J[i,:]+dU[i,:])*n1.diagonal()[:ns//2],i).sum())
            ham_I[i+ns//2,i+ns//2]=((U[i,:]*n1.diagonal()[:ns//2]).sum()
                                      +np.delete((U[i,:]-J[i,:]+dU[i,:])*n1.diagonal()[ns//2:],i).sum())
            for j in range(i+1,ns//2): #inter orb (m1!=m3)
                #first we consider i==l and j==m (m1=m4 and m2=m3)
                ham_I[i,j]=J[i,j]*n1[j+ns//2,i+ns//2]+(J[i,j]-U[i,j]-dU[i,j])*n1[j,i]
                ham_I[i+ns//2,j+ns//2]=J[i,j]*n1[i,j]+(J[i,j]-U[i,j]-dU[i,j])*n1[j+ns//2,i+ns//2]
                #consider other m1+m2=m3+m4
                for l in range(i+1,ns//2): #m1!=m4
                    m=l+j-i #i+m=j+l
                    if m<ns//2:
                        if i==2*lorb-m: #m1=-m2, m3=-m4 (correspond to pair hoppings)
                            ham_I[i,j]=ham_I[i,j]+(-1)**(j-i)*J[i,j]*n1[m+ns//2,l+ns//2]
                            ham_I[i,j]=ham_I[i,j]+((-1)**(j-i)*J[i,j]-G(i,m,l,j,cp,F))*n1[m,l]
                            ham_I[i+ns//2,j+ns//2]=ham_I[i+ns//2,j+ns//2]+(-1)**(j-i)*J[i,j]*n1[m,l]
                            ham_I[i+ns//2,j+ns//2]=ham_I[i+ns//2,j+ns//2]+((-1)**(j-i)*J[i,j]-G(i,m,l,j,cp,F))*n1[m+ns//2,l+ns//2]
                        else:
                            ham_I[i,j]=ham_I[i,j]+G(i,m,j,l,cp,F)*n1[m+ns//2,l+ns//2]
                            ham_I[i,j]=ham_I[i,j]+(G(i,m,j,l,cp,F)-G(i,m,l,j,cp,F))*n1[m,l]
                            ham_I[i+ns//2,j+ns//2]=ham_I[i+ns//2,j+ns//2]+G(i,m,j,l,cp,F)*n1[m,l]
                            ham_I[i+ns//2,j+ns//2]=ham_I[i+ns//2,j+ns//2]+(G(i,m,j,l,cp,F)-G(i,m,l,j,cp,F))*n1[m+ns//2,l+ns//2]
                            #ham_I[i,j+ns//2]=ham_I[i,j+ns//2]+G(i,m,l,j,cp,F)*n1[m+ns//2,l]
                            #ham_I[i+ns//2,j]=ham_I[i+ns//2,j]+G(i,m,l,j,cp,F)*n1[m,l+ns//2]
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

def get_ham(cnp.ndarray[cnp.int64_t,ndim=2] wf, hop, int nwf, cnp.ndarray[cnp.float64_t,ndim=2] U,
            cnp.ndarray[cnp.float64_t,ndim=2] J, int ns, cnp.ndarray[cnp.float64_t,ndim=1] F,
            int l=3,sw_all_g=True):
    """
    get many-body hamiltonian
    """
    cdef long i,j,j0,k,tmp,i0,i2,isgn,j2,jsgn,m1,m2,m3,m4
    cdef cnp.ndarray[cnp.int64_t,ndim=1] ist,jst,tmp1
    cdef cnp.ndarray[cnp.complex128_t,ndim=2] ham=np.zeros((nwf,nwf),dtype='c16')
    cdef cnp.ndarray[cnp.float64_t,ndim=3] cp

    cp=gencp(l+1,l,l)
    G=lambda m1,m2,m3,m4,cp,F:(-1)**abs(m1-m3)*(F[:l+1]*cp[m1,m3]*cp[m2,m4]).sum()
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
                        else: #spin anti parallel
                            ham[i,i]=ham[i,i]+U[j2,i2]
            elif(tmp==2): #hoppings one body (soc and crystal field)
                tmp1=ist-jst
                j2=np.where(tmp1==-1)[0][0]
                i2=np.where(tmp1==1)[0][0]
                sgn=(-1)**(jst[:j2].sum()+ist[:i2].sum())
                ham[i,j]=sgn*hop[i2,j2]
            elif(tmp==4): #four operators two body
                tmp1=ist-jst
                if(tmp1[:ns//2].sum()==0): #spin conservation rule
                    m3=np.where(tmp1==-1)[0][0] #1st one anihilate
                    m4=np.where(tmp1==-1)[0][1] #2nd one anihilate
                    m1=np.where(tmp1==1)[0][0]  #1st one create
                    m2=np.where(tmp1==1)[0][1]  #2nd one create
                    nst=jst[:m3].sum()+jst[:m4].sum()-1 #sign flip from anihilation op.
                    nen=ist[:m1].sum()+ist[:m2].sum()-1 #sign flip from creation op. 
                    sgn=(-1)**(nst+nen) #total flip
                    if(abs(tmp1[:ns//2]+tmp1[ns//2:]).sum()==0): #Hund's couplings m1=m4,m2=m3
                        ham[i,j]=J[m1,m3]*sgn
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
                            ham[i,j]=G(m1,m2,m3,m4,cp,F)*sgn
                            if(abs(tmp1[:ns//2]).sum()==2): #spin anti-parallel
                                pass
                            else: #spin parallel
                                ham[i,j]=ham[i,j]-G(m2,m1,m3,m4,cp,F)*sgn
            ham[j,i]=ham[i,j].conjugate()
    return(ham)
