subroutine get_ham(ham,wf,hop,Umat,Jmat,cp,F,nwf,ns,lmax) bind(C)
  implicit none
  integer(8),intent(in):: nwf, lmax, ns
  integer(8),intent(in),dimension(ns,nwf):: wf
  complex(8),intent(in),dimension(ns,ns):: hop
  real(8),intent(in),dimension(int(ns/2),int(ns/2)):: Umat,Jmat
  real(8),intent(in),dimension(lmax+1):: F
  real(8),intent(in),dimension(lmax+1,2*lmax+1,2*lmax+1):: cp
  complex(8),intent(out),dimension(nwf,nwf):: ham

  integer(8) i,j,k,l,m,sw1,sw2,tmp,tmp2,tmp1(ns)
  integer(8) m1,m2,m3,m4,nst,nen,sgn,jspn,kspn
  complex(8) ctmp

  !$omp parallel
  do i=1,nwf
     !diagonal
     !$omp single
     ctmp=0.0d0
     do j=1,ns
        if(wf(j,i)==1)then !if occupied states add onsite energy
           ctmp=ctmp+hop(j,j)
        end if
     end do

     do j=1,ns !consider U
        if(wf(j,i)==1)then
           if(j<=ns/2)then !relabel jspn: spin l: orbital
              jspn=1
              l=j
           else
              jspn=-1
              l=j-ns/2
           end if

           do k=j+1,ns
              if(wf(k,i)==1)then
                 if(k<=ns/2)then !relabel kspn: spin m: orbital
                    kspn=1
                    m=k
                 else
                    kspn=-1
                    m=k-ns/2
                 end if
                 if(jspn*kspn==1)then !if spin parallel add U-J
                    ctmp=ctmp+Umat(m,l)-Jmat(m,l)
                 else !if spin antiparallel add U
                    ctmp=ctmp+Umat(m,l)
                 end if
              end if
           end do
        end if
     end do
     ham(i,i)=ctmp
     !$omp end single

     !off-diagonal 
     !$omp do private(tmp,tmp1,tmp2,l,m,m1,m2,m3,m4,k,sw1,sw2,nst,nen,sgn)
     do j=i+1,nwf
        tmp1=wf(:,i)-wf(:,j)
        tmp=sum(abs(tmp1))
        if(tmp==2)then !inter orbital hoppings c^+_lc_m
           do k=1,ns
              if(tmp1(k)==1)then
                 l=k
              else if(tmp1(k)==-1)then
                 m=k
              end if
           end do
           sgn=(-1)**(sum(wf(:m,j))+sum(wf(:l,i)))
           ham(j,i)=hop(l,m)*sgn
        else if(tmp==4)then !2body interaction c^+_m1c^+_m2c_m4c_m3
           tmp2=sum(tmp1(:ns/2))
           if(tmp2==0)then !if 2body interaction sum(tmp1(:ns/2))=0
              sw1=0
              sw2=0
              do k=1,ns
                 if(tmp1(k)==-1)then
                    if(sw1==0)then
                       m3=k
                       sw1=1
                    else
                       m4=k
                    end if
                 else if(tmp1(k)==1)then
                    if(sw2==0)then
                       m1=k
                       sw2=1
                    else
                       m2=k
                    end if
                 end if
              end do
              nst=0
              do k=1,m3
                 nst=nst+wf(k,j)
              end do
              do k=1,m4
                 nst=nst+wf(k,j)
              end do
              nen=0
              do k=1,m1
                 nen=nen+wf(k,i)
              end do
              do k=1,m2
                 nen=nen+wf(k,i)
              end do
              sgn=(-1)**(nst+nen-2)
              tmp2=sum(abs(tmp1(:ns/2)+tmp1(ns/2+1:)))
              if(tmp2==0)then !exchange
                 ham(j,i)=Jmat(m3,m1)*sgn
              else !correspond to pair hoppings (total orbital momentum is conserved)
                 if(m3>ns/2)then !move down spin m_i label to orbital label
                    m3=m3-ns/2
                 end if
                 if(m4>ns/2)then
                    m4=m4-ns/2
                 end if
                 if(m1>ns/2)then
                    m1=m1-ns/2
                 end if
                 if(m2>ns/2)then
                    m2=m2-ns/2
                 end if
                 if((m1+m2-(m3+m4))==0)then
                    !write(80,'(4(1x,i4),1x,f8.4)')m1,m2,m3,m4,G(m1,m2,m3,m4)
                    ham(j,i)=G(m1,m2,m3,m4)*sgn
                    if(sum(abs(tmp1(:int(ns*0.5d0))))==2)then
                       continue
                    else
                       ham(j,i)=ham(j,i)-G(m2,m1,m3,m4)*sgn
                    end if
                 end if
              end if
           end if
        else
           cycle
        end if
        ham(i,j)=conjg(ham(j,i))
     end do !end set off-diagonal
     !$omp end do
  end do
  !$omp end parallel
contains
  real(8) function G(m1,m2,m3,m4)
    implicit none
    integer(8),intent(in):: m1,m2,m3,m4
    integer(8) i
    real(8) tmp

    G=0.0d0
    do i=1,lmax+1
       G=G+(-1)**abs(m1-m3)*(F(i)*cp(i,m3,m1)*cp(i,m4,m2))
    end do
  end function G
end subroutine get_ham
