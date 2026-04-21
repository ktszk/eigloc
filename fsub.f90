subroutine get_ham(ham,wf,hop,Umat,Jmat,cp,F,nwf,ns,lmax) bind(C)
! Constructs the many-body Hamiltonian matrix in the Slater determinant basis.
!
! Diagonal elements:
!   - Single-particle (crystal-field / hopping) on-site energy
!   - Pairwise Coulomb interaction via Kanamori parameters:
!       parallel spins   (jspn*kspn=+1): U - J  (Hund's rule)
!       antiparallel spins (jspn*kspn=-1): U
!
! Off-diagonal elements are classified by the L1-norm of the difference vector
! tmp1 = wf(:,i) - wf(:,j):
!   |tmp1|=2 : single-particle hopping    c^+_l c_m
!   |tmp1|=4 : two-electron interaction   c^+_m1 c^+_m2 c_m4 c_m3
!              (spin exchange or orbital pair hopping)
!
! Spin-orbital ordering convention:
!   indices    1 .. ns/2  : spin-up   orbitals (orbital index = spin-orbital index)
!   indices ns/2+1 .. ns  : spin-down orbitals (orbital index = spin-orbital index - ns/2)
!
! IMPORTANT: The caller must pre-initialize `ham` to zero before calling this
!            subroutine, because matrix elements that vanish are never explicitly
!            assigned.
!
! Arguments:
!   ham  (out)  nwf x nwf complex Hamiltonian matrix
!   wf   (in)   ns x nwf occupation-number matrix; wf(j,i)=1 if spin-orbital j
!               is occupied in Slater determinant i
!   hop  (in)   ns x ns single-particle (crystal-field / hopping) matrix
!   Umat (in)   (ns/2) x (ns/2) direct Coulomb integral matrix U(m,l) (orbital indices)
!   Jmat (in)   (ns/2) x (ns/2) Hund exchange integral matrix J(m,l) (orbital indices)
!   cp   (in)   Gaunt coefficients c^k(m',m), shape (lmax+1, 2*lmax+1, 2*lmax+1)
!   F    (in)   Slater integrals F^k, shape (lmax+1); k = 0, 2, ..., 2*lmax
!   nwf  (in)   number of Slater determinants (many-body basis size)
!   ns   (in)   number of spin-orbitals = 2*(2*lmax+1)
!   lmax (in)   maximum angular momentum quantum number
  use,intrinsic:: iso_fortran_env, only: int64,real64
  implicit none
  integer(int64),intent(in):: nwf, lmax, ns
  integer(int64),intent(in),dimension(ns,nwf):: wf
  complex(real64),intent(in),dimension(ns,ns):: hop
  real(real64),intent(in),dimension(int(ns/2),int(ns/2)):: Umat,Jmat
  real(real64),intent(in),dimension(lmax+1):: F
  real(real64),intent(in),dimension(lmax+1,2*lmax+1,2*lmax+1):: cp
  complex(real64),intent(out),dimension(nwf,nwf):: ham

  integer(int64) :: i  ! only the parallel-do loop variable lives at subroutine scope

  ! Parallelize over rows: each thread independently fills its assigned row i.
  ! Row i writes only to column i (lower triangle: ham(j,i), j>i) and row i
  ! (upper triangle: ham(i,j)), so there is no write conflict between threads.
  ! schedule(dynamic) balances load since row i has (nwf-i) off-diagonal elements.
  ! All working variables are declared inside block constructs below, making
  ! them automatically thread-private without a private(...) clause.
  !$omp parallel do schedule(dynamic)
  do i=1,nwf

    ! Outer block: scopes j so it is thread-private, reused as the loop
    ! variable in both the diagonal section and the off-diagonal section.
    block
      integer(int64) :: j

      ! ----------------------------------------------------------------
      ! Diagonal element ham(i,i): on-site energy + Coulomb interaction
      ! ----------------------------------------------------------------
      block
        integer(int64) :: k, l, m, jspn, kspn
        complex(real64) :: ctmp

        ctmp = 0.0d0
        ! Single-particle contribution: sum on-site energies of occupied orbitals
        do j=1,ns
          if(wf(j,i)==1) ctmp = ctmp + hop(j,j)
        end do

        ! Two-body Coulomb contribution (Kanamori U, J)
        do j=1,ns
          if(wf(j,i)==1)then
            ! Extract spin (jspn = +/-1) and orbital index (l) for spin-orbital j
            if(j<=ns/2)then
              jspn=1; l=j       ! spin-up
            else
              jspn=-1; l=j-ns/2 ! spin-down
            end if
            do k=j+1,ns
              if(wf(k,i)==1)then
                if(k<=ns/2)then
                  kspn=1; m=k
                else
                  kspn=-1; m=k-ns/2
                end if
                if(jspn*kspn==1)then  ! parallel spins: U-J
                  ctmp = ctmp + Umat(m,l) - Jmat(m,l)
                else                   ! antiparallel spins: U
                  ctmp = ctmp + Umat(m,l)
                end if
              end if
            end do
          end if
        end do
        ham(i,i) = ctmp
      end block

      ! ----------------------------------------------------------------
      ! Off-diagonal elements ham(i,j) and ham(j,i) for j > i
      !
      ! Difference vector tmp1 = wf(:,i) - wf(:,j):
      !   tmp1(k) = +1 : spin-orbital k occupied in i, vacant in j
      !                   -> c_k annihilates this electron going from i to j
      !   tmp1(k) = -1 : spin-orbital k vacant in i, occupied in j
      !                   -> c^+_k creates this electron going from i to j
      ! ----------------------------------------------------------------
      do j=i+1,nwf
        ! All working variables for the off-diagonal element (i,j) are
        ! declared here; each thread's j-loop iteration gets its own stack copy.
        block
          integer(int64) :: tmp, tmp1(ns), spn_diff, orb_check
          integer(int64) :: l, m, k, sw1, sw2, m1, m2, m3, m4, nst, nen, sgn

          tmp1 = wf(:,i) - wf(:,j)
          tmp  = sum(abs(tmp1))

          if(tmp==2)then
            ! Single-particle hopping: identify created orbital l (tmp1=+1)
            ! and annihilated orbital m (tmp1=-1)
            do k=1,ns
              if(tmp1(k)==1)  l=k
              if(tmp1(k)==-1) m=k
            end do
            ! Fermionic sign: count occupied orbitals passed by the operators
            sgn = (-1)**(sum(wf(:m,j)) + sum(wf(:l,i)))
            ham(j,i) = hop(l,m)*sgn

          else if(tmp==4)then
            ! Two-body interaction: c^+_m1 c^+_m2 c_m4 c_m3
            ! First check: total spin-up electron number is conserved
            spn_diff = sum(tmp1(:ns/2))
            if(spn_diff==0)then
              ! Identify the four spin-orbitals:
              !   m1, m2 : created   (tmp1=+1, occupied in i)
              !   m3, m4 : annihilated (tmp1=-1, occupied in j)
              sw1=0; sw2=0
              do k=1,ns
                if(tmp1(k)==-1)then
                  if(sw1==0)then; m3=k; sw1=1; else; m4=k; end if
                else if(tmp1(k)==1)then
                  if(sw2==0)then; m1=k; sw2=1; else; m2=k; end if
                end if
              end do

              ! Fermionic sign for the two-body operator c^+_m1 c^+_m2 c_m4 c_m3.
              ! nst counts occupied orbitals passed by c_m3 and c_m4 in state j.
              ! nen counts occupied orbitals passed by c^+_m1 and c^+_m2 in state i.
              ! Subtract 2 to correct for the double-counting of the acted-upon
              ! orbitals themselves (m3, m4 are occupied in j; m1, m2 in i).
              nst = sum(wf(:m3,j)) + sum(wf(:m4,j))
              nen = sum(wf(:m1,i)) + sum(wf(:m2,i))
              sgn = (-1)**(nst+nen-2)

              ! Second check: if tmp1(m)+tmp1(m+ns/2)=0 for all orbitals m,
              ! the interaction is pure spin exchange (no net orbital-charge transfer)
              orb_check = sum(abs(tmp1(:ns/2) + tmp1(ns/2+1:)))
              if(orb_check==0)then
                ! Pure spin exchange: <m1 m2|J|m3 m4> = J(m3,m1)
                ham(j,i) = Jmat(m3,m1)*sgn
              else
                ! Pair hopping: convert spin-orbital indices to orbital-only indices
                if(m3>ns/2) m3=m3-ns/2
                if(m4>ns/2) m4=m4-ns/2
                if(m1>ns/2) m1=m1-ns/2
                if(m2>ns/2) m2=m2-ns/2
                ! Orbital momentum must be conserved: m1+m2 = m3+m4
                if(m1+m2==m3+m4)then
                  ! Slater-Condon two-electron matrix element G(m1,m2,m3,m4)
                  ham(j,i) = G(m1,m2,m3,m4)*sgn
                  ! Antisymmetrization correction for same-spin pair hopping:
                  ! when both differing spin-orbitals are NOT in the same spin sector
                  ! (sum(|tmp1(:ns/2)|) /= 2 means mixed-spin or same-spin case that
                  ! requires the exchange-permuted term), subtract G(m2,m1,m3,m4).
                  if(sum(abs(tmp1(:int(ns*0.5d0)))).ne.2)then
                    ham(j,i) = ham(j,i) - G(m2,m1,m3,m4)*sgn
                  end if
                end if
              end if
            end if
          end if

          ! Exploit Hermitian symmetry: ham(i,j) = conjg(ham(j,i))
          ham(i,j) = conjg(ham(j,i))
        end block
      end do
    end block
  end do
  !$omp end parallel do

contains
  ! Evaluates the Slater-Condon two-electron Coulomb matrix element:
  !
  !   G(m1,m2,m3,m4) = sum_{k=0,2,...,2*lmax} (-1)^|m1-m3| * F^k * c^k(m3,m1) * c^k(m4,m2)
  !
  ! where F^k are Slater integrals and c^k(m',m) are Gaunt coefficients.
  ! This gives the direct interaction <m1 m2 | 1/r_{12} | m3 m4> in the
  ! spherical harmonic basis, using the Slater-Condon-Shortley convention.
  ! m1,m2 are creation operator orbital indices; m3,m4 are annihilation operator indices.
  ! Note: lmax, F, and cp are accessed from the host subroutine scope.
  !
  ! The loop runs i=1..lmax+1 corresponding to k=0,2,...,2*lmax (even k only),
  ! because odd-k Gaunt coefficients vanish for same-l matrix elements.
  ! The (-1)^|m1-m3| phase implements the Condon-Shortley phase convention.
  real(real64) function G(m1,m2,m3,m4)
    implicit none
    integer(int64),intent(in):: m1,m2,m3,m4
    integer(int64) i

    G=0.0d0
    do i=1,lmax+1
       G=G+(-1)**abs(m1-m3)*(F(i)*cp(i,m3,m1)*cp(i,m4,m2))
    end do
  end function G
end subroutine get_ham
