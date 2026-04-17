subroutine ssss_psss_test
  use module_interest_eri
  implicit none

  character(len=4) :: class
  real(8) :: factor
  real(8) :: fint_inp(1000)

  integer :: la_inp, lb_inp, lc_inp, ld_inp
  real(8) :: alpha_inp, ax_inp, ay_inp, az_inp, anorm_inp
  real(8) :: beta_inp,  bx_inp, by_inp, bz_inp, bnorm_inp
  real(8) :: gamma_inp, cx_inp, cy_inp, cz_inp, cnorm_inp
  real(8) :: delta_inp, dx_inp, dy_inp, dz_inp, dnorm_inp


  call interest_initialize( .false. ) 

  alpha_inp = 1.0d0; ax_inp = 0.0d0; ay_inp = 0.0d0; az_inp = 0.0d0; anorm_inp = 1d0
  beta_inp  = 1.0d0; bx_inp = 0.0d0; by_inp = 0.0d0; bz_inp = 0.0d0; bnorm_inp = 1d0
  gamma_inp = 1.0d0; cx_inp = 0.0d0; cy_inp = 0.0d0; cz_inp = 1.4d0; cnorm_inp = 1d0
  delta_inp = 1.0d0; dx_inp = 0.0d0; dy_inp = 0.0d0; dz_inp = 1.4d0; dnorm_inp = 1d0

  class = 'llll'
  factor = 0.71270547d0 *  0.71270547d0 *  0.71270547d0 *  0.71270547d0
  fint_inp = 0.0d0

  la_inp = 1
  lb_inp = 1
  lc_inp = 1
  ld_inp = 1

  write(*,*) 'Test of integrals: angular momenta are ' , la_inp, lb_inp, lc_inp, ld_inp
  write(*,*) 'Where the normalization factor is :', factor 

  call interest_eri(class,factor,fint_inp, &
      la_inp,alpha_inp,ax_inp,ay_inp,az_inp,anorm_inp, &
      lb_inp,beta_inp, bx_inp,by_inp,bz_inp,bnorm_inp, &
      lc_inp,gamma_inp,cx_inp,cy_inp,cz_inp,cnorm_inp, &
      ld_inp,delta_inp,dx_inp,dy_inp,dz_inp,dnorm_inp)

  print *, fint_inp(1)


  la_inp = 2
  lb_inp = 1
  lc_inp = 1
  ld_inp = 1

  factor = 0.71270547d0 *  0.71270547d0 *  0.71270547d0 *  1.42541094d0

  write(*,*)
  write(*,*)
  write(*,*)
  write(*,*) 'Test of integrals 2: angular momenta are:' , la_inp, lb_inp, lc_inp, ld_inp
  write(*,*) 'Where the normalization factor is :', factor 

  ! SUBROUTINE interest_eri_4c_new(class,factor,fint_inp,                          &
  !                                la_inp,alpha_inp,ax_inp,ay_inp,az_inp,anorm_inp,&
  !                                lb_inp,beta_inp, bx_inp,by_inp,bz_inp,bnorm_inp,&
  !                                lc_inp,gamma_inp,cx_inp,cy_inp,cz_inp,cnorm_inp,&
  !                                ld_inp,delta_inp,dx_inp,dy_inp,dz_inp,dnorm_inp )




  call interest_eri(class,factor,fint_inp, &
      la_inp,alpha_inp,ax_inp,ay_inp,az_inp,anorm_inp, &
      lb_inp,beta_inp, bx_inp,by_inp,bz_inp,bnorm_inp, &
      lc_inp,gamma_inp,cx_inp,cy_inp,cz_inp,cnorm_inp, &
      ld_inp,delta_inp,dx_inp,dy_inp,dz_inp,dnorm_inp)

  print *, fint_inp(1:3)

end subroutine ssss_psss_test