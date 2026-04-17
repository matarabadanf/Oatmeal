subroutine interest_prim_cls_interface( &
                factor, fint_inp, &
                la_inp, alpha_inp, ax_inp, ay_inp, az_inp, anorm_inp, &
                lb_inp, beta_inp, bx_inp, by_inp, bz_inp, bnorm_inp, &
                lc_inp, gamma_inp, cx_inp, cy_inp, cz_inp, cnorm_inp, &
                ld_inp, delta_inp, dx_inp, dy_inp, dz_inp, dnorm_inp) &
    bind(C, name="interest_prim_py")

    use iso_c_binding, only: c_int, c_double
    use module_interest_eri
    implicit none

    character(len=4) :: class
    real(c_double), value, intent(in) :: factor
    real(c_double), intent(inout) :: fint_inp(1000)

    integer(c_int), value, intent(in) :: la_inp, lb_inp, lc_inp, ld_inp
    real(c_double), value, intent(in) :: alpha_inp, ax_inp, ay_inp, az_inp, anorm_inp
    real(c_double), value, intent(in) :: beta_inp,  bx_inp, by_inp, bz_inp, bnorm_inp
    real(c_double), value, intent(in) :: gamma_inp, cx_inp, cy_inp, cz_inp, cnorm_inp
    real(c_double), value, intent(in) :: delta_inp, dx_inp, dy_inp, dz_inp, dnorm_inp


    call interest_initialize( .false. ) 

    ! alpha_inp = 1.0d0; ax_inp = 0.0d0; ay_inp = 0.0d0; az_inp = 0.0d0; anorm_inp = 1d0
    ! beta_inp  = 1.0d0; bx_inp = 0.0d0; by_inp = 0.0d0; bz_inp = 0.0d0; bnorm_inp = 1d0
    ! gamma_inp = 1.0d0; cx_inp = 0.0d0; cy_inp = 0.0d0; cz_inp = 1.4d0; cnorm_inp = 1d0
    ! delta_inp = 1.0d0; dx_inp = 0.0d0; dy_inp = 0.0d0; dz_inp = 1.4d0; dnorm_inp = 1d0

    class = 'llll'
    ! factor = 0.71270547d0 *  0.71270547d0 *  0.71270547d0 *  0.71270547d0
    fint_inp = 0.0d0

    ! la_inp = 1
    ! lb_inp = 1
    ! lc_inp = 1
    ! ld_inp = 1

    ! write(*,*) 'Test of integrals: angular momenta are ' , la_inp, lb_inp, lc_inp, ld_inp
    ! write(*,*) 'Where the normalization factor is :', factor 

    call interest_eri(class,factor,fint_inp, &
        la_inp,alpha_inp,ax_inp,ay_inp,az_inp,anorm_inp, &
        lb_inp,beta_inp, bx_inp,by_inp,bz_inp,bnorm_inp, &
        lc_inp,gamma_inp,cx_inp,cy_inp,cz_inp,cnorm_inp, &
        ld_inp,delta_inp,dx_inp,dy_inp,dz_inp,dnorm_inp)

    ! print *, fint_inp(1:81)

end subroutine interest_prim_cls_interface


program main 

  write(*,*) 'The interface has been called'

end program main 