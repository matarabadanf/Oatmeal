! ------------------------------------------------------------------------------------
!
! Program:      Dirac 
!
! Module:       module_interest_interface.f90 
!
! Description:  - interfacing basic data between Dirac & InteRest programs 
!
! Contains:     
!
! Licensing:    Copyright (c) by the authors of DIRAC.
!
!               This program is free software; you can redistribute it and/or
!               modify it under the terms of the GNU Lesser General Public
!               License version 2.1 as published by the Free Software Foundation.
!
!               This program is distributed in the hope that it will be useful,
!               but WITHOUT ANY WARRANTY; without even the implied warranty of
!               MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!               Lesser General Public License for more details.
!
!               If a copy of the GNU LGPL v2.1 was not distributed with this
!               code, you can obtain one at https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html.
!
! Author:       Michal Repisky (michal.repisky@uit.no)
!
! Logs:         
!
! Comments:     
!                            
! ------------------------------------------------------------------------------------

! define task symbols for CALL DIRAC_PARCTL( task )
#include "dirac_partask.h"

MODULE MODULE_INTEREST_INTERFACE

#ifdef USE_MPI_MOD_F90
  use mpi
#endif
  implicit none

#if defined (VAR_MPI) && !defined(USE_MPI_MOD_F90)
#include "mpif.h"
#endif

  public transfer_gto_info
  public initialize_interest 
  public initialize_interest_herd 

  private

  !matrix dimensions
     !note: by definition nrow, ncol refer to the number of cartesian GTOs
     integer, save :: nshl = 1
     integer, save :: nrow = 0
     integer, save :: ncol = 0
     integer, save :: nr_shells = 0 
     real(8), save :: cvalue
     real(8), save :: screening_threshold = 1.d-20
  !end

  !definition of "constant" data type 
     type type_constant
          real(8) :: alpha       !=(1/c)
          real(8) :: alpha2      !=(1/c)^2 
     end  type
     type(type_constant) :: constant 
  !end

  !definition of "symmetry" data type 
     !todo: later implementation 
     type type_symmetry
          integer :: dummy 
     end  type
     type(type_symmetry) :: symmetry 
  !end

  !definition of "atom" data type 
     type type_atom
          real(8) :: charge 
          real(8) :: coordinate_x
          real(8) :: coordinate_y
          real(8) :: coordinate_z
          real(8) :: gnu_exponent 
     end  type
     type(type_atom), allocatable :: atom(:)
  !end

  !definition of "gto" data type
     type type_gto
          integer :: index
          integer :: offset
          integer :: lvalue
          integer :: origin
          integer :: sdegen 
          integer :: cdegen 
          real(8) :: exponent(1)
          real(8) :: coefficient(1)
     end  type
     type(type_gto), allocatable, target :: gto(:)
  !end

  integer, save :: myrank  = 0
  integer, save :: nr_cpus = 1 

CONTAINS



! ------------------------------------------------------------------------------------
!
  SUBROUTINE initialize_interest()

#include "mxcent.h"
#include "maxorb.h"
#include "maxaqn.h"
#include "nuclei.h"
#include "shells.h"
#include "aovec.h"
#include "primit.h"
#include "dcbdhf.h"
#include "nucdata.h"
#include "dcbgen.h"
#include "cbihr2.h"

    !local variables
       integer :: i
       integer :: j
       integer :: ij
       integer :: ier
       type( type_gto ) :: tmpgto
    !end
  

    !test if the large component basis is uncontracted
       if( nplrg.ne.nlarge )then
         write(6,*) ' Error: contracted basis set found => stop!'
         write(6,*) ' Note: .MDIRAC does not support contracted basis sets'
         stop
       endif
    !end
  
    !interface constants (cval == speed of light used in Dirac [might be different from 137])
       constant = type_constant( (1.0d0/cval), (1.0d0/cval)**2 )
    !end

    !interface molecular data
       allocate( atom(nucind), stat=ier ); if( ier.ne.0 )stop ' Error in allocation: atom(:)'
       do i=1,size(atom)
         atom(i) = type_atom( charge(i), &
                              cord(1,i), &
                              cord(2,i), &
                              cord(3,i), &
                              gnuexp(i)  )
       enddo
    !end
  
    !interface basis set data
    !fixme: contracted basis sets
       allocate( gto(nlrgsh), stat=ier ); if( ier.ne.0 )stop ' Error in allocation: gto(:)'
       i =0
       ij=0
       do while( i < size(gto) )  
         do j=1,nrco(i+1)
           i=i+1
           gto(i) = type_gto( ij,                      &
                              ij,                      &
                              nhkt(i),                 &
                              ncent(i),                & 
                              (2*nhkt(i)-1),           &
                              (nhkt(i)*(nhkt(i)+1))/2, &
                              priexp(i),               &
                              priccf(i,j)              )
           !note: cartesian indexation
           ij=ij+(nhkt(i)*(nhkt(i)+1))/2
         enddo
       enddo
    !end

    !temporary print ... 
       do i=1,size(gto)
         write(6,'(1x,5(a,i3,3x),2(a,d15.5,3x))') ' i = ',i,                             &
                                                  ' index  = ',     gto(i)%index,        &
                                                  ' offset = ',     gto(i)%offset,       &
                                                  ' lvalue = ',     gto(i)%lvalue,       &
                                                  ' origin = ',     gto(i)%origin,       &
                                                  ' exponent = ',   gto(i)%exponent(1),  &
                                                  ' coefficient = ',gto(i)%coefficient(1)
       enddo

       !initialize InteRest integral package 
       call interest_initialize(.true.)

       write(6,*) 
       write(6,'(2x,a,i5  )') 'Total number of basis function shells:    ',size(gto)
       write(6,'(2x,a,i5  )') 'Total number of spherical basis functions:',sum(gto(:)%sdegen)
       write(6,'(2x,a,i5,a)') 'Total number of cartesian basis functions:',sum(gto(:)%cdegen),' (used in InteRest calculations)'
       write(6,*) 
    !end

    !global module variables
    nshl      = 1
    nrow      = sum( gto(:)%cdegen)
    ncol      = sum( gto(:)%cdegen)
    nr_shells = size(gto)
    cvalue    = cval
    if( scrfck > 0.0d0 ) screening_threshold = scrfck


#ifdef VAR_MPI
    call dirac_parctl( INTEREST_INIT_PAR ) 
#endif

    call initialize_interest_herd

  END SUBROUTINE



! ------------------------------------------------------------------------------------
!
  SUBROUTINE initialize_interest_herd()

#ifdef VAR_MPI
    !local MPI related variables
    integer :: i, j, ij, ier, nr_atoms
    integer(kind=mpi_address_kind) :: disp(9)                      !mpi_address_kind is always = 8
    integer                        :: blocklen(9), typex(9)
    integer                        :: newtype
  

    call MPI_comm_rank( MPI_comm_world, myrank,  ier )
    call MPI_comm_size( MPI_comm_world, nr_cpus, ier )

    !initialize InteRest integral package 
    call interest_initialize(.false.)

    !broadcast constants 
       !define the local absolute address of each element within a derived data type
       call mpi_bcast(constant%alpha, 1, mpi_real8,0,mpi_comm_world,ier)
       call mpi_bcast(constant%alpha2,1, mpi_real8,0,mpi_comm_world,ier)
    !end

    !broadcast atom 
       if( allocated(atom) ) nr_atoms = size(atom)
       call mpi_bcast(nr_atoms,1,mpi_integer,0,mpi_comm_world,ier)
       if( .not. allocated(atom) ) allocate( atom(nr_atoms), stat=ier ); if( ier.ne.0 )stop ' Error in allocation: atom(:)'
       !define the local absolute address of each element within a derived data type
       do i=1,nr_atoms
        call MPI_bcast( atom(i)%charge,       1, MPI_real8, 0, MPI_comm_world, ier)
        call MPI_bcast( atom(i)%coordinate_x, 1, MPI_real8, 0, MPI_comm_world, ier)
        call MPI_bcast( atom(i)%coordinate_y, 1, MPI_real8, 0, MPI_comm_world, ier)
        call MPI_bcast( atom(i)%coordinate_z, 1, MPI_real8, 0, MPI_comm_world, ier)
        call MPI_bcast( atom(i)%gnu_exponent, 1, MPI_real8, 0, MPI_comm_world, ier)
       end do
    !end

    !broadcast gto 
       if( allocated(gto) ) nr_shells = size(gto)
       call mpi_bcast(nr_shells,1,mpi_integer,0,mpi_comm_world,ier)
       if( .not. allocated(gto) ) allocate( gto(nr_shells), stat=ier ); if( ier.ne.0 )stop ' Error in allocation: gto(:)'
       do i=1,nr_shells
        call MPI_bcast( gto(i)%index,          1, MPI_integer, 0, MPI_comm_world, ier)
        call MPI_bcast( gto(i)%offset,         1, MPI_integer, 0, MPI_comm_world, ier)
        call MPI_bcast( gto(i)%lvalue,         1, MPI_integer, 0, MPI_comm_world, ier)
        call MPI_bcast( gto(i)%origin,         1, MPI_integer, 0, MPI_comm_world, ier)
        call MPI_bcast( gto(i)%sdegen,         1, MPI_integer, 0, MPI_comm_world, ier)
        call MPI_bcast( gto(i)%cdegen,         1, MPI_integer, 0, MPI_comm_world, ier)
        call MPI_bcast( gto(i)%exponent(1),    1, MPI_real8,   0, MPI_comm_world, ier)
        call MPI_bcast( gto(i)%coefficient(1), 1, MPI_real8,   0, MPI_comm_world, ier)
       end do
    !end

    !final broadcasting (global module variables)
       call mpi_bcast(nshl,               1,mpi_integer,0,mpi_comm_world,ier)
       call mpi_bcast(nrow,               1,mpi_integer,0,mpi_comm_world,ier)
       call mpi_bcast(ncol,               1,mpi_integer,0,mpi_comm_world,ier)
       call mpi_bcast(cvalue,             1,mpi_real8,  0,mpi_comm_world,ier)
       call mpi_bcast(screening_threshold,1,mpi_real8,  0,mpi_comm_world,ier)
    !end
#endif

  END SUBROUTINE



! ------------------------------------------------------------------------------------
  SUBROUTINE transfer_gto_info(cart2lval)

    !input/ouput
    integer, intent(out) :: cart2lval(*) 

    !local
    integer :: i, j

    do i=1,size(gto)
      j = gto(i)%cdegen
      cart2lval((1+gto(i)%offset):(j+gto(i)%offset)) = gto(i)%lvalue
    enddo

  END SUBROUTINE
! ------------------------------------------------------------------------------------
END MODULE
