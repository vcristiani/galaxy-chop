
! ============================================================================
! POTENTIAL ENERGY
! ============================================================================

module potential

use iso_fortran_env
!$ use omp_lib

implicit none
private
public fortran_potential

contains
   subroutine fortran_potential(x, y, z, m, soft, pe, n)
        ! Compute the potential energy for each particle.
        !
        ! Parameters
        ! ----------
        ! x, y, z :     (array)    Positions of particles.
        ! m :           (array)    Masses of particles.
        ! softening :   (float)    Softening parameter.
        !
        ! Returns
        ! -------
        ! pe :          (float)    Specific potential energy of particles.
        integer              :: n
        real, intent(in)     :: x(n), y(n), z(n), m(n), soft
        real, intent(out)    :: pe(n)
        real                 :: dist, soft2
        integer              :: i, j

        soft2 = soft ** 2

        !$OMP PARALLEL DEFAULT(NONE) &
        !$OMP SHARED (x, y, z, m, soft2, pe, n) &
        !$OMP PRIVATE(i, j, dist)
        !$OMP DO SCHEDULE(DYNAMIC)
        do i = 1, n
            pe(i) = 0.
            do j = 1, n
                if (i /= j) then

                    dist = sqrt( &
                            (x(i) - x(j)) ** 2 + &
                            (y(i) - y(j)) ** 2 + &
                            (z(i) - z(j)) ** 2 + &
                            soft2 &
                            )
                    pe(i) = pe(i) + m(j) / dist

                end if
            end do
        end do
        !$OMP END DO
        !$OMP END PARALLEL
    end subroutine fortran_potential

end module potential
