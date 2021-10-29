
! ============================================================================
! POTENTIAL ENERGY
! ============================================================================

Module potential  !
    Implicit None

Contains

    Subroutine fortran_potential(x, y, z, m, soft, ep, n)
        ! POTENTIAL ENERGY
        ! ----------------
        ! Compute the potential energy for each particle
        !
        ! input:
        ! - x, y, z    (array)    : Positions of particles
        ! - m          (array)    : Masses of particles
        ! - soft       (array)    : Softening parameter
        ! output:
        ! - ep:        (array)    : Potetntial energy
        !
        use iso_fortran_env
        use OMP_LIB

        integer              :: n
        real, intent(in)     :: x(n), y(n), z(n), m(n), soft(n)
        real, intent(out)    :: ep(n)
        real                 :: dist
        integer              :: i, j

        ! ====================================================================
        ! Calcula la energia potencial especifica de cada particula

        !$OMP PARALLEL DEFAULT(NONE) &
        !$OMP SHARED (x,y,z,m,soft,ep) &
        !$OMP PRIVATE(i,j,dist)
        !$OMP DO SCHEDULE(DYNAMIC)
        do i = 1, n
            ep(i) = 0.
            do j = 1, n
                if (i /= j) then

                    dist = sqrt(
                        (x(i) - x(j)) ** 2 +
                        (y(i) - y(j)) ** 2 +
                        (z(i) - z(j)) ** 2 +
                        soft(i) ** 2
                    )
                    ep(i) = ep(i) + m(j) / dist

                end if
            end do
        end do
        !$OMP END DO
        !$OMP END PARALLEL
    End Subroutine fortran_potential

End Module potential
