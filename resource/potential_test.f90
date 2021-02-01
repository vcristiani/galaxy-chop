program potencial
    use iso_fortran_env

    implicit none

    integer, parameter   :: n = 100
    real*8, dimension(:), allocatable   :: x, y, z, m, ep
    real*8, parameter      :: G = 4.299e-6, eps=0.0
    real*8                 :: dist
    integer                :: i, j


    allocate(x(n), y(n), z(n), m(n), ep(n))

    !-------------leo las posiciones de las particulas---------------------
    open(20, file='../tests/test_data/mock_particles.dat', status='old')
    do i=1, n
            read(20,*) x(i), y(i), z(i), m(i)
    end do
     close(20)

    !======================================================================
    !Calcula la energia potencial específica de cada una de las particulas


    do i = 1, n
            ep(i) = 0.
            do j = 1, n
                    if (i /= j) then

                            dist = sqrt((x(i)-x(j))**2 + (y(i)-y(j))**2 + (z(i)-z(j))**2 + eps**2)
                            ep(i) = ep(i) + m(j)/dist

                    end if
            end do
            ep(i) = ep(i)*G
    end do
    !$OMP END DO
    !$OMP END PARALLEL

    !======================================================================

    !-------------escribo los potenciales de las particulas---------------------
    open(30, file='../tests/test_data/fpotential_test.dat', status='unknown')
    do i = 1,n
            write(30,*) ep(i)
    end do
     close(30)

    deallocate(x, y, z, m, ep)

    end program
