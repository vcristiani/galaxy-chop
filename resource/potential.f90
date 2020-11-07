program potencial
use iso_fortran_env
use OMP_LIB

implicit none

integer, parameter   :: n = 5000000
real, dimension(:), allocatable   :: ID, x, y, z, m, ep
real, parameter      :: G = 4.299e+4
real                 :: dist
integer              :: i, j


allocate(ID(n), x(n), y(n), z(n), m(n), ep(n))

!-------------leo las posiciones de las particulas---------------------
open(20, file='/home/vcristiani/potencial/agama_M5.dat', status='old')
do i=1, n
        read(20,*) ID(i), x(i), y(i), z(i), m(i)
end do
 close(20)

!======================================================================
!Calcula la energia potencial específica de cada una de las particulas


!$OMP PARALLEL DEFAULT(NONE) &
!$OMP SHARED (x,y,z,m,ep) &
!$OMP PRIVATE(i,j,dist)
!$OMP DO SCHEDULE(DYNAMIC)
do i = 1, n
    ep(i) = 0.
    do j = 1, n
        if (i /= j) then

            dist = sqrt((x(i)-x(j))**2 + (y(i)-y(j))**2 + (z(i)-z(j))**2)
            ep(i) = ep(i) + m(j)/dist

        end if
    end do
    ep(i) = ep(i)*G
end do
!$OMP END DO
!$OMP END PARALLEL

!======================================================================

!-------------escribo los potenciales de las particulas---------------------
open(30, file='/home/vcristiani/potencial/potencial_agama_M5.dat', status='new')
do i = 1,n
        write(30,*) ep(i)
end do
 close(30)

deallocate(ID, x, y, z, m, ep)

end program
