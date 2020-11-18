program potencial
use iso_fortran_env
use OMP_LIB

implicit none

real(8), dimension(:), allocatable   :: ID, x, y, z, m, ep
real(8), parameter                   :: G = 4.299e+4
real(8)                              :: dist
integer                              :: i, j, n_gas, n_dark, n_star, n, k
character (len=2000)                  :: nro_de_ID, t1, t2, t3, t4, dat, c1
character (len=2000)                  :: dark, star, gas

t1 = '/home/vcristiani/doctorado/'
t2 = 'TNG_nro_particulas/nro_particulas_ID_'
t3 = 'TNG_galaxias_dat/'
t4 = 'TNG_potenciales/potencial_'
dat = '.dat'
dark = 'dark_ID_'
star = 'star_ID_'
gas = 'gas_ID_'

open(90, file='/home/vcristiani/doctorado/TNG_galaxias/potencial_faltante_1e10-105.dat', status='old')
open(99, file='/home/vcristiani/doctorado/control.dat', status='new')

do k=1,1644
        read(90,*) nro_de_ID

        !------------- leo la cantidad de partículas de cada tipo ---------------------

        write(c1,'(A,A)') trim(t1), trim(t2)
        write(c1,'(A,A)') trim(c1), trim(nro_de_ID)
        write(c1,'(A,A)') trim(c1), trim(dat)

        open(20, file=trim(c1), status='old')
        read(20,*) n_gas, n_dark, n_star
         close(20)

        n = n_gas + n_dark + n_star

        allocate(ID(n), x(n), y(n), z(n), m(n), ep(n))

        !-------------leo las posiciones de las particulas---------------------
        if (n_gas /= 0) then

                write(c1,'(A,A)') trim(t1), trim(t3)
                write(c1,'(A,A)') trim(c1), trim(gas)
                write(c1,'(A,A)') trim(c1), trim(nro_de_ID)
                write(c1,'(A,A)') trim(c1), trim(dat)

                open(30, file=trim(c1), status='old')
                do i=1, n_gas
                        read(30,*) ID(i), x(i), y(i), z(i), m(i)
                end do
                close(30)
        end if

        write(c1,'(A,A)') trim(t1), trim(t3)
        write(c1,'(A,A)') trim(c1), trim(dark)
        write(c1,'(A,A)') trim(c1), trim(nro_de_ID)
        write(c1,'(A,A)') trim(c1), trim(dat)

        open(40, file=trim(c1), status='old')
        do i=1, n_dark
                read(40,*) ID(i+n_gas), x(i+n_gas), y(i+n_gas), z(i+n_gas), m(i+n_gas)
        end do
         close(40)

        write(c1,'(A,A)') trim(t1), trim(t3)
        write(c1,'(A,A)') trim(c1), trim(star)
        write(c1,'(A,A)') trim(c1), trim(nro_de_ID)
        write(c1,'(A,A)') trim(c1), trim(dat)

        open(50, file=trim(c1), status='old')
        do i=1, n_star
                read(50,*) ID(i+n_gas+n_dark), x(i+n_gas+n_dark), y(i+n_gas+n_dark), z(i+n_gas+n_dark), m(i+n_gas+n_dark)
        end do
         close(50)

        !======================================================================
        !Calcula la energia potencial específica de cada una de las particulas


!$OMP PARALLEL DEFAULT(NONE) &
!$OMP SHARED (x,y,z,m,ep,n) &
!$OMP PRIVATE(i,j,dist)
!$OMP DO SCHEDULE(DYNAMIC)
        do i = 1, n
                ep(i) = 0.
                do j = 1, n
                        if (i /= j) then

                                dist = sqrt((x(i)-x(j))**2 + (y(i)-y(j))**2 +(z(i)-z(j))**2)
                                ep(i) = ep(i) + m(j)/dist

                        end if
                end do
                ep(i) = ep(i)*G
        end do
!$OMP END DO
!$OMP END PARALLEL


        !======================================================================

        !-------------escribo los potenciales de las particulas---------------------

        if (n_gas /= 0) then

                write(c1,'(A,A)') trim(t1), trim(t4)
                write(c1,'(A,A)') trim(c1), trim(gas)
                write(c1,'(A,A)') trim(c1), trim(nro_de_ID)
                write(c1,'(A,A)') trim(c1), trim(dat)

                open(60, file=trim(c1), status='new')
                do i = 1,n_gas
                        write(60,*) ID(i),ep(i)
                end do
                 close(60)
        end if

        j = n_gas+1
        n = n_gas + n_dark

        write(c1,'(A,A)') trim(t1), trim(t4)
        write(c1,'(A,A)') trim(c1), trim(dark)
        write(c1,'(A,A)') trim(c1), trim(nro_de_ID)
        write(c1,'(A,A)') trim(c1), trim(dat)

        open(70, file=trim(c1), status='new')
        do i = j,n
                write(70,*) ID(i),ep(i)
        end do
         close(70)

        j = n + 1
        n = n + n_star

        write(c1,'(A,A)') trim(t1), trim(t4)
        write(c1,'(A,A)') trim(c1), trim(star)
        write(c1,'(A,A)') trim(c1), trim(nro_de_ID)
        write(c1,'(A,A)') trim(c1), trim(dat)

        open(80, file=trim(c1), status='new')
        do i = j,n
                write(80,*) ID(i),ep(i)
        end do
         close(80)

        deallocate(ID, x, y, z, m, ep)

        write(99,*) k, nro_de_ID
end do
 close(90)
 close(99)

end program
