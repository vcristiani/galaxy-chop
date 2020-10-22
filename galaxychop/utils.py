import numpy as np


def _get_rot_matrix(m, pos, vel, r_corte=None):
    """
    Calculates the rotation matrix that aligns the TOTAL
    agular momentum of the particles with the z-axis. The 
    positions, velocities and masses of the particles are 
    used. Optionally, only particles within a cutting 
    radius `(r_corte)` can be used.

    Parameters
    ----------
    m : `np.ndarray`, shape(n,1)
        Masses of particles.
    pos : `np.ndarray`, shape(n,3)
        Positions of particles.
    vel : `np.ndarray`, shape(n,3)
        Velocities of particles.
    r_corte : `float`, optional
        The default is ``None``; if provided, it must be 
        positive and the rotation matrix `A` is calculated 
        from the particles with radii smaller than 
        r_corte.                
   
    Returns
    -------
    A : `np.ndarray`, shape(3,3)
        Rotation matrix.
    """
    
    jx = m*(pos[:, 1]*vel[:, 2] - pos[:, 2]*vel[:, 1])
    jy = m*(pos[:, 2]*vel[:, 0] - pos[:, 0]*vel[:, 2])
    jz = m*(pos[:, 0]*vel[:, 1] - pos[:, 1]*vel[:, 0])

    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    
    if r_corte is not None:
        mask = np.where(r < r_corte)
    else:
        mask = np.repeat(True, len(r)), 

    rjx = np.sum(jx[mask])
    rjy = np.sum(jy[mask])
    rjz = np.sum(jz[mask])

    rjp = np.sqrt(rjx**2 + rjy**2)
    rj = np.sqrt(rjx**2 + rjy**2 + rjz**2)

    e1x = rjy/rjp
    e1y = -rjx/rjp
    e1z = 0.

    e2x = rjx*rjz/(rjp*rj)
    e2y = rjy*rjz/(rjp*rj)
    e2z = -rjp/rj

    e3x = rjx/rj
    e3y = rjy/rj
    e3z = rjz/rj

    A = np.asarray(
        (
            [e1x,e1y,e1z],
            [e2x,e2y,e2z],
            [e3x,e3y,e3z]
            )
    )
    
    return(A)


def aling(m, pos, vel, r_corte):
    """This rotates the positions, speeds and angular 
    moments of the particles so that the total angular 
    moment coincides with the z-axis. 
    
    Optionally, only particles within a cutting radius 
    `(r_corte)` can be used to calculate the rotation 
    matrix.
    
    Parameters
    ----------
    m : `np.ndarray`, shape(n,1)
        Masses of particles.
    pos : `np.ndarray`, shape(n,3)
        Positions of particles.
    vel : `np.ndarray`, shape(n,3)
        Velocities of particles.
    r_corte : `float`, optional
        The default is ``None``; if provided, it must be 
        positive and the rotation matrix `A` is calculated 
        from the particles with radii smaller than r_corte.
   
    Returns
    -------
    pos_rot : `np.ndarray`, shape(n,3)
        Rotated, positions of particles
    vel_rot : `np.ndarray`, shape(n,3)
        Rotated, velocities of particles
    """
    A = _get_rot_matrix(m, pos, vel, r_corte)

    pos_rot = np.dot(A, pos.T)
    vel_rot = np.dot(A, vel.T)
    
    return pos_rot.T, vel_rot.T

###################################################################

def rot(x, A):
    """# Esto rota posiciones y velocidades usando la matriz de rotacion A.
    """
    x_rot = np.dot(A, x.T).T
    return x_rot


def dens_2D(rd, rbin, md):
    """# rd = radio de todas las particulas del disco
    # rbin = radio de los anillos sobre los cuales se calcula la densidad
    # md = masa de las particulas del disco.
    """
    aux = len(rbin)-1

    r = np.zeros(aux)
    sup = np.zeros(aux)
    M = np.zeros(aux)

    for i in range(0, aux):

        sup[i] = np.pi*(rbin[i+1]**2 - rbin[i]**2)
        mask, = np.where((rd > rbin[i]) & (rd <= rbin[i+1]))

        r[i] = rbin[i] + (rbin[i+1]-rbin[i])/2.
        M[i] = np.sum(md[mask])

    rho = M/sup
    return r, rho

# ======================================================================


def dens_3D(rd, rbin, md):
    """rd = radio de todas las particulas
    # rbin = radio de los cascarones sobre los cuales se calcula la densidad
    # md = masa de las particulas.
    """
    aux = len(rbin)-1

    r = np.zeros(aux)
    vol = np.zeros(aux)
    M = np.zeros(aux)

    for i in range(0, aux):

        vol[i] = 4.*np.pi*(rbin[i+1]**3 - rbin[i]**3)/3.
        mask, = np.where((rd > rbin[i]) & (rd <= rbin[i+1]))

        r[i] = rbin[i] + (rbin[i+1]-rbin[i])/2.
        M[i] = np.sum(md[mask])

    rho = M/vol
    return r, rho


def rbin1(x, nbin):
    """# Bineado Equal Number

    # Esto devuelve los centros (que estan calculado con la mediana)
    # y los nodos de hacer un bieneado equal number del vector x,
    # con nbin numero de bines.
    """
    # Ordeno el vector x de menor a mayor.
    x_sort = np.sort(x)
    # Longitud del vector x = cant de particulas.
    n = len(x)

    # Cant de particulas que entran en cada bin.
    delta = np.int(n / nbin)
    med = np.zeros(nbin)

    nodos = np.zeros(nbin+1)
    # Esto indica que el primer nodo va a coincidir
    # con la posicion de la primera particula.
    nodos[0] = x_sort[0]

    for i in range(0, nbin):
        med[i] = np.median(x_sort[i*delta:(i+1)*delta])
        nodos[i+1] = x_sort[i*delta:(i+1)*delta][-1]

    return med, nodos

###############################################################################
# Bineado equal number - cantidad de particulas por bin.
###############################################################################


def rbin2(x, npar):
    """# Esto devuelve los centros (que estan calculado con la mediana)
    # y los nodos de hacer un bieneado equal number del vector x,
    # con npar numero de particulas por bines.
    """
    # Ordeno el vector x de menor a mayor.
    x_sort = np.sort(x)
    # Longitud del vector x = cant de particulas.
    n = len(x)
    # Cant de particulas que entran en cada bin.
    nbin = np.int(n / npar)

    med = np.zeros(nbin)

    nodos = np.zeros(nbin+1)
    # Esto indica que el primer nodo va a coincidir
    # con la posicion de la primera particula.
    nodos[0] = x_sort[0]

    for i in range(0, nbin):
        med[i] = np.median(x_sort[i*npar:(i+1)*npar])
        nodos[i+1] = x_sort[i*npar:(i+1)*npar][-1]

    return med, nodos
