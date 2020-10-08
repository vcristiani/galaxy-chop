import numpy as np

# rd = radio de todas las particulas del disco
# rbin = radio de los anillos sobre los cuales se calcula la densidad
# md = masa de las particulas del disco

def dens_2D(rd,rbin,md):
    
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

#======================================================================

# rd = radio de todas las particulas 
# rbin = radio de los cascarones sobre los cuales se calcula la densidad
# md = masa de las particulas

def dens_3D(rd,rbin,md):
    
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