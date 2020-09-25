# Bineado Equal Number

# Esto devuelve los centros (que estan calculado con la mediana) y los nodos
# de hacer un bieneado equal number del vector x, con nbin numero de bines

import numpy as np

def rbin1(x, nbin):
    
    x_sort = np.sort(x) # ordeno el vector x de menor a mayor
    n = len(x)          # longitud del vector x = cant de particulas
    
    delta = np.int(n / nbin)    # cant de particulas que entran en cada bin
    med = np.zeros(nbin)
    
    nodos    = np.zeros(nbin+1)
    nodos[0] = x_sort[0] # esto indica que el primer nodo va a coincidir
                         # con la posicion de la primera particula
    
    for i in range(0,nbin):
        med[i]     = np.median(x_sort[i*delta:(i+1)*delta])
        nodos[i+1] = x_sort[i*delta:(i+1)*delta][-1]
    
    return med, nodos

#################################################################################
# Bineado equal number - cantidad de particulas por bin
#################################################################################

# Esto devuelve los centros (que estan calculado con la mediana) y los nodos
# de hacer un bieneado equal number del vector x, con npar numero de particulas 
# por bines

def rbin2(x, npar):
    
    x_sort = np.sort(x) # ordeno el vector x de menor a mayor
    n = len(x)          # longitud del vector x = cant de particulas
    
    nbin = np.int(n / npar)    # cant de particulas que entran en cada bin
   
    
    med = np.zeros(nbin)
    
    nodos    = np.zeros(nbin+1)
    nodos[0] = x_sort[0] # esto indica que el primer nodo va a coincidir
                         # con la posicion de la primera particula
    
    for i in range(0,nbin):
        med[i]     = np.median(x_sort[i*npar:(i+1)*npar])
        nodos[i+1] = x_sort[i*npar:(i+1)*npar][-1]
    
    return med, nodos