import numpy as np

# Esto devuelve las posiciones, las velocidades y J rotados, de forma que Jz quede 
# alineado con z.

def aling(m,pos,vel,r_corte):
    
    jx = m*(pos[:,1]*vel[:,2] - pos[:,2]*vel[:,1])
    jy = m*(pos[:,2]*vel[:,0] - pos[:,0]*vel[:,2])
    jz = m*(pos[:,0]*vel[:,1] - pos[:,1]*vel[:,0])

    r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    mask, = np.where(r < r_corte)

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

    A = np.asarray(([e1x,e1y,e1z],[e2x,e2y,e2z],[e3x,e3y,e3z]))

    pos_rot = np.dot(A,pos.T)
    vel_rot = np.dot(A,vel.T)
    
    return pos_rot.T, vel_rot.T, A

###################################################################

# Esto rota posiciones y velocidades usando la matriz de rotacion A

def rot(x,A):
    x_rot = np.dot(A,x.T).T
    return x_rot