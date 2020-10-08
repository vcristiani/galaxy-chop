import numpy as np
from aling import *
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.mixture import GaussianMixture
import random

####### En caso de que la galaxia TENGA particulas de GAS #############################################################
try:
    ## Leemos los archivos de la galaxia
    ## columna 0 = masa (estan en unidades de 1e10 M_sun)
    ## columna 1-3 = posiciones (en kpc)
    ## columna 4-6 = velocidades (en km/s)
    gas_ = np.load(path+'TNG_galaxias/gas_/TNG100-1_gal_ID_'+str(ID)+'_particle_type_0.npy')
    dark = np.load(path+'TNG_galaxias/dark/TNG100-1_gal_ID_'+str(ID)+'_particle_type_1.npy')
    star = np.load(path+'TNG_galaxias/star/TNG100-1_gal_ID_'+str(ID)+'_particle_type_4.npy')

    ## Rotamos las posiciones y velocidades de las estrelals para alinear con la dirección de J
    ## Tambien obtenemos la matriz de rotacion, y rotamos dm y gas
    pos_star_rot,vel_star_rot,A = aling(star[:,0],star[:,1:4],star[:,4:7],3.*R[j])

    pos_dark_rot = rot(dark[:,1:4],A)
    vel_dark_rot = rot(dark[:,4:7],A)

    pos_gas_rot = rot(gas_[:,1:4],A)
    vel_gas_rot = rot(gas_[:,4:7],A)

    ## Calculamos las componentes de momento angular de estrellas, gas y DM
    L_dark  = np.asarray((pos_dark_rot[:,1]*vel_dark_rot[:,2] - pos_dark_rot[:,2]*vel_dark_rot[:,1], 
                          pos_dark_rot[:,2]*vel_dark_rot[:,0] - pos_dark_rot[:,0]*vel_dark_rot[:,2], 
                          pos_dark_rot[:,0]*vel_dark_rot[:,1] - pos_dark_rot[:,1]*vel_dark_rot[:,0]))
    L_star = np.asarray((pos_star_rot[:,1]*vel_star_rot[:,2] - pos_star_rot[:,2]*vel_star_rot[:,1], 
                         pos_star_rot[:,2]*vel_star_rot[:,0] - pos_star_rot[:,0]*vel_star_rot[:,2], 
                         pos_star_rot[:,0]*vel_star_rot[:,1] - pos_star_rot[:,1]*vel_star_rot[:,0]))
    L_gas  = np.asarray((pos_gas_rot[:,1]*vel_gas_rot[:,2] - pos_gas_rot[:,2]*vel_gas_rot[:,1], 
                         pos_gas_rot[:,2]*vel_gas_rot[:,0] - pos_gas_rot[:,0]*vel_gas_rot[:,2], 
                         pos_gas_rot[:,0]*vel_gas_rot[:,1] - pos_gas_rot[:,1]*vel_gas_rot[:,0]))

    L_part = np.concatenate((L_gas,L_dark,L_star),axis=1)

    Lr_star = np.sqrt(L_star[0,:]**2 + L_star[1,:]**2) # Componente en el plano del momento angular de estrellas
    Lr = np.sqrt(L_part[0,:]**2 + L_part[1,:]**2)      # Componente en el plano del momento angular de las particulas

    ## Calculamos la energia cinetica de las partículas
    k_star = 0.5*(star[:,4]**2 + star[:,5]**2 + star[:,6]**2)
    k_dark = 0.5*(dark[:,4]**2 + dark[:,5]**2 + dark[:,6]**2)
    k_gas  = 0.5*(gas_[:,4]**2 + gas_[:,5]**2 + gas_[:,6]**2)

    ## Leemos los potenciales que guardamos en el archivo (notar q los files de potencial tienen dos columnas: ID y pot)
    path_potencial = '/home/vcristiani/doctorado/TNG_potenciales/potencial_'

    pot_star = np.loadtxt(path_potencial+'star_ID_'+str(ID[j])+'.dat')
    pot_dark = np.loadtxt(path_potencial+'dark_ID_'+str(ID[j])+'.dat')
    pot_gas = np.loadtxt(path_potencial+'gas_ID_'+str(ID[j])+'.dat')

    ## Calculamos la energia
    E_tot_star = k_star - pot_star[:,1]
    E_tot_dark = k_dark - pot_dark[:,1]
    E_tot_gas  = k_gas  - pot_gas[:,1]

    E_tot = np.concatenate((E_tot_gas,E_tot_dark,E_tot_star))

####### En caso de que la galaxia NO TENGA particulas de GAS ##########################################################
except:
    ## Leemos los archivos de la galaxia
    dark = np.load(path+'TNG_galaxias/dark/TNG100-1_gal_ID_'+str(ID[j])+'_particle_type_1.npy')
    star = np.load(path+'TNG_galaxias/star/TNG100-1_gal_ID_'+str(ID[j])+'_particle_type_4.npy')

    ## Rotamos las posiciones y velocidades.
    pos_star_rot,vel_star_rot,A = aling(star[:,0],star[:,1:4],star[:,4:7],3.*R[j]) # lo estabamos haciendo con 10.

    pos_dark_rot = rot(dark[:,1:4],A)
    vel_dark_rot = rot(dark[:,4:7],A)

    ## Calculamos las componentes de momento angular.
    L_dark  = np.asarray((pos_dark_rot[:,1]*vel_dark_rot[:,2] - pos_dark_rot[:,2]*vel_dark_rot[:,1], 
                          pos_dark_rot[:,2]*vel_dark_rot[:,0] - pos_dark_rot[:,0]*vel_dark_rot[:,2], 
                          pos_dark_rot[:,0]*vel_dark_rot[:,1] - pos_dark_rot[:,1]*vel_dark_rot[:,0]))
    L_star = np.asarray((pos_star_rot[:,1]*vel_star_rot[:,2] - pos_star_rot[:,2]*vel_star_rot[:,1], 
                         pos_star_rot[:,2]*vel_star_rot[:,0] - pos_star_rot[:,0]*vel_star_rot[:,2], 
                         pos_star_rot[:,0]*vel_star_rot[:,1] - pos_star_rot[:,1]*vel_star_rot[:,0]))

    L_part = np.concatenate((L_dark,L_star),axis=1)

    Lr_star = np.sqrt(L_star[0,:]**2 + L_star[1,:]**2) # Componente en el plano del momento angular de estrellas
    Lr = np.sqrt(L_part[0,:]**2 + L_part[1,:]**2)      # Componente en el plano del momento angular de las particulas

    ## Calculamos la energia cinetica de las partículas
    k_star = 0.5*(star[:,4]**2 + star[:,5]**2 + star[:,6]**2)
    k_dark = 0.5*(dark[:,4]**2 + dark[:,5]**2 + dark[:,6]**2)

    ## Leemos los potenciales que guardamos en el archivo
    path_potencial = '/home/vcristiani/doctorado/TNG_potenciales/potencial_'

    pot_star = np.loadtxt(path_potencial+'star_ID_'+str(ID[j])+'.dat')
    pot_dark = np.loadtxt(path_potencial+'dark_ID_'+str(ID[j])+'.dat')

    ## Calculamos la energia
    E_tot_star = k_star - pot_star[:,1]
    E_tot_dark = k_dark - pot_dark[:,1]

    E_tot = np.concatenate((E_tot_dark,E_tot_star))
##############################################################################################################################
## Acá hacemos un filtrado de las partículas que no vamos a usar en la descomposición dinamica
## Nos sacamos de encima las partículas que no están ligadas: E > 0
neg, = np.where(E_tot <= 0.)
neg_star, = np.where(E_tot_star <= 0.)

## Nos sacamos de encima las partículas con E = -inf
fin, = np.where(E_tot[neg] != -inf)
fin_star, = np.where(E_tot_star[neg_star] != -inf)

## Normalizamos las dos variables: E entre 0 y 1; L entre -1 y 1
E = E_tot[neg][fin]/np.abs(np.min(E_tot[neg][fin]))
L = L_part[2,:][neg][fin]/np.max(np.abs(L_part[2,:][neg][fin]))

## Hacemos el bineado en energía y seleccionamos los valores de Jz con los que calculamos el J_circ
aux0 = np.arange(-1.,-0.1,0.05)
aux1 = np.arange(-0.1,0.,0.005)

aux = np.concatenate((aux0,aux1),axis=0)

x = np.zeros(len(aux)+1)
y = np.zeros(len(aux)+1)

x[0] = -1.
y[0] = np.abs(L[np.argmin(E)])

for i in range(1,len(aux)):
    mask, = np.where((E <= aux[i]) & (E > aux[i-1]))
    s = np.argsort(np.abs(L[mask]))

    # Aca tenemos en cuenta si en los bines de energia hay o no particulas
    if len(s) != 0: 
        if len(s) == 1:
            x[i] = E[mask][s]
            y[i] = np.abs(L[mask][s])
        else:
            if (1.-(np.abs(L[mask][s][-2])/np.abs(L[mask][s][-1]))) >= 0.01 :
                x[i] = E[mask][s][-2]
                y[i] = np.abs(L[mask][s][-2])
            else:
                x[i] = E[mask][s][-1]
                y[i] = np.abs(L[mask][s][-1])
    else: pass

# Mascara para completar el ultimo bin, en caso de que no haya bines vacios
mask, = np.where(E > aux[len(aux)-1])

if len(mask) != 0:
    x[len(aux)] = E[mask][np.abs(L[mask]).argmax()]
    y[len(aux)] = np.abs(L[mask][np.abs(L[mask]).argmax()])

# En el caso en que haya bines vacios, nos deshacemos de ellos
else:
    i = len(np.where(y == 0)[0]) - 1
    if i == 0:
        x = x[:-1]
        y = y[:-1]
    else:
        x = x[:-i]
        y = y[:-i]    

## En caso de que algun bin intermedio no tenga puntos
zero, = np.where(x != 0.)
x = x[zero]
y = y[zero]

## Guardamos los puntos para calcular el J_circ
stack = np.column_stack((x, y))
np.savetxt('/home/vcristiani/doctorado/TNG_envolventes/envolvente_ID_'+str(ID)+'.dat', stack, fmt=['%18.14f','%18.14f'])

## Hacemos la interpolación para calcular el J_circ
spl = InterpolatedUnivariateSpline(x,y,k=1)

## Normalizamos E, Lz y Lr para las estrellas
E_star  = E_tot_star[neg_star][fin_star]/np.abs(np.min(E_tot[neg][fin]))
L_star_ = L_star[2,:][neg_star][fin_star]/np.max(np.abs(L_part[2,:][neg][fin]))
Lr_star_ = Lr_star[neg_star][fin_star]/np.max(np.abs(Lr[neg][fin]))

## Calculamos el parametro de circularidad Lz/Lc
eps = L_star_/spl(E_star)

## Calculamos lo mismo para Lp/Lc
eps_r = Lr_star_/spl(E_star)

############ Ojo con esto, hay que ver que las particulas que estamos sacando no sean significativas ############
## Nos sacamos de encima las partículas que tengan circularidad < -1 y circularidad > 1
mask, = np.where((eps <= 1.)&(eps >= -1.))

## ID de las particulas estelares limpias
ID_star = np.arange(0,len(star))[neg_star][fin_star][mask]

## Guardamos ID, E y circularidad de las partículas estelares que nos interesan
indice = np.arange(0,len(ID_star))
stack = np.column_stack((ID_star, E_star[mask], eps[mask], eps_r[mask],indice))
np.savetxt('/home/vcristiani/doctorado/TNG_circularidad_y_energia/circularidad_y_L_ID_'+str(ID)+'.dat', stack, 
           fmt=['%10.0f','%18.14f','%18.14f','%18.14f','%10.0f'])


###################################################################################################
# Método de descomposicion dinamica de Abadi + Energia
###################################################################################################

## Leemos ID, E, circularidad, circularidad en el plano de las estrellas de la galaxia e i.
data = np.loadtxt('/home/vcristiani/doctorado/TNG_circularidad_y_energia/circularidad_y_L_ID_'+str(ID)+'.dat')

## Construimos el histograma del parametro de circularidad
n_bin = 100
h = np.histogram(data[:,2],n_bin,range=(-1.,1.))[0]
edges = np.round(np.histogram(data[:,2],n_bin,range=(-1.,1.))[1],2)
a_bin = edges[1]-edges[0]
center = np.histogram(data[:,2],n_bin,range=(-1.,1.))[1][:-1] + a_bin/2.
cero, = np.where(edges == 0.)
m = cero[0]

# Creamos un diccionario: n={} donde vamos a guardar los ID de las particulas que cumplan
# con las restricciones que le ponemos a la máscara. Así luego podemos tener control
# sobre cuales son las partículas que se seleccionan.
n = {}

for i in range(0,n_bin-1):
    mask,= np.where((data[:,2] >= edges[i]) & (data[:,2] < edges[i+1]))
    n['bin'+'%s'%i] = data[:,4][mask]

mask, = np.where((data[:,2] >= edges[n_bin-1]) & (data[:,2] <= edges[n_bin]))
n['bin'+'%s'%(len(center)-1)] = data[:,4][mask]

## Seleccionamos las particulas que pertenecen al esferoide en función del parámetro del circularidad y E
np.random.seed(10)
halo3 = {}

for i in range(0,m):
    halo3['bin'+'%s'%i] = n['bin'+'%s'%i]
#     print i, len(halo3['bin'+'%s'%(i)])

if len(h) >= 2*m:
    lim_aux = 0
else:
    lim_aux = 2*m - len(h)

for i in range(lim_aux,m):

    if len(n['bin'+'%s'%i]) >= len(n['bin'+'%s'%(2*m-1-i)]):
        halo3['bin'+'%s'%(2*m-1-i)] = n['bin'+'%s'%(2*m-1-i)]
#         print i
    else:
#         print i
        nbin_E = 20

        h0, b0 = np.histogram(data[np.int_(n['bin'+'%s'%i]),1], bins=nbin_E, range=(-1.,0.))
        h1, b1 = np.histogram(data[np.int_(n['bin'+'%s'%(2*m-1-i)]),1], bins=nbin_E, range=(-1.,0.))

        aux0 = []

        for j in range(0,nbin_E):
            if h0[j] != 0 :
                if (h0[j] >= h1[j]):
                    ll, = np.where((data[np.int_(n['bin'+'%s'%(2*m-1-i)]),1] >= b1[j]) & (data[np.int_(n['bin'+'%s'%(2*m-1-i)]),1] < b1[j+1]))
                    aux1 = data[np.int_(n['bin'+'%s'%(2*m-1-i)]),4][ll]
                    aux0 = np.concatenate((aux0,aux1),axis=None)

                else:
                    ll, = np.where((data[np.int_(n['bin'+'%s'%(2*m-1-i)]),1] >= b1[j]) & (data[np.int_(n['bin'+'%s'%(2*m-1-i)]),1] < b1[j+1]))
                    aux1 = np.random.choice(data[np.int_(n['bin'+'%s'%(2*m-1-i)]),4][ll],h0[j],replace=False)
                    aux0 = np.concatenate((aux0,aux1),axis=None)
            else:
                aux1 = []
                aux0 = np.concatenate((aux0,aux1),axis=None)

        halo3['bin'+'%s'%(2*m-1-i)] = aux0

## Al resto de las particulas las asignamos al disco
disco3 = n.copy()

for i in range(0,m):
    disco3['bin'+'%s'%i] = [] #----> Dejamos vacios los bines que sólo tienen particulas del HALO

x = set()
y = set()

if len(h)>=2*m:
    lim = m
else:
    lim = len(h)-m

for i in range(lim,len(halo3)):
    x = set(halo3['bin'+'%s'%i])
    y = set(n['bin'+'%s'%i])
    y -= x
    y = np.array(list(y))
    disco3['bin'+'%s'%i] = y

## Guardamos los indices de las particulas que pertenecen al esferoide y al disco, 
## que estan en el archivo cirularidad_y_L_ID_str(ID).dat
esf_ = []
for i in range(len(halo3)):
    esf_ = np.concatenate((esf_,halo3['bin'+'%s'%(i)]))
esf_ = np.int_(esf_)

disk_ = []
for i in range(len(disco3)):
    disk_ = np.concatenate((disk_,disco3['bin'+'%s'%(i)]))
disk_ = np.int_(disk_)


## Para obtener posiciones, masas y velocidades del archivo de la simulación usar:
esferoide = np.int_(data[esf_,0])
disco = np.int_(data[disk_,0])

## Guardamos los indices de las particulas estelares para cada una de las componentes
## Estos inidices sirven para hacer la identificacion de a que componente pertenece cada estrella
np.savetxt(path+'TNG_desc_dinam/esf_ID_'+str(ID)+'.dat', esferoide, fmt=['%10.0f'])
np.savetxt(path+'TNG_desc_dinam/dsk_ID_'+str(ID)+'.dat', disco, fmt=['%10.0f'])

## Guardamos los indices de los parametros de circularidad para cada una de las componentes:
## Esto es para hacer el plot del histograma de circularidades
np.savetxt(path+'TNG_circularidad_componentes/circularidad_esf_ID_'+str(ID)+'.dat', esf_, fmt=['%10.0f'])
np.savetxt(path+'TNG_circularidad_componentes/circularidad_dsk_ID_'+str(ID)+'.dat', disk_, fmt=['%10.0f'])

###################################################################################################
# Método de descomposicion dinamica de Abadi + Energia + Jp/J_circ
###################################################################################################   
## Leemos ID, E, circularidad, circularidad en el plano de las estrellas de la galaxia e i.
data = np.loadtxt('/home/vcristiani/doctorado/TNG_circularidad_y_energia/circularidad_y_L_ID_'+str(ID)+'.dat')

## Construimos el histograma del parametro de circularidad
n_bin = 100
h = np.histogram(data[:,2],n_bin,range=(-1.,1.))[0]
edges = np.round(np.histogram(data[:,2],n_bin,range=(-1.,1.))[1],2)
a_bin = edges[1]-edges[0]
center = np.histogram(data[:,2],n_bin,range=(-1.,1.))[1][:-1] + a_bin/2.
cero, = np.where(edges == 0.)
m = cero[0]

# Creamos un diccionario: n={} donde vamos a guardar los ID de las particulas que cumplan
# con las restricciones que le ponemos a la máscara. Así luego podemos tener control
# sobre cuales son las partículas que se seleccionan.
n = {}

for i in range(0,n_bin-1):
    mask,= np.where((data[:,2] >= edges[i]) & (data[:,2] < edges[i+1]))
    n['bin'+'%s'%i] = data[:,4][mask]

mask, = np.where((data[:,2] >= edges[n_bin-1]) & (data[:,2] <= edges[n_bin]))
n['bin'+'%s'%(len(center)-1)] = data[:,4][mask]

## Seleccionamos las particulas que pertenecen al esferoide en función del parámetro del circularidad y E
np.random.seed(10)
halo3 = {}

for i in range(0,m):
    halo3['bin'+'%s'%i] = n['bin'+'%s'%i]
#     print i, len(halo3['bin'+'%s'%(i)])

if len(h) >= 2*m:
    lim_aux = 0
else:
    lim_aux = 2*m - len(h)

for i in range(lim_aux,m):

    if len(n['bin'+'%s'%i]) >= len(n['bin'+'%s'%(2*m-1-i)]):  # si la cant de particulas en el bin contrarrotante es mayor que en el bin corrotante,
        halo3['bin'+'%s'%(2*m-1-i)] = n['bin'+'%s'%(2*m-1-i)] # entonces todas pertenecen al esferoide
#         print i
    else:
######################################## ESTA ES LA PARTE NUEVA ########################################
######################################## QUE TIENE EN CUENTA LA DISTRIB DE E Y JP/J_CIRC ###############
        nbin_E = 20
        xmin = -1.
        xmax = 0.

        nbin_eps_r = 50
        ymin = 0.
        ymax = 1.

        ## indices de las celdas correspondientes a las particulas de los bines contra-rotantes
        xx_contra = np.int_(((data[np.int_(n['bin'+'%s'%i]),1]-xmin)/(xmax-xmin))*nbin_E)
        yy_contra = np.int_(((data[np.int_(n['bin'+'%s'%i]),3]-ymin)/(ymax-ymin))*nbin_eps_r)

        ## indices de las celdas correspondientes a las particulas de los bines co-rotantes
        xx_co = np.int_(((data[np.int_(n['bin'+'%s'%(2*m-1-i)]),1]-xmin)/(xmax-xmin))*nbin_E)
        yy_co = np.int_(((data[np.int_(n['bin'+'%s'%(2*m-1-i)]),3]-ymin)/(ymax-ymin))*nbin_eps_r)

        aux0 = []
        for indx in range(0,nbin_E):
            mask_indx, = np.where(xx_contra==indx)
            for indy in range(0,nbin_eps_r):
                mask_indy, = np.where(yy_contra[mask_indx]==indy)
        #         print(len(mask_indy),indy)
                if len(mask_indy) == 0: pass ## esto quiere decir que no hay particulas con ese rango de valor de E y eps_r en el bin contrarotante de eps
                else: 
                    #print(len(mask_indy))
                    mask_indx_co, = np.where(xx_co==indx)
                    mask_indy_co, = np.where(yy_co[mask_indx_co]==indy)
                    if len(mask_indy_co) == 0: pass ## esto quiere decir que no hay particulas con ese rango de valor de E y eps_r en el bin corotante de eps

                    elif len(mask_indy_co) <= len(mask_indy): ## el nro particulas con ese rango de valor de E y eps_r en el bin corotante de eps es menor o igual que en el contrarotante, luego se seleccionan todas
                        aux1 = data[np.int_(n['bin'+'%s'%(2*m-1-i)]),4][mask_indx_co][mask_indy_co]
                        aux0 = np.hstack((aux0,aux1))

                    else: ## el nro particulas con ese rango de valor de E y eps_r en el bin corotante de eps es mayor que en el contrarotante, luego se seleccionan algunas random
                        aux1 = np.random.choice(data[np.int_(n['bin'+'%s'%(2*m-1-i)]),4][mask_indx_co][mask_indy_co],len(mask_indy),replace=False)
                        aux0 = np.hstack((aux0,aux1))

##################################################################################################################    

        halo3['bin'+'%s'%(2*m-1-i)] = aux0

## Al resto de las particulas las asignamos al disco
disco3 = n.copy()

for i in range(0,m):
    disco3['bin'+'%s'%i] = [] #----> Dejamos vacios los bines que sólo tienen particulas del HALO

x = set()
y = set()

if len(h)>=2*m:
    lim = m
else:
    lim = len(h)-m

for i in range(lim,len(halo3)):
    x = set(halo3['bin'+'%s'%i])
    y = set(n['bin'+'%s'%i])
    y -= x
    y = np.array(list(y))
    disco3['bin'+'%s'%i] = y

## Guardamos los indices de las particulas que pertenecen al esferoide y al disco, 
## que estan en el archivo cirularidad_y_L_ID_str(ID).dat
esf_ = []
for i in range(len(halo3)):
    esf_ = np.concatenate((esf_,halo3['bin'+'%s'%(i)]))
esf_ = np.int_(esf_)

disk_ = []
for i in range(len(disco3)):
    disk_ = np.concatenate((disk_,disco3['bin'+'%s'%(i)]))
disk_ = np.int_(disk_)

## Para obtener posiciones, masas y velocidades del archivo de la simulación usar:
esferoide = np.int_(data[esf_,0])
disco = np.int_(data[disk_,0])
## estos indices corresponden al archivo de la simulación original

## Guardamos los indices de las particulas estelares para cada una de las componentes
np.savetxt(path+'TNG_desc_dinam/esf_ID_'+str(ID)+'_con_jp.dat', esferoide, fmt=['%10.0f'])
np.savetxt(path+'TNG_desc_dinam/dsk_ID_'+str(ID)+'_con_jp.dat', disco, fmt=['%10.0f'])

## Guardamos los indices de los parametros de circularidad para cada una de las componentes:
np.savetxt(path+'TNG_circularidad_componentes/circularidad_esf_ID_'+str(ID)+'_con_jp.dat', esf_, fmt=['%10.0f'])
np.savetxt(path+'TNG_circularidad_componentes/circularidad_dsk_ID_'+str(ID)+'_con_jp.dat', disk_, fmt=['%10.0f'])

###################################################################################################
# Método de descomposicion dinamica de Obreja
###################################################################################################  

np.loadtxt('/home/vcristiani/doctorado/TNG_circularidad_y_energia/circularidad_y_L_ID_'+str(ID[ind])+'.dat')
X_train = data[:,1:4]

## Aplicamos el GMM
n_comp = 2 #--> nro de componentes

clf_1 = GaussianMixture(n_components=n_comp, n_init=10)
clf_1.fit(X_train)

## Clase asignada
comp_1 = clf_1.predict(X_train)
part0, = np.where(comp_1 == 0)
part1, = np.where(comp_1 == 1)

## Centros de las clases
cent_1 = clf_1.means_

## Probabilidad de pertenecer a cada clase
pro_1 = clf_1.predict_proba(X_train)

if cent_1[0,1] > cent_1[1,1]:
    dsk = pro_1[:,0]
    esf = pro_1[:,1]
else:
    dsk = pro_1[:,1]
    esf = pro_1[:,0]

## Guardamos los pesos de la descomposicion dinamica
np.savetxt(path+'desc_dina/gmm2_esf_ID'+str(ID)+'.dat', esf, fmt=['%12.8f'])
np.savetxt(path+'desc_dina/gmm2_dsk_ID'+str(ID)+'.dat', dsk, fmt=['%12.8f'])


###################################################################################################
# Método de descomposicion dinamica de Obreja
################################################################################################### 

numero_de_componentes_du = np.zeros(2)
numero_de_componentes_du[0] = ID

random.seed(2**32 - 1)

comp = np.arange(2,16)
BIC_med = np.zeros(len(comp))

data = np.loadtxt('/home/vcristiani/doctorado/TNG_circularidad_y_energia/circularidad_y_L_ID_'+str(ID)+'.dat')
X_train = data[:,1:4]

## Acá vamos de decidir cuantas componentes vamos a usar con el criterio de Du et al.
for i in range(0,len(comp)):
    ## Aplicamos el GMM
    n_comp = comp[i]  #--> nro de componentes

    clf_1 = GaussianMixture(n_components=n_comp, n_init=1)
    clf_1.fit(X_train)
    BIC_med[i] = clf_1.bic(X_train)/len(X_train)

BIC_min = np.sum(BIC_med[-5:])/5.
delta_BIC = BIC_med - BIC_min

C_BIC = 0.1
mask, = np.where(delta_BIC <= C_BIC)
n_comp = np.min(comp[mask]) #--> nro de componentes
numero_de_componentes_du [1] = n_comp

clf_1 = GaussianMixture(n_components=n_comp, n_init=10)
clf_1.fit(X_train)

## Centros de las clases
cent_1 = clf_1.means_

## Probabilidad de pertenecer a cada clase
pro_1 = clf_1.predict_proba(X_train)

## Repartimos las probabilidades de los diferentes clusters en las componentes dinámicas
## Sub-componentes: esferoide y disco
esf = np.zeros(len(X_train))
dsk = np.zeros(len(X_train))

for i in range(0,n_comp):
    if cent_1[i,1] >= 0.5:
        dsk = dsk + pro_1[:,i]
    if cent_1[i,1] < 0.5:
        esf = esf + pro_1[:,i]

## Guardamos los pesos de la descomposicion dinamica
np.savetxt(path+'TNG_desc_dinam_du/du_esf_ID_'+str(ID)+'.dat', esf, fmt=['%12.8f'])
np.savetxt(path+'TNG_desc_dinam_du/du_dsk_ID_'+str(ID)+'.dat', dsk, fmt=['%12.8f'])

np.savetxt(path+'TNG_desc_dinam_du/nro_de_componentes.dat', numero_de_componentes_du, fmt=['%18.f','%12.f'])
