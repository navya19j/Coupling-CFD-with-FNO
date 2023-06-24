import numpy
import numpy as np
import matplotlib
#matplotlib.use("pgf")
from shapes import *

def generate_Npoint_shape_plot(N=10, res=100):
    coeffs = [] 
    coords = []
    coeffs.append(np.zeros((2,2*1+1)))
    coeffs.append(np.zeros((2,2*2+1)))
    coeffs.append(np.zeros((2,2*3+1)))
    coords.append(np.zeros((2,res)))
    coords.append(np.zeros((2,res)))
    coords.append(np.zeros((2,res)))
    np.random.seed(4)
    bad_shape =True
    while bad_shape == True:
        pos_r = np.random.uniform(0,0.5,(N))
        for M_  in range(3):
            M = M_ +1 
            #pos_thet = np.random.uniform(0,2*np.pi,(1,N))
            pos_thet =np.linspace(0,2*np.pi,num=N,endpoint=False)
            posx = pos_r*np.cos(pos_thet)
            posy = pos_r*np.sin(pos_thet)
            pos = np.row_stack((posx,posy))
            center = np.mean(pos,axis=1)
            r = pos - center[:,np.newaxis]
            r_mag  = np.sqrt(r[0,:]**2 + r[1,:]**2)
            x = np.zeros((2,np.shape(r)[1]))
            x[0,:] = 1
            costh =np.diag(np.matmul(r.T,x))#r.x
            costh = costh/r_mag
            theta = np.arccos(costh)
            ry = r[1,:]
            rx = r[0,:]
            neg = np.where(ry<0)
            theta[neg] = 2*np.pi - theta[neg]
            #print(r)
            #print(theta)
            rx = rx[np.argsort(theta)]
            ry = ry[np.argsort(theta)]
            theta = theta[np.argsort(theta)]
            #print(rx,ry)
            #print(theta)
            b = np.append(rx,ry)
            #print(np.shape(b))
            #M = 4
            m =  2*M+1
            A = np.zeros((N,m))
            A[:,0] = 1.0
            for j in range(1,M+1):
                A[:,2*j-1] = np.cos(j*theta)
                A[:,2*j]   = np.sin(j*theta)
            # Use the same A for both x and y coordinates.

            AA = np.matmul(A.T,A)
            #print("solving")
            #print(np.shape(AA))
            #print(np.shape(rx))
            coeffs_x = np.linalg.solve(AA,np.matmul(A.T,rx))
            coeffs_y = np.linalg.solve(AA,np.matmul(A.T,ry))
            coeffs[M_] = np.row_stack((coeffs_x,coeffs_y))
            #oeffs = scale_area(coeffs)
                #coeffs[:,2*mi-1,np.newaxis]*np.cos( mi*tt) + coeffs[:,2*mi,np.newaxis]*np.sin(mi*tt)
            #np.cos(M*theta)
            t = np.linspace(0, 2.0*np.pi, num=res, endpoint=True)
            dt = t[1]-t[0]
            coords[M_] = fourier2Cart(coeffs[M_],t)
            coords_prime = np.gradient(coords[M_],dt,axis=1)
            integrand = coords_prime[1,:] * coords[M_][0,:]
            area = np.trapz(integrand, x=t)
            self_intersection = check_self_intersection(coords[M_]) 
            scale = np.sqrt(0.5 / np.absolute(area))
            coeffs[M_] = scale * coeffs[M_]
            coords[M_] = fourier2Cart(coeffs[M_],t)
            domain_intersection = check_domain_intersection(coords[M_])
            #bad_shape = False
            bad_shape = self_intersection or domain_intersection

    shape={"coeffs":coeffs,
           "coords":coords,
           "pos_r":pos_r}
    
    return shape

ITER = 10000
i = 0
thisfolder = "gts_files"

while (i < ITER):

    i+=1

    with open(f'shape_{i}.gts', 'w') as fp:
        pass

    shape = generate_Npoint_shape()

    name = write_shape(shape)

    f=open(f"{thisfolder}/shape_{i}.gts","w")
    call(["shapes",f"shapes/coords/{name}"],stdout=f) 