# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:03:15 2018

@author: miguet
"""
import numpy as np
import math
def find_circle(image,hough_radii):
    
    from skimage.feature import peak_local_max
    from skimage.transform import hough_circle
    
    hough_res = hough_circle(image, hough_radii,full_output=True)

    centers = []
    accums = []
    radii = []
    
    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        peaks = peak_local_max(h, num_peaks=2)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius, radius])
        
    #from skimage.draw import circle_perimeter
    # Draw the detected edge (circle, largest accumulator)
    idx = np.argsort(accums)[::-1][0]
    
    center_x, center_y = centers[idx]
    radius = radii[idx]
    
    
    center_x=center_x-hough_radii[-1]
    center_y=center_y-hough_radii[-1]
    tip=[center_y,center_x-radius]
    
    return center_x,center_y,radius,tip
    
def theoretical_contour(image,lc,radius,tip):
    calib=0.00124/400#400 pixel = 1.24mm
    r0=radius*calib
    r_st=r0/lc
    
    Z0=image.shape[0]-tip[1]#image.shape[0]-50##Position de Z0 w/r x (coordonnée verticale)
    Zmax=Z0/lc*calib#maximum possible values of Z to be upgraded
    nPoints=Z0*2
    ds=Zmax/nPoints
    #####Bouadary for validity of sperical approx
    if r_st <1:
        sLimit=r_st*0.2
    else:
        sLimit=.2
        
    s=0
    R=[]
    Z=[]
    while s<sLimit:    
        R.append(r_st*np.sin(s/r_st)+s**5/(40*r_st**2))
        Z.append(1/(2*r_st)*s**(2)*(1-(0.75+1/(12*r_st**2))*s**2))
        s+=ds
#    s-=ds
    #    psi=s/r_st*(1-s**2/8)
        
#    width=image.shape[1]#(ax.get_ylim()[0]- ax.get_ylim()[1])
#    height=image.shape[0]    
    
    ####Fonction dérivation des paramètres:
    def deriv(variables):
        k=[]
        k.append(math.cos(variables[1][-1]))#dérivée 
        k.append(variables[2][-1])
        k.append(-np.sin(variables[1][-1])+np.cos(variables[1][-1])*(np.sin(variables[1][-1])/variables[0][-1]-variables[2][-1])/variables[0][-1])
        k.append(np.sin(variables[1][-1]))
        return k
    #    
    ###initial conditions with approximate solution:
    nVar=4
    variables=[[0] for i in range(nVar)]
    tmp=[[] for i in range(nVar)]
    
    variables[0][0]=r_st*np.sin(s/r_st)+s**5/(40*r_st**2)
    variables[1][0]=s*(1-0.125*s**2)/r_st
    variables[2][0]=(1-0.375*s**2)/r_st
    variables[3][0]=1/(2*r_st)*s**(2)*(1-(0.75+1/(12*r_st**2))*s**2)
    Zlocal=0
    while Zlocal<image.shape[0]-1:
#    while variables[3][-1]<Zmax:
        k1=deriv(variables)
        for i in range(nVar):
            tmp[i]=[variables[i][-1]+0.5*ds*k1[i]]
        k2=deriv(tmp)
        for i in range(nVar):
            tmp[i]=[variables[i][-1]+0.5*ds*k2[i]]
        k3=deriv(tmp)
        for i in range(nVar):
            tmp[i]=[variables[i][-1]+0.5*ds*k3[i]]
        k4=deriv(tmp)
        for i in range(nVar):
            variables[i].append(variables[i][-1]+ds*(k1[i]+2*(k2[i]+k3[i])+k4[i])) 
        Zlocal=lc/calib*variables[3][-1]+tip[1]#+ax.get_ylim()[1]
#        Zlocal=lc/calib*variables[3][-1]+tip[1]-10            
        
    tak=len(R)
            
    R.extend(variables[0][:])
    Z.extend(variables[3][:])
    return R,Z,tak
    
def rotate_lines(R,Z, center, theta):
    from math import sin, cos#, radians
    """ Rotate self.polylines the given angle about their centers. """
#    theta = radians(deg)  # Convert angle from degrees to radians
    theta=-theta*np.pi/180##sombre histoire de convention...
    cosang, sinang = cos(theta), sin(theta)

#    for pl in self.polylines:
#        # Find logical center (avg x and avg y) of entire polyline
#        n = len(pl.lines)*2  # Total number of points in polyline
    cx = center[0]
    cy = center[1]
    R_rot=[]
    Z_rot=[]
    for i in range(len(R)):
       xr=R[i]-cx
       yr=Z[i]-cy
       
       nx=( xr*cosang - yr*sinang) + cx
       ny=( xr*sinang + yr*cosang) + cy
    
#        nx=(R[i]+Z[i]-cx-cy)/(2*cosang)+cx
#        ny=-(R[i]-Z[i]-cx+cy)/(2*sinang)+cy
#        
       R_rot.append(nx)
       Z_rot.append(ny)
    return R_rot,Z_rot
    
def partial_ksi(theorique,python):
    ####calculate the location on the python contour that minimizes the distance to the theoretical one
    dist=[]
#    m=0
#    while not dist:
#        m+=1
#        z_inds=np.where(np.abs(python[1]-theorique[1])<m)[0]
#        for i in z_inds:
#            dist.append(abs(python[0][i]-theorique[0])**2)
#    ind=np.where(dist==min(dist))[0][0]
    m=0
    while not dist:
        m+=1
        z_inds=np.where(np.abs(python[1]-theorique[1])<m)[0]
        for i in z_inds:
            dist.append(((python[0][i]-theorique[0])**2+(python[1][i]-theorique[1])**2)**(1/2))
    ind=np.where(dist==min(dist))[0][0]
    return [python[1][z_inds[ind]],python[0][z_inds[ind]]],min(dist)
  
def error_calculation(R,Z,python):  
    ind_milieu=int(len(Z)/2)    

    ksi_z=[]
    mini_inds=[]
    for i in range (ind_milieu):
        ind_droite=ind_milieu+i
        ind_gauche=ind_milieu-i
        if i==0:
            min_dist_ind,ksi0=partial_ksi([R[ind_milieu],Z[ind_milieu]],python)
            mini_inds.append(min_dist_ind)
            ksi_z.append(ksi0)
        else:
            min_dist_ind,ksi_gauche=partial_ksi([R[ind_gauche],Z[ind_gauche]],python)
            mini_inds.append(min_dist_ind)
            min_dist_ind,ksi_droite=partial_ksi([R[ind_droite],Z[ind_droite]],python)
            mini_inds.append(min_dist_ind)
            ksi_z.append(ksi_gauche+ksi_droite)
        n=i*2+1
        RMSd=np.sqrt(np.sum(ksi_z)/n)
    return np.sum(ksi_z),ksi_z,mini_inds,RMSd
    
def error_calculation_2(R,Z,R_python,Z_python,guess_tipy):  
    from scipy.interpolate import interp1d
    
    R_python_l=R_python[R_python<guess_tipy]
    Z_python_l=Z_python[R_python<guess_tipy]
    py_interp_l=interp1d(Z_python_l,R_python_l)

    R_python_r=R_python[R_python>guess_tipy]
    Z_python_r=Z_python[R_python>guess_tipy]    
    py_interp_r=interp1d(Z_python_r,R_python_r)    

    ind_milieu=int(len(Z)/2)
    
    mini=[[],[]]
        
    ksi_z=[]
    for i in range (ind_milieu):
        if i==0:
            continue
        ind_droite=ind_milieu+i
        ind_gauche=ind_milieu-i
                     
        ksi_gauche=(R[ind_gauche]-py_interp_l(Z[ind_gauche]))**2
        mini[0].append([float(py_interp_l(Z[ind_gauche])),Z[ind_gauche]])

        ksi_droite=(R[ind_droite]-py_interp_r(Z[ind_droite]))**2  
        mini[1].append([float(py_interp_r(Z[ind_droite])),Z[ind_droite]])
            
        ksi_z.append(ksi_gauche+ksi_droite)
    n=i*2
    RMSd=np.sqrt(np.sum(ksi_z)/n)    
    
#    mini_inds=[]
#    for i in range (ind_milieu):
#        if i==0:
#            continue
#        ind_droite=ind_milieu+i
#        ind_gauche=ind_milieu-i
#        
#        if Z[ind_gauche]>min(Z_python_l):                
#            ksi_gauche=(R[ind_gauche]-py_interp_l(Z[ind_gauche]))**2
#            mini[0].append([float(py_interp_l(Z[ind_gauche])),Z[ind_gauche]])
#        else:
#            ksi_gauche=0
#        if Z[ind_droite]>min(Z_python_r):
#            ksi_droite=(R[ind_droite]-py_interp_r(Z[ind_droite]))**2  
#            mini[1].append([float(py_interp_r(Z[ind_droite])),Z[ind_droite]])
#        else:
#            ksi_droite=0
#            
#        ksi_z.append(ksi_gauche+ksi_droite)
#        if ksi_gauche+ksi_droite>200 and Z[ind_droite]>min(Z_python_r) and Z[ind_gauche]>min(Z_python_l):
#            print('ksi_gauche='+str(ksi_gauche)+' ; ind_gauche='+str(ind_gauche) + " ; Rinterp="+str(float(py_interp_l(Z[ind_gauche]))))
#            print('ksi_droite='+str(ksi_droite)+' ; ind_droite='+str(ind_droite) + " ; Rinterp="+str(float(py_interp_r(Z[ind_droite]))))
#    n=i*2
#    RMSd=np.sqrt(np.sum(ksi_z)/n)
    
#    return ksi_z,RMSd
    return np.sum(ksi_z),ksi_z,mini,RMSd    
#def sort_exp(R_python,Z_python,shape):
#    for val in range(shape):
#        level=shape-val-1
#        np.where(Z_python==level)