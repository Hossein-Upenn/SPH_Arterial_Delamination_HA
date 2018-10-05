#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 10:17:01 2017
@author: Hossein Ahmadzadeh: hsn.ahmadzadeh@gmail.com
The SPH code related to"Particle-based computational modelling of arterial disease, Roy. Soc Interface.
The code needs mpirun for python. use this command to run:
    mpirun -n 55 python SPH_Arterial_Delamination_MPI4_Hossein_Ahmadzadeh.py
"""
import sys, os  #import the required modules
import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import eigvals
import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank=comm.rank
import matplotlib.pyplot as plt
from matplotlib.patches import Arc    
from copy import  deepcopy
from numpy.linalg import inv
import math

almost_black = '#262626'

#Following steps should be done sequentially:
Prestretch_key="ON"      #Introduce deposition pre-stretches
Pressure_key="OFF"        #Pressurize the vessel 
Relaxation="OFF"          #perform relaxation at a desired pressure
GAG2_key="OFF"            #Filling the GAG particles with swelling
Damage_key="OFF"        #Damage at the tip of the GAG and delamination

Foldername="output"
pathname = os.path.dirname(sys.argv[0])  
os.chdir(os.path.abspath(pathname))

lambdaa=10000  #Lame constants
mu=89.71

if Pressure_key=="ON":  #density of the particles
    rho=100000
if GAG2_key=="ON":
    rho=50000    #Note down
    viscosity=1000
if Prestretch_key=="ON":
    rho=1000
if Relaxation=="ON":
    rho=100
    viscosity=1000
if Damage_key=="ON":
   rho_ini=5000 
   viscosity=1000

lines=np.loadtxt('/home/Geometry_input_systolic.txt')      #Geometry_input_systolic has r_out, r_in,...
r_in=lines[0]
r_mid=lines[1]
r_out=lines[2]
Angle=90*(np.pi/180)   #vessel opening angle
N_Elas=int(lines[4])   #No. of elastic laminae
h=3*8             
Nearest_search=80       #Nearest neighbor circle radius

T_start=0    #Use below suggested durations
T_final=40000
if Pressure_key=="ON":
    T_final=40000
if GAG2_key=="ON":
    T_final=40000
    T_mid=int(T_final/4)
    T_relax=500*6
if Damage_key=="ON":
    T_final=40000
if Prestretch_key=="ON":
    T_final=20000*0+2400*6
if Relaxation=="ON":
    T_final=500*12
    
Time_intervals=3*T_final
Time_Factor=int(Time_intervals/T_final)
Simulation_t=np.linspace(T_start,T_final,Time_intervals)   
Freq=np.round(T_final/80,0)     #result frequency--Giving 240 output   
plot_ratio=1.5
plot_range_gag=100

if Prestretch_key=="OFF":
    plot_ratio=1.2
else:
    plot_ratio=1.1
dot_size=2
    
Inner_Pressure=np.zeros(Simulation_t.size)
Dist_rate=np.zeros(Simulation_t.size)
Pre_rate=np.zeros(Simulation_t.size)
GAG_empty_rate=np.zeros(Simulation_t.size)
G_e=np.zeros((Simulation_t.size,2,2))
G_m=np.zeros((Simulation_t.size,2,2))
G_c=np.zeros((Simulation_t.size))

Density_rate=np.zeros(Simulation_t.size)
Contractility=np.zeros(Simulation_t.size)

Axial_Stretch=np.ones(Simulation_t.size)
Axial_Stretch_G=np.ones(Simulation_t.size)

GAG_density0=30                   #Fixed charge density in mEq/L )
GAG_density_pool=10*GAG_density0
GAG_size_factor=0.25 #change initial GAG size (0.25 is control, 1.2,2,3.55 are extended) 
GAG_pool_Angle=[50.5-2*GAG_size_factor,52.5+2*GAG_size_factor]

for t in range (0,Simulation_t.size):
    if Pressure_key=="ON" or GAG2_key=="ON" or Relaxation=="ON":
        Inner_Pressure[t]=(Simulation_t[t]<T_final)*(Simulation_t[t]/T_final)+(Simulation_t[t]>=T_final)*1
        Axial_Stretch_G[t]=(Simulation_t[t]<T_final)*(1+0.62*Simulation_t[t]/T_final)+(Simulation_t[t]>=T_final)*1.62
        Pre_rate[t]=(Simulation_t[t]<T_final)*(Simulation_t[t]/T_final)+(Simulation_t[t]>=T_final)*1
        G_e[t,1,1]=1+Pre_rate[t]*(1.9-1)
        G_e[t,0,0]=1/(Axial_Stretch_G[t]*G_e[t,1,1])
        G_m[t,0,0]=1
        G_m[t,1,1]=1+Pre_rate[t]*(1.2-1)
        G_c[t]=1+Pre_rate[t]*(1.25-1)
        Contractility[t]=1
    if GAG2_key=="ON":
        Density_rate[t]=(Simulation_t[t]>T_mid)*((Simulation_t[t]-T_mid)/(T_final-T_mid))*GAG_density_pool+(Simulation_t[t]>=T_final)*GAG_density_pool
        GAG_empty_rate[t]=(Simulation_t[t]>T_relax)*(Simulation_t[t]<T_mid)*((Simulation_t[t]-T_relax)/(T_mid-T_relax))+(Simulation_t[t]>=T_mid)*1  
    if Prestretch_key=="ON":
        Axial_Stretch_G[t]=(Simulation_t[t]<T_final)*(1+0.62*Simulation_t[t]/T_final)+(Simulation_t[t]>=T_final)*1.62
        Contractility[t]=(Simulation_t[t]<T_final)*(Simulation_t[t]/T_final)+(Simulation_t[t]>=T_final)*1  
        Pre_rate[t]=(Simulation_t[t]<T_final)*(Simulation_t[t]/T_final)+(Simulation_t[t]>=T_final)*1
        G_e[t,1,1]=1+Pre_rate[t]*(1.9-1)
        G_e[t,0,0]=1/(Axial_Stretch_G[t]*G_e[t,1,1])
        G_m[t,0,0]=1
        G_m[t,1,1]=1+Pre_rate[t]*(1.2-1)
        G_c[t]=1+Pre_rate[t]*(1.25-1)

Axial_Str=1.6       #Axial stretch
Pressure=75         #Displacement of the inner radius by pressurizing in microns

critical_stress=40
Damage_rate=np.zeros(Simulation_t.size)
for t in range (0,Simulation_t.size):
    Damage_rate[t]=(Simulation_t[t]<7000)*(0.005/ np.sqrt(np.log(2)))+(Simulation_t[t]>=7000)*(0.05/ np.sqrt(np.log(2)))


GAG_tip_indices=[[790,789,844,852],[763,762,871,879],[736,735,898,906],[690,689,945,953],[799,798,835,843]]  #From main script, GAG_tip_particles for each GAG_size_factor  [771,763,870,871] for 1
GAG_top_bottom_index=[809,818,819] #For All GAG sizes
if Damage_key=="ON":
    GAG_Con_rate=GAG_density_pool/20000

plt.plot(Dist_rate)    
Damage_restart=0
Restart=np.genfromtxt('Prestretch_output.txt',dtype=float, delimiter=' ')   #Output of the previous step. No input is needed for prestretch step.
    
Hour_glass_factor=1
if Pressure_key=="ON": 
    Hour_glass_factor=2
if Damage_key=="ON":
    Hour_glass_factor=0.15 
if GAG2_key=="ON":
    Hour_glass_factor=0.1 
if Relaxation=="ON":
    Hour_glass_factor=2

Width=round((r_mid-r_in)/(N_Elas-1))
Area=np.pi*((r_out**2-r_in**2)/4)*Angle/(np.pi/2)
del lines

Nodes1=int(np.genfromtxt('/home/mesh_systolic4.txt',dtype=float, delimiter=' ',skip_header=0, max_rows=1))  #reading nodes and mesh settings.
lines=np.genfromtxt('/home/mesh_systolic4.txt',dtype=float, delimiter=' ',skip_header=1, max_rows=Nodes1)
Elastic_radii=np.loadtxt('/Elastic_lam_rad.txt',delimiter=',')      #radiu of elastic lamina    

Coord=np.zeros((Nodes1,2))
Coord[:,0]=lines[:,0]         #Array of all nodes
Coord[:,1]=lines[:,1]

Elements1=int(np.genfromtxt('/home/mesh_systolic4.txt',dtype=float, delimiter=' ',skip_header=Nodes1, max_rows=1))
lines=np.genfromtxt('/home/mesh_systolic4.txt',dtype=float, delimiter=' ',skip_header=Nodes1+2, max_rows=Elements1)    

El=(np.zeros((Elements1,3)))
El[:,0:3]=lines[:,0:3]         #Array of all Elements

r_1=np.zeros(Nodes1)           #radius of each particle
r_1[:]=np.round(np.sqrt(Coord[:,0]**2+Coord[:,1]**2),2)

GAG_labels=np.zeros((Nodes1),dtype=bool)  #particle labels when modeling damage
GAG_Conc=np.zeros((Nodes1),dtype=float)
D=np.zeros((Simulation_t.size+1,Nodes1),)           #damage 

if rank==0:
    if not os.path.exists(Foldername):
        os.makedirs(Foldername)
    os.chdir(os.path.join(os.path.abspath(pathname),Foldername))
    if not os.path.exists("Figures"):
        os.makedirs("Figures")
    if not os.path.exists("Coords"):
        os.makedirs("Coords")
    if Damage_key=="ON":
        if not os.path.exists("D"):
            os.makedirs("D")   
        if not os.path.exists("GAG_label"):
            os.makedirs("GAG_label")  
        if not os.path.exists("Radial_Stress"):
            os.makedirs("Radial_Stress")  
        if not os.path.exists("GAG_Conc"):
            os.makedirs("GAG_Conc")  
        if not os.path.exists("Radial_Stretch"):
            os.makedirs("Radial_Stretch")  
        if not os.path.exists("Radial_Stretch_ini"):
            os.makedirs("Radial_Stretch_ini")  
    print ("Work directory is:",os.getcwd())
    print("Running %d parallel MPI processes" % comm.size)
    print("==================================================================")
################################### Cluster settings ##########################
comm_size=comm.size 
x = range(Nodes1)
m = int(math.ceil(Nodes1) / comm_size)
x_chunk = x[comm.rank*m:(comm.rank+1)*m]
comm.barrier()
################################Save coordinates###############################
if rank==0:
    print ("No. of nodes: %d" % Coord.shape[0])
    print ("No. of nodes from Comsol: %d" % Nodes1)

def Find_Elastic_Lamina(k):    
    radius=np.round(np.sqrt(Coord[k][0]**2+Coord[k][1]**2),2)
    itemindex = np.where(np.round(Elastic_radii,2)==radius)
    if itemindex[0].size != 0:
        return k
    else:
        return (-1)
Elastic_Indice=np.zeros((Nodes1),int) 
Elastic_Indice_chunk=np.zeros(len(x_chunk),dtype=int)
Elastic_Indice_chunk=np.array(list(map(Find_Elastic_Lamina,x_chunk)))
comm.Allgather([Elastic_Indice_chunk, MPI.DOUBLE],[Elastic_Indice, MPI.DOUBLE])
comm.barrier()
print("the rank is " + str(comm.rank))
Elastic_Indice=np.array(list(filter(lambda x:x !=-1,Elastic_Indice)))
############################Particle groups######################################
itemindex1 = np.where(r_1==r_out)
itemindex2 = np.where(r_1==r_in)
itemindex3 = np.where(np.round(Coord[:,1]/Coord[:,0],3)==0)
itemindex4 = np.where(np.round(Coord[:,0],2)==0)
itemindex5 = np.where(r_1==r_mid)

GAG_pool=np.array([i for i in range(len(r_1)) if r_1[i] > 653 and r_1[i] < 668 and \
                     np.arctan2(Coord[i,1],Coord[i,0])<math.radians(GAG_pool_Angle[1]) and np.arctan2(Coord[i,1],Coord[i,0])>math.radians(GAG_pool_Angle[0])])    #gag pool particles
multiple_GAG=False      #the case with multiple GAGs
if multiple_GAG==True:
    seperation_angle=[4.1,4.6]  #Tried [3.5,3.5   #seperation_angle[0] is the lower pool, seperation_angle[1] is the upper pool 
    GAG_pool=np.array([i for i in range(len(r_1)) if r_1[i] > 653 and r_1[i] < 668 and \
                         ((np.arctan2(Coord[i,1],Coord[i,0])<math.radians(GAG_pool_Angle[1]+seperation_angle[1]) and np.arctan2(Coord[i,1],Coord[i,0])>math.radians(GAG_pool_Angle[0]+seperation_angle[1])) \
                         or (np.arctan2(Coord[i,1],Coord[i,0])<math.radians(GAG_pool_Angle[1]-seperation_angle[0]) and np.arctan2(Coord[i,1],Coord[i,0])>math.radians(GAG_pool_Angle[0]-seperation_angle[0])))])    #gag pool particles

GAG_pool_layer=  np.array([i for i in range(len(r_1)) if r_1[i] > 653 and r_1[i] < 668])    #Particles within the pool layer

GAG_surround=np.array([i for i in range(len(r_1)) if np.arctan2(Coord[i,1],Coord[i,0])<math.radians(GAG_pool_Angle[1]+5) and np.arctan2(Coord[i,1],Coord[i,0])> math.radians(GAG_pool_Angle[0]-5)])    #Particles with gag pool
if multiple_GAG==True:
    GAG_surround=np.array([i for i in range(len(r_1)) if np.arctan2(Coord[i,1],Coord[i,0])<math.radians(GAG_pool_Angle[1]+seperation_angle[1]+5) and \
                           np.arctan2(Coord[i,1],Coord[i,0])> math.radians(GAG_pool_Angle[0]-5-seperation_angle[0])])    #Particles with gag pool

Inner=itemindex2[0]
Outer=itemindex1[0]
Horiz=itemindex3[0]
Ver=itemindex4[0]
Fix=deepcopy(Outer)         #Outer edge
Moving=deepcopy(Inner)      #Inner edge

Adv_Bound=itemindex5[0]
mask_Adv=np.zeros((Nodes1),dtype=bool)
mask_Media=np.zeros((Nodes1),dtype=bool)
mask_Ver=np.zeros((Nodes1),dtype=bool)
mask_Moving=np.zeros((Nodes1),dtype=bool)
mask_Fix=np.zeros((Nodes1),dtype=bool)
mask_Moving[Moving]=True
mask_Fix[Fix]=True

mask_Ver[Ver]=True
mask_Hor=np.zeros((Nodes1),dtype=bool)
mask_Hor[Horiz]=True
mask_Adv[r_1>=r_mid]=True
mask_Media[r_1<r_mid]=True
mask_Elastic=np.zeros((Nodes1),dtype=bool)
#mask_Elastic[Elastic_Indice]=True

mask_damaged=np.zeros((Nodes1),dtype=bool)
mask_D_0=np.zeros((Nodes1),dtype=bool)
mask_D_50=np.zeros((Nodes1),dtype=bool)
mask_D_25=np.zeros((Nodes1),dtype=bool)
mask_D_75=np.zeros((Nodes1),dtype=bool)

Damaged_Neighbors=np.zeros((Nodes1,Nearest_search),dtype=bool)

mask_pool=np.zeros((Nodes1),dtype=bool)
mask_pool[GAG_pool]=True

mask_pool_layer=np.zeros((Nodes1),dtype=bool)
mask_pool_layer[GAG_pool_layer]=True

mask_gag_surround=np.zeros((Nodes1),dtype=bool)
mask_gag_surround[GAG_surround]=True

mask_gag_tip=np.zeros((Nodes1),dtype=bool)
mask_gag_tip[GAG_tip_indices[int(GAG_size_factor)]]=True

if Damage_key=="ON":
    if Damage_restart==0:
        GAG_labels[mask_pool]=True
        GAG_Conc[:]=0

GAG_tip_particles=np.zeros(4,dtype=int)
################################ Plot the initial geometry ####################
if rank==0:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(Coord[mask_Adv,0],Coord[mask_Adv,1], facecolor='g',s=dot_size, edgecolor='black',linewidth='0')
    mask = (mask_Adv == 0) & (mask_Elastic == 0)
    ax.scatter(Coord[mask,0],Coord[mask,1], facecolor='b',s=dot_size, edgecolor='black',linewidth='0')   
    ax.scatter(Coord[Ver,0],Coord[Ver,1], facecolor='k',s=dot_size, edgecolor='black',linewidth='0')  
    ax.scatter(Coord[Horiz,0],Coord[Horiz,1], facecolor='k',s=dot_size, edgecolor='black',linewidth='0')   
    ax.scatter(Coord[mask_pool,0],Coord[mask_pool,1], facecolor='red',s=2*dot_size, edgecolor='black',linewidth='0') 
    if Damage_key=="OFF":
        ax.scatter(Coord[GAG_tip_indices[int(GAG_size_factor)],0],Coord[GAG_tip_indices[int(GAG_size_factor)],1], facecolor='cyan',s=dot_size, edgecolor='cyan',linewidth=0)
        ax.scatter(Coord[GAG_top_bottom_index,0],Coord[GAG_top_bottom_index,1], facecolor='cyan',s=dot_size, edgecolor='cyan',linewidth='0')

    for i in range (Nodes1):
        ax.annotate(i, (Coord[i,0],Coord[i,1]),fontsize=0.001)
    ax.set_xlabel('X (μm)',color=almost_black,fontsize=10)
    ax.set_ylabel('Y (μm)',color=almost_black,fontsize=10)
    ax.xaxis.set_ticks(np.append(np.arange(0, 700, 100),700))
    ax.yaxis.set_ticks(np.append(np.arange(0, 700, 100),700))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.get_xaxis().set_tick_params(direction='in', width=0.5,colors=almost_black)
    ax.get_yaxis().set_tick_params(direction='in', width=0.5,colors=almost_black)
    ax.set_xlim([-20,700])
    ax.set_ylim([-20,700])
    #ax.set_xlim([360,460])
    #ax.set_ylim([470,570])
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*1)
    spines_to_keep = ['bottom', 'left', 'right', 'top']
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
    plt.axes().set_aspect('equal')
    plt.show()
    fig.savefig('Initial_Config.png', format='png', dpi=1000)
    plt.close()
################################### Material Constants ########################
phi_e =np.zeros(Nodes1)
phi_m =np.zeros(Nodes1)
phi_c_2 =np.zeros(Nodes1)
phi_c_3 =np.zeros(Nodes1)
c_e=np.zeros(Nodes1)
lambda_e=np.zeros(Nodes1)
E=np.zeros(Nodes1)
c_c_1_2=np.zeros(Nodes1)
c_c_2_2=np.zeros(Nodes1)
c_c_1_3=np.zeros(Nodes1)
c_c_2_3=np.zeros(Nodes1)
c__c_1_2=np.zeros(Nodes1)
c__c_2_2=np.zeros(Nodes1)
c__c_1_3=np.zeros(Nodes1)
c__c_2_3=np.zeros(Nodes1)
alpha_0=np.zeros(Nodes1)

c_e[mask_Media]=mu                 #Interlamellar
E[mask_Media]=Hour_glass_factor*mu
c_c_1_2[mask_Media]=261.4   #Circumferential collagen+SMC (tension)
c_c_2_2[mask_Media]=0.24    #Circumferential collagen+SMC (tension)
c_c_1_3[mask_Media]=234.9   #Diagonal collagen (tension)
c_c_2_3[mask_Media]=4.08     #Diagonal collagen (tension)
c__c_1_2[mask_Media]=249.5   #Circumferential collagen+SMC (compression)
c__c_2_2[mask_Media]=0.15    #Circumferential collagen+SMC (compression)
c__c_1_3[mask_Media]=29.14   #Diagonal collagen (compression)
c__c_2_3[mask_Media]=4.08     #Diagonal collagen (compression)
alpha_0[mask_Media]=29.91*np.pi/180
phi_e[mask_Media]=0.4714
phi_c_2[mask_Media]=0.4714     #Circumferential collagen+SMC
phi_c_3[mask_Media]=0.0572*0.933  #diagonal collagen
lambda_e[mask_Media]=lambdaa

c_e[mask_Elastic]=mu         	   #Elastic layers
E[mask_Elastic]=Hour_glass_factor*mu
c_c_1_2[mask_Elastic]=261.4   #Circumferential collagen+SMC (tension)
c_c_2_2[mask_Elastic]=0.24    #Circumferential collagen+SMC (tension)
c_c_1_3[mask_Elastic]=239.9   #Diagonal collagen (tension)
c_c_2_3[mask_Elastic]=4.08     #Diagonal collagen (tension)
c__c_1_2[mask_Elastic]=249.5   #Circumferential collagen+SMC (compression)
c__c_2_2[mask_Elastic]=0.15    #Circumferential collagen+SMC (compression)
c__c_1_3[mask_Elastic]=29.17   #Diagonal collagen (compression)
c__c_2_3[mask_Elastic]=4.08     #Diagonal collagen (compression)
alpha_0[mask_Elastic]=29.91*np.pi/180
phi_e[mask_Elastic]=0.4714
phi_c_2[mask_Elastic]=0.4714    #Circumferential collagen+SMC
phi_c_3[mask_Elastic]=0.0572*0.933  #diagonal collagen
lambda_e[mask_Elastic]=lambdaa

c_e[mask_Adv]=mu
E[mask_Adv]=Hour_glass_factor*mu
c_c_1_2[mask_Adv]=234.9   #Circumferential collagen (tension)
c_c_2_2[mask_Adv]=4.08    #Circumferential collagen (tension)
c_c_1_3[mask_Adv]=234.9   #Diagonal collagen (tension)
c_c_2_3[mask_Adv]=4.08     #Diagonal collagen (tension)
c__c_1_2[mask_Adv]=29.14   #Circumferential collagen+SMC (compression)
c__c_2_2[mask_Adv]=4.08    #Circumferential collagen+SMC (compression)
c__c_1_3[mask_Adv]=29.14   #Diagonal collagen (compression)
c__c_2_3[mask_Adv]=4.08     #Diagonal collagen (compression)
alpha_0[mask_Adv]=29.91*np.pi/180
phi_e[mask_Adv]=0.01*0+0.03   #was 0.1
phi_c_2[mask_Adv]=0.97*0.056  #Circumferential collagen
phi_c_3[mask_Adv]=0.97*0.877  #diagonal collagen
lambda_e[mask_Adv]=lambdaa
############################# Find nearest neighbors ##########################
nbrs = NearestNeighbors(n_neighbors=Nearest_search, algorithm='ball_tree').fit(Coord)
distances, indices = nbrs.kneighbors(Coord)
for i in range (0,Nodes1):
    for j in range (1,Nearest_search):
        pt=indices[i,j]
        dist=np.sqrt((Coord[i][0]-Coord[pt][0])**2+(Coord[i][1]-Coord[pt][1])**2)
        if dist>h:
            indices[i,j]=-1
No_neighbors=np.zeros((Nodes1),dtype=int)     #Array showing No. of neighbors for each node
for i in range (0,Nodes1):
    Neihgbors=0
    for j in range (1,Nearest_search):
        if indices[i,j] != -1:
            Neihgbors=Neihgbors+1
    No_neighbors[i]=Neihgbors
########################### Volume of each particle ###########################
No_closest_neighbors=np.zeros((Nodes1),dtype=int)     #Array showing No. of closest neighbors for each node
closest_neighbpr_indic=np.zeros((Nodes1,20),dtype=int) 
closest_neighbpr_indic[:,:]=-1
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
Av_Near_Dist=np.zeros(Nodes1)           #Average nearest distance
Av_Long_Dist=np.zeros(Nodes1)           #Average largest distance
Volumes=np.zeros((Nodes1))
for i in range (0,Nodes1):
    x_i,y_i=Coord[i,0],Coord[i,1]
    Dist=np.zeros(No_neighbors[i])
    k=0
    for j in range (1,Nearest_search):
        if indices[i,j] != -1:
            x_j,y_j=Coord[indices[i,j],0],Coord[indices[i,j],1]
            Dist[k]=np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)
            k=k+1
    if k != 0:
        Av_Near_Dist[i]=np.amin(Dist)
        Av_Long_Dist[i]=np.amax(Dist)
      
for i in range (0,Nodes1):          #Compute volume from input mesh and closest neigbor list
    counter=0  
    Tot_El_Vol=0
    itemindex = np.where(El==i)
    Elem_No=itemindex[0].size
    No_closest_neighbors[i]=Elem_No*2
    x=np.zeros((3,1),dtype=float)
    y=np.zeros((3,1),dtype=float)
    for j in range (0,itemindex[0].size):
        pick1=int(El[itemindex[0][j],0])
        pick2=int(El[itemindex[0][j],1])
        pick3=int(El[itemindex[0][j],2])
        x=np.array([Coord[pick1,0],Coord[pick2,0],Coord[pick3,0]])
        y=np.array([Coord[pick1,1],Coord[pick2,1],Coord[pick3,1]])
        Tot_El_Vol=Tot_El_Vol+PolyArea(x,y)/3
        picks=np.array([pick1,pick2,pick3])
        closest_neighbpr_indic[i,counter]=list(filter(lambda x : x != i,picks))[0]
        closest_neighbpr_indic[i,counter+1]=list(filter(lambda x : x != i,picks))[1]
        counter += 2
        
    Volumes[i]=Tot_El_Vol
if rank==0:
    plt.hist(No_neighbors, bins='auto')
    plt.title("No of neighbours histogram")
    print ("Average No. of neighbors is: %d" % np.average(No_neighbors))
    print ("Minimum No. of neighbors is: %d" % np.amin(No_neighbors))
    print ("Maximum No. of neighbors is: %d" % np.amax(No_neighbors))
    print ("Total volume is: %5.2f" % np.sum(Volumes))
    print ("Total volume obtained from geometry is: %5.2f" % Area)
    print ("No. of elements with zero volume: %5.0f" % (Volumes==0).sum())
    print ("Average minimum distance between nodes is: %5.1f" % np.average(Av_Near_Dist))
    print ("Average maximum distance between nodes is: %5.1f" % np.average(Av_Long_Dist))
    if Damage_key=="OFF":
        print ("Density is: %5.1f" % rho)
    print ("T_final is: %5.1f" % T_final)
    print ("lambdaa is: %s" % lambdaa)
    print("pressure is: %s" % Pressure)
    print ("Folder name is: %s" % Foldername)
    print("============================================================================")  
########################## SPH main body ######################################
F_prefactor=10/ (np.pi * h **5 )
Current_Coord=np.zeros((Simulation_t.size+1,Nodes1,2),)
Current_Coord[:,:]=deepcopy(Coord[:])

Acc_tot=np.zeros((Nodes1,2),)
Velocity_tot=np.zeros((Nodes1,2),)

F_tot=np.zeros((Nodes1,2,2),)
P_tot=np.zeros((Nodes1,2,2),)
Sigma_tot=np.zeros((Nodes1,2,2),)
Sigma_result=np.zeros((Nodes1,2,2),)
A_tot=np.zeros((Nodes1,2,2),)
A_inverse=np.zeros((Nodes1,2,2),)
Gradient_W_tot=np.zeros((Nodes1,Nearest_search,2))
C_tot=np.zeros((Nodes1,2,2),)
C_damaged=np.zeros((Nodes1,2,2),)
J_tot=np.zeros((Nodes1,2,2),)
Current_Axial_Stretch=np.zeros((Nodes1,2,2),)
Ini_Principal_Str=np.zeros((Nodes1),)
RelaxationKey=np.zeros((Nodes1),)

SSD_radius=np.zeros(int(Simulation_t.size/Freq))
radius_max=np.zeros(int(Simulation_t.size/Freq))
radius_min=np.zeros(int(Simulation_t.size/Freq))
Ave_Velocity=np.zeros(int(Simulation_t.size/Freq))
Ave_Acc=np.zeros(int(Simulation_t.size/Freq))

Ave_Pressure=np.zeros(int(Simulation_t.size/Freq))
Ave_Pressure_Interior=np.zeros(int(Simulation_t.size/Freq))
Ave_Pressure_middle=np.zeros(int(Simulation_t.size/Freq))
Circum_stress_Hor=np.zeros(int(Simulation_t.size/Freq))
Circum_stress_Ver=np.zeros(int(Simulation_t.size/Freq))

flag=np.zeros((Nodes1),)
Rotated_radial_Stress=np.zeros((Nodes1),)
Rotated_radial_Stretch_ini=np.zeros((Nodes1),)
Rotated_radial_Stretch=np.zeros((Nodes1),)

Temp_Coord=np.zeros((Nodes1,2),)
Old_F_Tot=np.zeros((Nodes1,2,2),)

if Pressure_key=="ON" or GAG2_key=="ON"  or Relaxation=="ON" or Damage_key=="ON":
    Current_Coord[0]=Restart[:]

def Correction(i):  #Corrected kernel function
    A=np.zeros((2,2))
    A_inverse=np.zeros((2,2))
    Gradient_W=np.zeros((Nearest_search,2))
    X_i,Y_i=Coord[i,0],Coord[i,1]
    for j in range (1,No_neighbors[i]+1):
        pt=indices[i,j]
        dist=np.sqrt((Coord[i][0]-Coord[pt][0])**2+(Coord[i][1]-Coord[pt][1])**2)
        X_j,Y_j=Coord[pt,0],Coord[pt,1]
        R=np.array([[X_j-X_i,Y_j-Y_i]])
        Gradient_W[j]=F_prefactor*3*((h-dist)**2)*(-1)*R/dist   
        A += Volumes[pt]* np.outer(R,Gradient_W[j])
    A_inverse=np.linalg.solve(A, np.identity(2))     

    return {'A':A,'Gradient_W':Gradient_W,'A_inverse':A_inverse}

A_chunk=np.zeros((len(x_chunk),2,2),)
A_inverse_chunk=np.zeros((len(x_chunk),2,2),)
GradientW_chunk=np.zeros((len(x_chunk),Nearest_search,2))

for i in range (len(x_chunk)):
    A_chunk[i]=Correction(x_chunk[i])["A"]
    GradientW_chunk[i]=Correction(x_chunk[i])["Gradient_W"]
    A_inverse_chunk[i]=Correction(x_chunk[i])["A_inverse"]
    
comm.Allgather([A_chunk, MPI.DOUBLE],[A_tot, MPI.DOUBLE])
comm.Allgather([GradientW_chunk, MPI.DOUBLE],[Gradient_W_tot, MPI.DOUBLE])
comm.Allgather([A_inverse_chunk, MPI.DOUBLE],[A_inverse, MPI.DOUBLE])


for t in range (Simulation_t.size):       #Simulation start

    Deltat=(t ==0)*0+(t !=0)*(Simulation_t[t]-Simulation_t[t-1])    
    Temp_Coord=deepcopy(Current_Coord[t])
    
    if (Relaxation=="ON" and (t>0)) or (GAG2_key=="ON" and t>0) or (Damage_key=="ON"):
        Old_F_Tot=deepcopy(F_tot)

    if Damage_key=="ON" and np.max(D[t])>= 0.95 :    #Modify list of damaged neigbors
        for i in (GAG_surround):
            x_i,y_i=Temp_Coord[i,0],Temp_Coord[i,1]
            for j in range (1,No_neighbors[i]+1):
                pt=indices[i,j]
                if mask_damaged[pt]==True:
                    Damaged_Neighbors[i,j]==True
                    
            for j in range (1,No_neighbors[i]+1):
                pt=indices[i,j]
                if mask_damaged[pt]==False and Damaged_Neighbors[i,j]==False:   #undamaged particles in the shadow of the damaged particle
                    x_j,y_j=Temp_Coord[pt,0],Temp_Coord[pt,1]
                    dist1=np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)
                    theta1=np.arctan2(y_j-y_i,x_j-x_i)
                    for k in range (1,No_neighbors[i]+1):
                        pt2=indices[i,k]
                        if mask_damaged[pt2]==True:
                            x_k,y_k=Temp_Coord[pt2,0],Temp_Coord[pt2,1]
                            dist2=np.sqrt((x_i-x_k)**2+(y_i-y_k)**2)
                            theta2=np.arctan2(y_k-y_i,x_k-x_i)
                            if dist1>=dist2 and theta1<theta2+np.radians(20) and theta1>theta2-np.radians(20):
                                Damaged_Neighbors[i,j]==True  

    def SPH(i):                             #Calculate strain components
        C=np.zeros((2,2))
        F=np.zeros((2,2))
        J=np.ones((2,2))
        x_i,y_i=Temp_Coord[i,0],Temp_Coord[i,1]
        for j in range (1,No_neighbors[i]+1):
            pt=indices[i,j]
            x_j,y_j=Temp_Coord[pt,0],Temp_Coord[pt,1]
            r=np.array([x_j-x_i,y_j-y_i])
            Gradient_W2=Gradient_W_tot[i,j]
            F[0,0] += Volumes[pt]* r[0]*Gradient_W2[0]
            F[0,1] += Volumes[pt]* r[0]*Gradient_W2[1]
            F[1,0] += Volumes[pt]* r[1]*Gradient_W2[0]
            F[1,1] += Volumes[pt]* r[1]*Gradient_W2[1]
        F=np.dot(F,A_inverse[i])         
        
        C[0,0]=F[0,0]*F[0,0]+F[1,0]*F[1,0]
        C[0,1]=F[0,0]*F[0,1]+F[1,0]*F[1,1]
        C[1,0]=F[0,1]*F[0,0]+F[1,1]*F[1,0]
        C[1,1]=F[0,1]*F[0,1]+F[1,1]*F[1,1]
        J[0,0]=(F[0,0]*F[1,1]-F[0,1]*F[1,0])*1         
        return (F,C,J)

    F_chunk=np.zeros((len(x_chunk),2,2),)
    C_chunk=np.zeros((len(x_chunk),2,2),)
    J_chunk=np.zeros((len(x_chunk),2,2),)
    
    for i in range (len(x_chunk)):
        F_chunk[i]=SPH(x_chunk[i])[0]
        C_chunk[i]=SPH(x_chunk[i])[1]
        J_chunk[i]=SPH(x_chunk[i])[2]

    comm.Allgather([F_chunk, MPI.DOUBLE],[F_tot, MPI.DOUBLE])
    comm.Allgather([C_chunk, MPI.DOUBLE],[C_tot, MPI.DOUBLE])
    comm.Allgather([J_chunk, MPI.DOUBLE],[J_tot, MPI.DOUBLE])


    def Stress_Cal(i):          #Calculate stress components                       
        pi_pool=np.zeros((2,2))
        pi_pool_rotated=np.zeros((2,2))

        P=np.zeros((2,2))
        Sigma=np.zeros((2,2))
        Sigma_out=np.zeros((2,2))
        D_output=0
        c_f_pool=0
        Rotated_radial_Stress=0
        Rotated_radial_Stretch_local=0
        Radial_Stretch_ini=0
        Relaxationkey_local=0
        theta=np.arctan2(Temp_Coord[i,1],Temp_Coord[i,0])  
        theta1=-theta
        C_inverse=np.linalg.solve(C_tot[i], np.identity(2))
        F_inverse=np.linalg.solve(F_tot[i], np.identity(2))
        C_axial=(J_tot[i,0,0]**2)/(C_tot[i,0,0]*C_tot[i,1,1]-C_tot[i,1,0]*C_tot[i,0,1])


        S_e=np.zeros((2,2))     #Elastin stress
        S_e=phi_e[i]*(c_e[i]*np.identity(2)-c_e[i]*C_inverse+lambda_e[i]*np.log(J_tot[i,0,0])*C_inverse) 
        
        S_c_2=np.zeros((2,2))               #second family: Circumferencial + SMC
        Rotated_M_c_2=np.zeros((2,2)) 
        M_c_2=np.array(([0,0],[0,1]))           
        Rotated_M_c_2[0,0]=M_c_2[0,0]*np.cos(theta1)**2+M_c_2[1,1]*np.sin(theta1)**2 + \
            2*M_c_2[0,1]*np.sin(theta1)*np.cos(theta1)
        Rotated_M_c_2[1,1]=M_c_2[0,0]*np.sin(theta1)**2+M_c_2[1,1]*np.cos(theta1)**2 - \
            2*M_c_2[0,1]*np.sin(theta1)*np.cos(theta1) 
        Rotated_M_c_2[1,0]=-M_c_2[0,0]*np.cos(theta1)*np.sin(theta1)+M_c_2[1,1]*np.cos(theta1)*np.sin(theta1)+ \
            M_c_2[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)
        Rotated_M_c_2[0,1]=-M_c_2[0,0]*np.cos(theta1)*np.sin(theta1)+M_c_2[1,1]*np.cos(theta1)*np.sin(theta1)+ \
            M_c_2[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)
            
        lambda_c_2=C_tot[i,0,0]*Rotated_M_c_2[0,0]+C_tot[i,1,1]*Rotated_M_c_2[1,1]+C_tot[i,0,1]*Rotated_M_c_2[0,1]+C_tot[i,1,0]*Rotated_M_c_2[1,0]
        if lambda_c_2>=1:
            S_c_2 = c_c_1_2[i]*(lambda_c_2-1)*np.exp(c_c_2_2[i]*(lambda_c_2-1)**2)*Rotated_M_c_2        
        if lambda_c_2<1:
            S_c_2 = c__c_1_2[i]*(lambda_c_2-1)*np.exp(c__c_2_2[i]*(lambda_c_2-1)**2)*Rotated_M_c_2   
            

        S_c_3=np.zeros((2,2))
        Rotated_M_c_3=np.zeros((2,2))
        M_c_3=np.array(([0,0],[0,np.sin(alpha_0[i])**2]))      #Third family (diagonal)
        
        Rotated_M_c_3[0,0]=M_c_3[0,0]*np.cos(theta1)**2+M_c_3[1,1]*np.sin(theta1)**2 + \
            2*M_c_3[0,1]*np.sin(theta1)*np.cos(theta1)
        Rotated_M_c_3[1,1]=M_c_3[0,0]*np.sin(theta1)**2+M_c_3[1,1]*np.cos(theta1)**2 - \
            2*M_c_3[0,1]*np.sin(theta1)*np.cos(theta1) 
        Rotated_M_c_3[1,0]=-M_c_3[0,0]*np.cos(theta1)*np.sin(theta1)+M_c_3[1,1]*np.cos(theta1)*np.sin(theta1)+ \
            M_c_3[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)
        Rotated_M_c_3[0,1]=-M_c_3[0,0]*np.cos(theta1)*np.sin(theta1)+M_c_3[1,1]*np.cos(theta1)*np.sin(theta1)+ \
            M_c_3[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)    

        lambda_c_3=C_tot[i,0,0]*Rotated_M_c_3[0,0]+C_tot[i,1,1]*Rotated_M_c_3[1,1]+C_tot[i,0,1]*Rotated_M_c_3[0,1]+C_tot[i,1,0]*Rotated_M_c_3[1,0]+ \
                    C_axial*np.cos(alpha_0[i])**2
        if lambda_c_3>=1:
            S_c_3=c_c_1_3[i]*(lambda_c_3-1)*np.exp(c_c_2_3[i]*(lambda_c_3-1)**2)*Rotated_M_c_3   
        if lambda_c_3<1:
            S_c_3=c__c_1_3[i]*(lambda_c_3-1)*np.exp(c__c_2_3[i]*(lambda_c_3-1)**2)*Rotated_M_c_3   

        if GAG2_key=="ON" and mask_pool[i]==True:  #add the swelling
            #c_f_pool=0.8/(J_tot[i,0,0]-1+0.8)*Density_rate[t]        #Updated concentration of GAGs
            c_f_pool=Density_rate[t]
            pi_pool[0,0]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000    #Osmolarity of the surrounding bath is 300 mol/m^3 (15M NaCl)
            pi_pool[1,1]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000    #Osmolarity of the surrounding bath is 300 mol/m^3 (15M NaCl)

        if Damage_key=="ON" and GAG_labels[i]==True:
            if Damage_restart==0:
                if (np.mean(D[t-5000:t-1,i]!=0.99) and mask_pool[i]==False):   #Relaxation
                    pi_pool=np.zeros((2,2))
                    Relaxationkey_local=1
                    
                elif RelaxationKey.max()!=1:
                    if GAG_Conc[i] < GAG_density_pool:
                        c_f_pool=GAG_Conc[i] + GAG_Con_rate
                        pi_pool[0,0]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000          
                        pi_pool[1,1]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000   
                    else: 
                        c_f_pool=GAG_Conc[i]
                        pi_pool[0,0]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000          
                        pi_pool[1,1]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000    
                elif RelaxationKey.max()==1:
                        c_f_pool=GAG_Conc[i]
                        pi_pool[0,0]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000          
                        pi_pool[1,1]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000  
                        
            if Damage_restart!=0:    
                if (np.mean(D[t-5000:t-1,i]!=0.99) and mask_pool[i]==False and t>5000):   #Relaxation
                    pi_pool=np.zeros((2,2))
                    Relaxationkey_local=1
                
                elif RelaxationKey.max()!=1:
                    if GAG_Conc[i] < GAG_density_pool:
                        c_f_pool=GAG_Conc[i] + GAG_Con_rate
                        pi_pool[0,0]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000          
                        pi_pool[1,1]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000   
                    else: 
                        c_f_pool=GAG_Conc[i]
                        pi_pool[0,0]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000          
                        pi_pool[1,1]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000    
                elif RelaxationKey.max()==1:
                        c_f_pool=GAG_Conc[i]
                        pi_pool[0,0]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000          
                        pi_pool[1,1]=-(8.314*298*(np.sqrt(c_f_pool**2+300**2)-300))/1000                    
                
                
                
        S_G_e=np.zeros((2,2))     #Residual Stresses
        P_G_e=np.zeros((2,2))     #Residual Stresses
        Sigma_G_e=np.zeros((2,2))     #Residual Stresses
        S_c_G_2=np.zeros((2,2))
        S_c_G_3=np.zeros((2,2))
        
        if GAG2_key=="OFF":
            c_f_pool=phi_c_2[i]*Contractility[t]
            Circ_stretch=np.sqrt(lambda_c_2)
            pi_pool[1,1]=c_f_pool*550*Circ_stretch*(1-((1.1-Circ_stretch)/(1.1-0.6))**2) 
            
        elif GAG2_key=="ON" and mask_pool[i]==True:
            Circ_stretch=np.sqrt(lambda_c_2)
            pi_pool[1,1]=pi_pool[1,1]+phi_c_2[i]*(1-GAG_empty_rate[t])*550*Circ_stretch*(1-((1.1-Circ_stretch)/(1.1-0.6))**2)             

        elif GAG2_key=="ON" and mask_pool[i]==False:
            c_f_pool=phi_c_2[i]
            Circ_stretch=np.sqrt(lambda_c_2)
            pi_pool[1,1]=c_f_pool*550*Circ_stretch*(1-((1.1-Circ_stretch)/(1.1-0.6))**2) 
            
        pi_pool_rotated[0,0]=pi_pool[0,0]*np.cos(theta1)**2+pi_pool[1,1]*np.sin(theta1)**2 + \
            2*pi_pool[0,1]*np.sin(theta1)*np.cos(theta1)
        pi_pool_rotated[1,1]=pi_pool[0,0]*np.sin(theta1)**2+pi_pool[1,1]*np.cos(theta1)**2 - \
            2*pi_pool[0,1]*np.sin(theta1)*np.cos(theta1) 
        pi_pool_rotated[1,0]=-pi_pool[0,0]*np.cos(theta1)*np.sin(theta1)+pi_pool[1,1]*np.cos(theta1)*np.sin(theta1)+ \
            pi_pool[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)
        pi_pool_rotated[0,1]=-pi_pool[0,0]*np.cos(theta1)*np.sin(theta1)+pi_pool[1,1]*np.cos(theta1)*np.sin(theta1)+ \
            pi_pool[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)               
            

        if Prestretch_key=="ON":

            Q=np.array([[np.cos(theta1),np.sin(theta1)],[-np.sin(theta1),np.cos(theta1)]])
            Q2=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])            
            Rotated_G_e=np.dot(np.dot(np.dot(Q2,F_tot[i]),np.transpose(Q2)),G_e[t])
            Rotated_C_G_e=np.zeros((2,2))          #Neo-hookean
            C_G_e=np.dot(np.transpose(Rotated_G_e),Rotated_G_e) 

            Rotated_C_G_e[0,0]=C_G_e[0,0]*np.cos(theta1)**2+C_G_e[1,1]*np.sin(theta1)**2 + \
                2*C_G_e[0,1]*np.sin(theta1)*np.cos(theta1)
            Rotated_C_G_e[1,1]=C_G_e[0,0]*np.sin(theta1)**2+C_G_e[1,1]*np.cos(theta1)**2 - \
                2*C_G_e[0,1]*np.sin(theta1)*np.cos(theta1) 
            Rotated_C_G_e[1,0]=-C_G_e[0,0]*np.cos(theta1)*np.sin(theta1)+C_G_e[1,1]*np.cos(theta1)*np.sin(theta1)+ \
                C_G_e[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)
            Rotated_C_G_e[0,1]=-C_G_e[0,0]*np.cos(theta1)*np.sin(theta1)+C_G_e[1,1]*np.cos(theta1)*np.sin(theta1)+ \
                C_G_e[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)                

            C_G_e_inverse=np.linalg.solve(Rotated_C_G_e, np.identity(2))
            S_G_e=c_e[i]*np.identity(2)-c_e[i]*C_G_e_inverse+lambda_e[i]*np.log(J_tot[i,0,0])*C_G_e_inverse 
            P_G_e=np.dot(np.dot(np.dot(Q,Rotated_G_e),np.transpose(Q)),S_G_e)
            Sigma_G_e=(1/J_tot[i,0,0])*(np.dot(P_G_e,np.transpose(np.dot(np.dot(Q,Rotated_G_e),np.transpose(Q)))))


            Rotated_G_c_1=np.zeros((2,2))      #Circumferential Collagen +SMC
            M_c_2=np.array(([0,0],[0,1]))          
            RotatedM_c_2=np.dot(np.dot(np.dot(Q2,F_tot[i]),np.transpose(Q2)),M_c_2 )
            C_G_c_1=C_tot[i]
            Rotated_G_c_1[0,0]=C_G_c_1[0,0]*np.cos(theta)**2+C_G_c_1[1,1]*np.sin(theta)**2 + \
                2*C_G_c_1[0,1]*np.sin(theta)*np.cos(theta)
            Rotated_G_c_1[1,1]=C_G_c_1[0,0]*np.sin(theta)**2+C_G_c_1[1,1]*np.cos(theta)**2 - \
                2*C_G_c_1[0,1]*np.sin(theta)*np.cos(theta) 
            Rotated_G_c_1[1,0]=-C_G_c_1[0,0]*np.cos(theta)*np.sin(theta)+C_G_c_1[1,1]*np.cos(theta)*np.sin(theta)+ \
                C_G_c_1[1,0]*(np.cos(theta)**2-np.sin(theta)**2)
            Rotated_G_c_1[0,1]=-C_G_c_1[0,0]*np.cos(theta)*np.sin(theta)+C_G_c_1[1,1]*np.cos(theta)*np.sin(theta)+ \
                C_G_c_1[1,0]*(np.cos(theta)**2-np.sin(theta)**2)    
            lambda_c_G_1=(Rotated_G_c_1[0,0]*M_c_2[0,0]+Rotated_G_c_1[1,1]*M_c_2[1,1]+ \
                        Rotated_G_c_1[0,1]*M_c_2[0,1]+Rotated_G_c_1[1,0]*M_c_2[1,0])*(G_m[t,1,1]**2)
            
            if mask_Adv[i]==True:
                lambda_c_G_1=(Rotated_G_c_1[0,0]*M_c_2[0,0]+Rotated_G_c_1[1,1]*M_c_2[1,1]+ \
                            Rotated_G_c_1[0,1]*M_c_2[0,1]+Rotated_G_c_1[1,0]*M_c_2[1,0])*(G_c[t]**2)            
            if lambda_c_G_1>=1:
                S_c_G_2=c_c_1_2[i]*(lambda_c_G_1-1)*np.exp(c_c_2_2[i]*(lambda_c_G_1-1)**2)*Rotated_M_c_2   
            if lambda_c_G_1<1:
                S_c_G_2=c__c_1_2[i]*(lambda_c_G_1-1)*np.exp(c__c_2_2[i]*(lambda_c_G_1-1)**2)*Rotated_M_c_2
            P_c_G_2=np.dot(np.dot(np.dot(Q,RotatedM_c_2*G_m[t,1,1]),np.transpose(Q)),S_c_G_2)
            Sigma_c_G_2=np.dot(P_c_G_2,np.transpose(np.dot(np.dot(Q,RotatedM_c_2*G_m[t,1,1]),np.transpose(Q))))
            
            
            if mask_Adv[i]==True:
                P_c_G_2=np.dot(np.dot(np.dot(Q,RotatedM_c_2*G_c[t]),np.transpose(Q)),S_c_G_2)
                Sigma_c_G_2=np.dot(P_c_G_2,np.transpose(np.dot(np.dot(Q,RotatedM_c_2*G_c[t]),np.transpose(Q))))         
            

            Rotated_G_c_2=np.zeros((2,2))       #Third family collagen
            M_c_3=np.array(([0,0],[0,np.sin(alpha_0[i])**2]))
            RotatedM_c_3=np.dot(np.dot(np.dot(Q2,F_tot[i]),np.transpose(Q2)),M_c_3)
            C_G_c_2=C_tot[i]
            Rotated_G_c_2[0,0]=C_G_c_2[0,0]*np.cos(theta)**2+C_G_c_2[1,1]*np.sin(theta)**2 + \
                2*C_G_c_2[0,1]*np.sin(theta)*np.cos(theta)
            Rotated_G_c_2[1,1]=C_G_c_2[0,0]*np.sin(theta)**2+C_G_c_2[1,1]*np.cos(theta)**2 - \
                2*C_G_c_2[0,1]*np.sin(theta)*np.cos(theta) 
            Rotated_G_c_2[1,0]=-C_G_c_2[0,0]*np.cos(theta)*np.sin(theta)+C_G_c_2[1,1]*np.cos(theta)*np.sin(theta)+ \
                C_G_c_2[1,0]*(np.cos(theta)**2-np.sin(theta)**2)
            Rotated_G_c_2[0,1]=-C_G_c_2[0,0]*np.cos(theta)*np.sin(theta)+C_G_c_2[1,1]*np.cos(theta)*np.sin(theta)+ \
                C_G_c_2[1,0]*(np.cos(theta)**2-np.sin(theta)**2)    

            lambda_c_G_2=(Rotated_G_c_2[0,0]*M_c_3[0,0]+Rotated_G_c_2[1,1]*M_c_3[1,1]+Rotated_G_c_2[0,1]*M_c_3[0,1]+ \
                    Rotated_G_c_2[1,0]*M_c_3[1,0]+C_axial*np.cos(alpha_0[i])**2)*(G_c[t]**2)
            if lambda_c_G_2>=1: 
                S_c_G_3=c_c_1_3[i]*(lambda_c_G_2-1)*np.exp(c_c_2_3[i]*(lambda_c_G_2-1)**2)*Rotated_M_c_2   
            if lambda_c_G_2<1: 
                S_c_G_3=c__c_1_3[i]*(lambda_c_G_2-1)*np.exp(c__c_2_3[i]*(lambda_c_G_2-1)**2)*Rotated_M_c_2    

            P_c_G_3=np.dot(np.dot(np.dot(Q,RotatedM_c_3*G_c[t]),np.transpose(Q)),S_c_G_3)
            Sigma_c_G_3=np.dot(P_c_G_3,np.transpose(np.dot(np.dot(Q,G_c[t]*RotatedM_c_2),np.transpose(Q))))


            
        if Pressure_key=="ON" or GAG2_key=="ON" or Relaxation=="ON" or Damage_key=="ON":
            
            if GAG2_key=="ON" and mask_pool[i]==True:
                c_e_new=c_e[i]*(1-0.98*GAG_empty_rate[t])
                lambda_new=lambda_e[i]*(1-1*GAG_empty_rate[t])
                
            elif Relaxation=="moo" and mask_pool[i]==True:
                c_e_new=c_e[i]*(1-0.98)
                lambda_new=lambda_e[i]*(1-1)
                
            elif Damage_key=="ON" and mask_pool[i]==True:
                c_e_new=c_e[i]*(1-0.98)
                lambda_new=lambda_e[i]*(1-1)   

            elif Damage_key=="ON" and mask_pool[i]==False:
                c_e_new=c_e[i]*(1-D[t,i]*0.98/0.9)
                lambda_new=lambda_e[i]*(1-D[t,i]/0.9)                

            else:
                c_e_new=c_e[i]
                lambda_new=lambda_e[i]

            Q=np.array([[np.cos(theta1),np.sin(theta1)],[-np.sin(theta1),np.cos(theta1)]])
            Q2=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])            
            Rotated_G_e=np.dot(np.dot(np.dot(Q2,F_tot[i]),np.transpose(Q2)),G_e[-1])
            Rotated_C_G_e=np.zeros((2,2))          #Neo-hookean
            C_G_e=np.dot(np.transpose(Rotated_G_e),Rotated_G_e) 

            Rotated_C_G_e[0,0]=C_G_e[0,0]*np.cos(theta1)**2+C_G_e[1,1]*np.sin(theta1)**2 + \
                2*C_G_e[0,1]*np.sin(theta1)*np.cos(theta1)
            Rotated_C_G_e[1,1]=C_G_e[0,0]*np.sin(theta1)**2+C_G_e[1,1]*np.cos(theta1)**2 - \
                2*C_G_e[0,1]*np.sin(theta1)*np.cos(theta1) 
            Rotated_C_G_e[1,0]=-C_G_e[0,0]*np.cos(theta1)*np.sin(theta1)+C_G_e[1,1]*np.cos(theta1)*np.sin(theta1)+ \
                C_G_e[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)
            Rotated_C_G_e[0,1]=-C_G_e[0,0]*np.cos(theta1)*np.sin(theta1)+C_G_e[1,1]*np.cos(theta1)*np.sin(theta1)+ \
                C_G_e[1,0]*(np.cos(theta1)**2-np.sin(theta1)**2)                

            C_G_e_inverse=np.linalg.solve(Rotated_C_G_e, np.identity(2))
            S_G_e=c_e_new*np.identity(2)-c_e_new*C_G_e_inverse+lambda_new*np.log(J_tot[i,0,0])*C_G_e_inverse 
            P_G_e=np.dot(np.dot(np.dot(Q,Rotated_G_e),np.transpose(Q)),S_G_e)
            Sigma_G_e=(1/J_tot[i,0,0])*(np.dot(P_G_e,np.transpose(np.dot(np.dot(Q,Rotated_G_e),np.transpose(Q)))))


            Rotated_G_c_1=np.zeros((2,2))      #Circumferential Collagen +SMC
            M_c_2=np.array(([0,0],[0,1]))          
            RotatedM_c_2=np.dot(np.dot(np.dot(Q2,F_tot[i]),np.transpose(Q2)),M_c_2 )
            C_G_c_1=C_tot[i]
            Rotated_G_c_1[0,0]=C_G_c_1[0,0]*np.cos(theta)**2+C_G_c_1[1,1]*np.sin(theta)**2 + \
                2*C_G_c_1[0,1]*np.sin(theta)*np.cos(theta)
            Rotated_G_c_1[1,1]=C_G_c_1[0,0]*np.sin(theta)**2+C_G_c_1[1,1]*np.cos(theta)**2 - \
                2*C_G_c_1[0,1]*np.sin(theta)*np.cos(theta) 
            Rotated_G_c_1[1,0]=-C_G_c_1[0,0]*np.cos(theta)*np.sin(theta)+C_G_c_1[1,1]*np.cos(theta)*np.sin(theta)+ \
                C_G_c_1[1,0]*(np.cos(theta)**2-np.sin(theta)**2)
            Rotated_G_c_1[0,1]=-C_G_c_1[0,0]*np.cos(theta)*np.sin(theta)+C_G_c_1[1,1]*np.cos(theta)*np.sin(theta)+ \
                C_G_c_1[1,0]*(np.cos(theta)**2-np.sin(theta)**2)    
            
            lambda_c_G_1=(Rotated_G_c_1[0,0]*M_c_2[0,0]+Rotated_G_c_1[1,1]*M_c_2[1,1]+ \
                        Rotated_G_c_1[0,1]*M_c_2[0,1]+Rotated_G_c_1[1,0]*M_c_2[1,0])*(G_m[-1,1,1]**2)
            
            if mask_Adv[i]==True:
                lambda_c_G_1=(Rotated_G_c_1[0,0]*M_c_2[0,0]+Rotated_G_c_1[1,1]*M_c_2[1,1]+ \
                            Rotated_G_c_1[0,1]*M_c_2[0,1]+Rotated_G_c_1[1,0]*M_c_2[1,0])*(G_c[-1]**2)            
            if lambda_c_G_1>=1:
                S_c_G_2=c_c_1_2[i]*(lambda_c_G_1-1)*np.exp(c_c_2_2[i]*(lambda_c_G_1-1)**2)*Rotated_M_c_2   
            if lambda_c_G_1<1:
                S_c_G_2=c__c_1_2[i]*(lambda_c_G_1-1)*np.exp(c__c_2_2[i]*(lambda_c_G_1-1)**2)*Rotated_M_c_2
            P_c_G_2=np.dot(np.dot(np.dot(Q,RotatedM_c_2*G_m[-1,1,1]),np.transpose(Q)),S_c_G_2)
            Sigma_c_G_2=np.dot(P_c_G_2,np.transpose(np.dot(np.dot(Q,RotatedM_c_2*G_m[-1,1,1]),np.transpose(Q))))
            
            if mask_Adv[i]==True:
                P_c_G_2=np.dot(np.dot(np.dot(Q,RotatedM_c_2*G_c[-1]),np.transpose(Q)),S_c_G_2)
                Sigma_c_G_2=np.dot(P_c_G_2,np.transpose(np.dot(np.dot(Q,RotatedM_c_2*G_c[-1]),np.transpose(Q))))    
 

            Rotated_G_c_2=np.zeros((2,2))       #Third family collagen
            M_c_3=np.array(([0,0],[0,np.sin(alpha_0[i])**2]))
            RotatedM_c_3=np.dot(np.dot(np.dot(Q2,F_tot[i]),np.transpose(Q2)),M_c_3)
            C_G_c_2=C_tot[i]
            Rotated_G_c_2[0,0]=C_G_c_2[0,0]*np.cos(theta)**2+C_G_c_2[1,1]*np.sin(theta)**2 + \
                2*C_G_c_2[0,1]*np.sin(theta)*np.cos(theta)
            Rotated_G_c_2[1,1]=C_G_c_2[0,0]*np.sin(theta)**2+C_G_c_2[1,1]*np.cos(theta)**2 - \
                2*C_G_c_2[0,1]*np.sin(theta)*np.cos(theta) 
            Rotated_G_c_2[1,0]=-C_G_c_2[0,0]*np.cos(theta)*np.sin(theta)+C_G_c_2[1,1]*np.cos(theta)*np.sin(theta)+ \
                C_G_c_2[1,0]*(np.cos(theta)**2-np.sin(theta)**2)
            Rotated_G_c_2[0,1]=-C_G_c_2[0,0]*np.cos(theta)*np.sin(theta)+C_G_c_2[1,1]*np.cos(theta)*np.sin(theta)+ \
                C_G_c_2[1,0]*(np.cos(theta)**2-np.sin(theta)**2)    

            lambda_c_G_2=(Rotated_G_c_2[0,0]*M_c_3[0,0]+Rotated_G_c_2[1,1]*M_c_3[1,1]+Rotated_G_c_2[0,1]*M_c_3[0,1]+ \
                    Rotated_G_c_2[1,0]*M_c_3[1,0]+C_axial*np.cos(alpha_0[i])**2)*(G_c[-1]**2)
            if lambda_c_G_2>=1: 
                S_c_G_3=c_c_1_3[i]*(lambda_c_G_2-1)*np.exp(c_c_2_3[i]*(lambda_c_G_2-1)**2)*Rotated_M_c_2   
            if lambda_c_G_2<1: 
                S_c_G_3=c__c_1_3[i]*(lambda_c_G_2-1)*np.exp(c__c_2_3[i]*(lambda_c_G_2-1)**2)*Rotated_M_c_2    
            P_c_G_3=np.dot(np.dot(np.dot(Q,RotatedM_c_3*G_c[-1]),np.transpose(Q)),S_c_G_3)
            Sigma_c_G_3=np.dot(P_c_G_3,np.transpose(np.dot(np.dot(Q,G_c[-1]*RotatedM_c_2),np.transpose(Q))))
            
        
        if GAG2_key=="ON" and mask_pool[i]==True:
            Sigma=phi_e[i]*Sigma_G_e+(phi_c_2[i]*Sigma_c_G_2 +phi_c_3[i]*Sigma_c_G_3)*(1-GAG_empty_rate[t]) +0* pi_pool + pi_pool_rotated

        elif Relaxation=="moo" and mask_pool[i]==True:
            Sigma=phi_e[i]*Sigma_G_e+(phi_c_2[i]*Sigma_c_G_2 +phi_c_3[i]*Sigma_c_G_3)*(1-1) + pi_pool + pi_pool_rotated
        
        elif Damage_key=="ON":
            if mask_pool[i]==True:
                Sigma=phi_e[i]*Sigma_G_e+(phi_c_2[i]*Sigma_c_G_2 +phi_c_3[i]*Sigma_c_G_3)*(1-1) + pi_pool
            else:
                Sigma=phi_e[i]*Sigma_G_e+(phi_c_2[i]*Sigma_c_G_2 +phi_c_3[i]*Sigma_c_G_3)*(1-D[t,i]/0.99) + pi_pool            

        else:
            Sigma=phi_e[i]*Sigma_G_e+phi_c_2[i]*Sigma_c_G_2 +phi_c_3[i]*Sigma_c_G_3+ 0*pi_pool+pi_pool_rotated


        if Damage_key=="ON" and GAG_labels[i]==False and mask_Adv[i]==False and mask_Moving[i]==False and mask_Hor[i]==False and \
                mask_Fix[i]==False and mask_Ver[i]==False and mask_gag_surround[i]==True and mask_pool_layer[i]==True:    

            Rotated_radial_Stress=Sigma[0,0]*np.cos(theta)**2+Sigma[1,1]*np.sin(theta)**2+2*Sigma[0,1]*np.sin(theta)*np.cos(theta) 
            Rotated_radial_Stretch_local=np.sqrt(C_tot[i,0,0]*np.cos(theta)**2+C_tot[i,1,1]*np.sin(theta)**2+2*C_tot[i,0,1]*np.sin(theta)*np.cos(theta)) 

            if Rotated_radial_Stress > critical_stress and D[t,i]==0:
                Radial_Stretch_ini=Rotated_radial_Stretch_local
                D_output=0.001

            if D[t,i]>0:
                Radial_Stretch_ini=Rotated_radial_Stretch_ini[i]
                D_temp=np.exp(((Rotated_radial_Stretch_local-Radial_Stretch_ini)/Damage_rate[t])**2)-1

                if D_temp>D[t,i]:
                    D_output=D_temp
                elif D_temp<D[t,i]:
                    D_output=D[t,i]
                if D_output>0.99:
                    D_output=0.99

        else: 
            D_output=D[t,i]

        P=(J_tot[i,0,0]*np.dot(Sigma,np.transpose(F_inverse)))

        if (Relaxation=="ON" and (t > 0)) or (GAG2_key=="ON" and t<=T_relax and t>0) or (Damage_key=="ON" and RelaxationKey.max()==1):
            l=np.zeros((2,2))
            tau=np.zeros((2,2))
            d=np.zeros((2,2))
            l=(1/Deltat)*np.dot((F_tot[i]-Old_F_Tot[i]),F_inverse)
            d=(1/2)*(l+np.transpose(l))
            tau=2*viscosity*d
            P=P+J_tot[i,0,0]*tau*np.transpose(F_inverse)

        P=np.dot(P,A_inverse[i])
        #P=P+P_gag2
        Sigma_out=deepcopy(Sigma)
        Sigma=np.dot(Sigma,A_inverse[i])       

        return (P,Sigma,lambda_c_2,lambda_c_3,S_e,S_c_2,S_c_3,Sigma_out,D_output,c_f_pool,\
                Rotated_radial_Stress,Radial_Stretch_ini,Rotated_radial_Stretch_local,Relaxationkey_local)

    P_chunk=np.zeros((len(x_chunk),2,2),)
    Sigma_chunk=np.zeros((len(x_chunk),2,2),)
    Sigma_result_chunk=np.zeros((len(x_chunk),2,2),)
    D_chunk=np.zeros((len(x_chunk)),)
    c_f_chunk=np.zeros((len(x_chunk)),)
    Rotated_radial_Stress_chunk=np.zeros((len(x_chunk)),)
    Rotated_radial_Stretch_ini_chunk=np.zeros((len(x_chunk)),)
    Rotated_radial_Stretch_chunk=np.zeros((len(x_chunk)),)
    Relaxationkey_chunk=np.zeros((len(x_chunk)),)

    for i in range (len(x_chunk)):
        P_chunk[i]=Stress_Cal(x_chunk[i])[0]
        Sigma_chunk[i]=Stress_Cal(x_chunk[i])[1]
        Sigma_result_chunk[i]=Stress_Cal(x_chunk[i])[7]   #Note this
        D_chunk[i]=Stress_Cal(x_chunk[i])[8]
        c_f_chunk[i]=Stress_Cal(x_chunk[i])[9]
        Rotated_radial_Stress_chunk[i]=Stress_Cal(x_chunk[i])[10]
        Rotated_radial_Stretch_ini_chunk[i]=Stress_Cal(x_chunk[i])[11]
        Rotated_radial_Stretch_chunk[i]=Stress_Cal(x_chunk[i])[12]
        Relaxationkey_chunk[i]=Stress_Cal(x_chunk[i])[13]
        
    comm.Allgather([P_chunk, MPI.DOUBLE],[P_tot, MPI.DOUBLE])
    comm.Allgather([Sigma_chunk, MPI.DOUBLE],[Sigma_tot, MPI.DOUBLE])
    comm.Allgather([Sigma_result_chunk, MPI.DOUBLE],[Sigma_result, MPI.DOUBLE])

    if Damage_key=="ON":
        comm.Allgather([c_f_chunk, MPI.DOUBLE],[GAG_Conc, MPI.DOUBLE])
        comm.Allgather([Rotated_radial_Stress_chunk, MPI.DOUBLE],[Rotated_radial_Stress, MPI.DOUBLE])   
        comm.Allgather([Rotated_radial_Stretch_ini_chunk, MPI.DOUBLE],[Rotated_radial_Stretch_ini, MPI.DOUBLE])           
        comm.Allgather([Rotated_radial_Stretch_chunk, MPI.DOUBLE],[Rotated_radial_Stretch, MPI.DOUBLE])    
        comm.Allgather([D_chunk, MPI.DOUBLE],[D[t+1], MPI.DOUBLE])
        comm.Allgather([Relaxationkey_chunk, MPI.DOUBLE],[RelaxationKey, MPI.DOUBLE])        
        
    if Damage_key=="ON":
        for i in GAG_surround:
            if D[t,i]>=0.99 and GAG_labels[i]==False:
                GAG_labels[i]=True 
            elif GAG_labels[i]==True:
                GAG_labels[i]=True
    
    if Damage_key=="ON":
        max_angle=51.5
        min_angle=51.5
        for i in (GAG_surround):
            if GAG_labels[i]==True:
                theta=np.degrees(np.arctan2(Temp_Coord[i,1],Temp_Coord[i,0]))
                if theta>max_angle:
                    max_angle=theta
                if theta<min_angle:
                    min_angle=theta

        GAG_surround=np.array([i for i in range(Nodes1) if np.arctan2(Coord[i,1],Coord[i,0])<math.radians(max_angle+5) \
                               and np.arctan2(Coord[i,1],Coord[i,0])> math.radians(min_angle-5)])   
        mask_gag_surround=np.zeros((Nodes1),dtype=bool)
        mask_gag_surround[GAG_surround]=True
            
    mask_damaged[:]=False
    mask_D_0[:]=False  
    mask_D_25[:]=False
    mask_D_50[:]=False
    mask_D_75[:]=False   
    for i in range (Nodes1):
        if D[t,i]>0 and D[t,i]<0.25:
            mask_D_0[i]=True    
        if D[t,i]>=0.25 and D[t,i]<0.5:
            mask_D_25[i]=True   
        if D[t,i]>=0.5 and D[t,i]<0.75:
            mask_D_50[i]=True    
        if D[t,i]>=0.75 and D[t,i]<0.90:
            mask_D_75[i]=True    
        if D[t,i]>=0.90 and D[t,i]<0.98:
            mask_damaged[i]=True
           
    if Damage_key=="ON" and RelaxationKey.max()==1:
        rho=rho_ini
    elif Damage_key=="ON" and RelaxationKey.max()==0:
        rho=rho_ini

    if GAG2_key=="ON" and t<=T_relax:
        rho=100
    if GAG2_key=="ON" and t>T_relax:
        rho=50000
        
    def Acceleration(i):
        f=np.zeros(2)
        f_hg=np.zeros(2)
        X_i,Y_i=Coord[i,0],Coord[i,1]
        x_i,y_i=Temp_Coord[i,0],Temp_Coord[i,1]
        if Damage_key=="OFF" or  Damage_key=="ON" :
            for j in range (1,No_neighbors[i]+1):
                pt=indices[i,j]
                dist=np.sqrt((Coord[i][0]-Coord[pt][0])**2+(Coord[i][1]-Coord[pt][1])**2)
                X_j,Y_j=Coord[pt,0],Coord[pt,1]
                x_j,y_j=Temp_Coord[pt,0],Temp_Coord[pt,1]
                dist2=np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)
                r=np.array([x_j-x_i,y_j-y_i])
                R=np.array([X_j-X_i,Y_j-Y_i])
                W=F_prefactor*((h-dist)**3)
                Gradient_W2=Gradient_W_tot[i,j]
                delta_i=np.inner(np.transpose(np.dot(F_tot[i],np.transpose(R)))-r,r)/dist2
                delta_j=np.inner((np.transpose(np.dot(F_tot[pt],np.transpose(-R)))+r),(-r))/dist2
                f_hg=-((50)*E[i]*Volumes[i]*Volumes[pt]*W/(2*dist**2))* (delta_i+delta_j)*r/dist2
                f += Volumes[i]*Volumes[pt]* np.transpose(np.dot((P_tot[i]+P_tot[pt]), \
                                   np.transpose(Gradient_W2)))+f_hg
        elif Damage_key=="moo":
            for j in range (1,No_neighbors[i]+1):
                pt=indices[i,j]
                if Damaged_Neighbors[i,j]==False:
                    
                    dist=np.sqrt((Coord[i][0]-Coord[pt][0])**2+(Coord[i][1]-Coord[pt][1])**2)
                    X_j,Y_j=Coord[pt,0],Coord[pt,1]
                    x_j,y_j=Temp_Coord[pt,0],Temp_Coord[pt,1]
                    dist2=np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)
                    r=np.array([x_j-x_i,y_j-y_i])
                    R=np.array([X_j-X_i,Y_j-Y_i])
                    W=F_prefactor*((h-dist)**3)
                    Gradient_W2=Gradient_W_tot[i,j]
                    delta_i=np.inner(np.transpose(np.dot(F_tot[i],np.transpose(R)))-r,r)/dist2
                    delta_j=np.inner((np.transpose(np.dot(F_tot[pt],np.transpose(-R)))+r),(-r))/dist2
                    f_hg=-((50)*E[i]*Volumes[i]*Volumes[pt]*W/(2*dist**2))* (delta_i+delta_j)*r/dist2
                    f += Volumes[i]*Volumes[pt]* np.transpose(np.dot((P_tot[i]+P_tot[pt]), \
                                       np.transpose(Gradient_W2)))+f_hg
        return (f)


    force_chunk=np.zeros((len(x_chunk),2),)
    force_run=np.zeros((Nodes1,2),)
    for i in range (len(x_chunk)):
        force_chunk[i]=Acceleration(x_chunk[i])
    comm.Allgather([force_chunk, MPI.DOUBLE],[force_run, MPI.DOUBLE])


#LEAP-FROG Time integration:
    if t==0:
        Current_Coord[t+1,:]=Temp_Coord[:]       
        Velocity_tot[:]=0   #This is velocity at t=1/2
        
    if t != 0:
        Acc_tot[:,0]=(1/(Volumes[:]*rho))*force_run[:,0]
        Acc_tot[:,1]=(1/(Volumes[:]*rho))*force_run[:,1]
        Velocity_tot[:] += Acc_tot[:]*Deltat  #This is velocity at t+1/2
        Current_Coord[t+1,:]=Temp_Coord[:]+Velocity_tot[:]* Deltat    
    
        if Pressure_key=="ON":
            for i in range (Moving.size):
                pt=Moving[i]
                theta=np.arctan2(Current_Coord[0,pt,1],Current_Coord[0,pt,0]) 
                r_1_In=np.sqrt(Current_Coord[0,pt,0]**2+Current_Coord[0,pt,1]**2)
                #r_1_In=np.average(np.sqrt(Current_Coord[0,Moving,0]**2+Current_Coord[0,Moving,1]**2))
                New_r_1=r_1_In+Inner_Pressure[t]*Pressure
                Current_Coord[t+1, pt,0]=New_r_1*np.cos(theta)
                Current_Coord[t+1, pt,1]=New_r_1*np.sin(theta)    


        if GAG2_key=="ON" or Prestretch_key=="ON" or Relaxation=="ON" or Damage_key=="ON":
            for i in range (Moving.size):
                pt=Moving[i]
                theta=np.arctan2(Current_Coord[0,pt,1],Current_Coord[0,pt,0])
                r_1_In=np.sqrt(Current_Coord[0,pt,0]**2+Current_Coord[0,pt,1]**2)
                Current_Coord[t+1, pt,0]=r_1_In*np.cos(theta)
                Current_Coord[t+1, pt,1]=r_1_In*np.sin(theta)   
                        
        for i in range (Ver.size):
            pt=Ver[i]
            Current_Coord[t+1, pt,0]=0  
        
        for i in range (Horiz.size):
            pt=Horiz[i]
            Current_Coord[t+1, pt,1]=0  


    #comm.barrier()
    if rank==0:

        if t % Freq == 0:
            Time_Frame=int(t/Freq)
            
            SSD_radius[Time_Frame]=np.average(np.sqrt(Current_Coord[t,:,0]**2+Current_Coord[t,:,1]**2))
            radius_max[Time_Frame]=np.max(np.sqrt(Current_Coord[t,:,0]**2+Current_Coord[t,:,1]**2))
            radius_min[Time_Frame]=np.min(np.sqrt(Current_Coord[t,:,0]**2+Current_Coord[t,:,1]**2))
            Ave_Velocity[Time_Frame]=np.sqrt(np.average(Velocity_tot[:,0]**2+Velocity_tot[:,1]**2))
            Ave_Acc[Time_Frame]=np.sqrt(np.average(Acc_tot[:,0]**2+Acc_tot[:,1]**2))
        
            Ave_Pressure_temp=0             #Pressure
            Ave_Pressure_Interior_temp=0   #Pressure excluding Ver and Horiz
            Ave_Pressure_middle_temp=0
            
            Circum_stress_Hor[Time_Frame]=np.average(Sigma_result[Horiz,1,1])
            Circum_stress_Ver[Time_Frame]=np.average(Sigma_result[Ver,0,0])
            
            for i in range (Moving.size):
                pt=Moving[i]
                theta=np.arctan2(Current_Coord[0,pt,1],Current_Coord[0,pt,0]) 
                Q=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
                Rotated_Sigma_tot=np.dot(np.dot(Q,Sigma_result[pt]),np.transpose(Q))
                Ave_Pressure_temp +=Rotated_Sigma_tot[0,0]
                if mask_Ver[pt]==False and mask_Hor[pt]==False:
                    Ave_Pressure_Interior_temp +=Rotated_Sigma_tot[0,0]
            Ave_Pressure[Time_Frame]=Ave_Pressure_temp/Moving.size
            Ave_Pressure_Interior[Time_Frame]=Ave_Pressure_Interior_temp/(Moving.size-2)
    
            
            if Pressure_key=="ON":
                FileNumber=1000+t/Freq
            elif GAG2_key=="ON" and t<=3*T_mid:
                FileNumber=1700+t/Freq
            elif GAG2_key=="ON" and t>=3*T_mid:
                FileNumber=2000+t/Freq
            elif Damage_key=="ON":
                if Damage_restart==0:
                    FileNumber=2000+t/Freq 
                else:
                    FileNumber=3000+t/Freq 
            elif Relaxation=="ON":
                FileNumber=4000+t/Freq
            elif Prestretch_key=="ON":
                FileNumber=0+t/Freq
            if GAG2_key=="ON" or Damage_key=="ON":
                dot_size=20
            if GAG2_key=="ON" or Damage_key=="ON":
                dot_size=40
                
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.scatter(Current_Coord[t,mask_Adv,0],Current_Coord[t,mask_Adv,1], facecolor='green',s=dot_size, edgecolor='black',linewidth='0')
            ax.scatter(Current_Coord[t,mask_Elastic,0],Current_Coord[t,mask_Elastic,1], facecolor='r',s=dot_size, edgecolor='black',linewidth='0')
            if Damage_key=="OFF":
                mask = (mask_Adv == 0) & (mask_Elastic == 0)
                ax.scatter(Current_Coord[t,mask,0],Current_Coord[t,mask,1], facecolor='b',s=dot_size, edgecolor='black',linewidth='0')
                ax.scatter(Current_Coord[t,mask_pool,0],Current_Coord[t,mask_pool,1], facecolor='red',s=1.5*dot_size, edgecolor='black',linewidth='0') 
            elif Damage_key=="ON": 
                mask = (mask_Adv == 0) & (mask_Elastic == 0) & (GAG_labels==False)
                ax.scatter(Current_Coord[t,mask,0],Current_Coord[t,mask,1], facecolor='b',s=dot_size, edgecolor='black',linewidth=0)
                ax.scatter(Current_Coord[t,mask_damaged,0],Current_Coord[t,mask_damaged,1], facecolor='c',marker="*",s=dot_size,linewidth='0')  
                ax.scatter(Current_Coord[t,mask_D_0,0],Current_Coord[t,mask_D_0,1], facecolor='c',marker="v",s=dot_size,linewidth='0')  
                ax.scatter(Current_Coord[t,mask_D_25,0],Current_Coord[t,mask_D_25,1], facecolor='c',marker="<",s=dot_size,linewidth='0')  
                ax.scatter(Current_Coord[t,mask_D_50,0],Current_Coord[t,mask_D_50,1], facecolor='c',marker="^",s=dot_size,linewidth='0')  
                ax.scatter(Current_Coord[t,mask_D_75,0],Current_Coord[t,mask_D_75,1], facecolor='c',marker=">",s=dot_size,linewidth='0')  

                cm = plt.cm.get_cmap('Reds')
                cm2=[cm(i) for i in range(100,cm.N)]
                cmap = cm.from_list('Custom cmap', cm2, cm.N)
                Color=GAG_Conc[GAG_labels]
                im1=ax.scatter(Current_Coord[t,GAG_labels,0],Current_Coord[t,GAG_labels,1],cmap=cmap,c=Color,edgecolor='grey',linewidth=0,s=2*dot_size, vmin=0, vmax=GAG_density_pool) 
                cb1=fig.colorbar(im1, ax=ax,fraction=0.046, pad=0.04, format='%.0f')
                cb1.ax.tick_params(labelsize=8,length=1) 
                cb1.ax.set_ylabel('GAG concentration (mEq/l)', rotation=270, labelpad=15)

            for i in range (Elastic_radii.size):
                Elastic_nodes=[]
                for j in range (Elastic_Indice.size):
                    if np.round(r_1[Elastic_Indice[j]],2)==np.round(Elastic_radii[i],2):
                        Elastic_nodes.append(Elastic_Indice[j])
                        ax.plot(Current_Coord[t,Elastic_nodes,0],Current_Coord[t,Elastic_nodes,1],linewidth=1, alpha=1,color='r')

            ax.set_xlabel('Current radius (μm)',color=almost_black,fontsize=12)
            ax.set_ylabel('Current radius (μm)',color=almost_black,fontsize=12)
            ax.xaxis.set_ticks(np.append(np.arange(200, 600, 100),600))
            ax.yaxis.set_ticks(np.append(np.arange(300, 700, 100),700))
            #ax1.yaxis.set_ticks(np.arange(-10,110,10))
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            ax.get_xaxis().set_tick_params(direction='in', width=0.5,colors=almost_black)
            ax.get_yaxis().set_tick_params(direction='in', width=0.5,colors=almost_black)
            ax.set_xlim([-20,700])
            ax.set_ylim([-20,700])
            #ax.set_xlim([360,460])
            #ax.set_ylim([470,570])
            xleft, xright = ax.get_xlim()
            ybottom, ytop = ax.get_ylim()
            ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*1)
        
            spines_to_keep = ['bottom', 'left', 'right', 'top']
            for spine in spines_to_keep:
                ax.spines[spine].set_linewidth(0.5)
            plt.axes().set_aspect('equal')
            if GAG2_key=="ON" or Damage_key=="ON":
                center_x=np.average(Current_Coord[t,mask_pool,0])
                center_y=np.average(Current_Coord[t,mask_pool,1])
                plt.xlim(center_x-plot_range_gag,center_x+plot_range_gag)
                plt.ylim(center_y-plot_range_gag,center_y+plot_range_gag)
                #plt.xlim(260-20,660-20)
                #plt.ylim(260-20,660-20)
            else:
                plt.xlim(-plot_ratio*r_out*0,plot_ratio*r_out)
                plt.ylim(-plot_ratio*r_out*0,plot_ratio*r_out)
            plt.show()
            fig.savefig('Figures/%3.0f.pdf' %FileNumber, format='pdf')
            plt.close('all')

            np.savetxt('Coords/%3.0f.txt' %FileNumber, Current_Coord[t],fmt='%.3f')
            if Damage_key=="ON":
                np.savetxt('GAG_label/%3.0f.txt' %FileNumber,GAG_labels)   
                np.savetxt('D/%3.0f.txt' %FileNumber,D[t])  
                np.savetxt('Radial_Stress/%3.0f.txt' %FileNumber,Rotated_radial_Stress)  
                np.savetxt('GAG_Conc/%3.0f.txt' %FileNumber,GAG_Conc[:])    
                np.savetxt('Radial_Stretch_ini/%3.0f.txt' %FileNumber,Rotated_radial_Stretch_ini[:])    
                np.savetxt('Radial_Stretch/%3.0f.txt' %FileNumber,Rotated_radial_Stretch[:])  
            print ("Maximum acceleration is: %3.5f" % np.amax(Acc_tot[:]))
            print ("Minimum acceleration is: %3.5f" % np.amin(Acc_tot[:]))
            print ("Average acceleration is: %3.5f" % Ave_Acc[Time_Frame])
            print ("Maximum velocity is: %3.5f" % np.amax(Velocity_tot[:]))
            print ("Minimum velocity is: %3.5f" % np.amin(Velocity_tot[:]))
            print ("Average velocity is: %3.5f" % Ave_Velocity[Time_Frame])
            print ("Average radius is: %4.2f" % SSD_radius[Time_Frame])
            print ("Outer radius is: %4.2f" % radius_max[Time_Frame])
            print ("Inner radius is: %4.2f" % radius_min[Time_Frame])
            print ("Elapsed time is: %3.0f of %3.0f" %(t, Simulation_t.size))
            print('average J: ' + str(np.average(J_tot[:,0,0])))
            print('Average D: ' + str(np.average(D[t])))
            print('Maximum D: ' + str(np.max(D[t])))
            print('No of damaged particles: ' + str(np.array([mask_damaged]).sum()))
            print('No of particles with D>0: ' + str(np.array([mask_D_0]).sum()))
            print('No of particles with D>0.25: ' + str(np.array([mask_D_25]).sum()))
            print('No of particles with D>0.5: ' + str(np.array([mask_D_50]).sum()))
            print('No of particles with D>0.75: ' + str(np.array([mask_D_75]).sum()))
            print("============================================================================")
############################################Post processing####################
comm.barrier()
if rank==0:
    Output_Time=t
    np.savetxt('Initial_Coord.txt', Coord,fmt='%.3f')
    np.savetxt('Current_Coord.txt', Current_Coord[Output_Time],fmt='%.3f')
    
    fig = plt.figure()          #SSD_radii line plots
    plt.plot(SSD_radius,linewidth=2, color='r')
    plt.xlabel('time (sec)')
    plt.ylabel('mean radius (μm)')
    fig.savefig('SSD_radius.png', format='png', dpi=1200)
    plt.show()
    np.savetxt('SSD_radius.txt', np.transpose(SSD_radius),fmt='%.3f')
    
    
    fig = plt.figure()          #SSD_radii line plots
    plt.plot(radius_min,linewidth=2, color='r')
    plt.xlabel('time (sec)')
    plt.ylabel('Inner radius (μm)')
    fig.savefig('Inner_radius.png', format='png', dpi=1200)
    plt.show()
    np.savetxt('Inner_radius.txt', np.transpose(radius_min),fmt='%.3f')
    
    fig = plt.figure()          #SSD_radii line plots
    plt.plot(radius_max,linewidth=2, color='r')
    plt.xlabel('time (sec)')
    plt.ylabel('Outer radius (μm)')
    fig.savefig('Outer_radius.png', format='png', dpi=1200)
    plt.show()
    np.savetxt('Outer_radius.txt', np.transpose(radius_max),fmt='%.3f')
    
    fig = plt.figure()          #Average_Pressure
    plt.plot(radius_max, Ave_Pressure,linewidth=2, color='r')
    plt.xlabel('Outer radius (um)')
    plt.ylabel('Pressure (kPa)')
    fig.savefig('Pressure_radius.png', format='png', dpi=1200)
    plt.show()
    np.savetxt('Pressure_radius.txt', np.transpose((radius_max, Ave_Pressure)),fmt='%.3f')

    fig = plt.figure()          #Average Pressure except corner points
    plt.plot(radius_max, Ave_Pressure_Interior,linewidth=2, color='r')
    plt.xlabel('Outer radius (um)')
    plt.ylabel('Pressure (kPa)')
    fig.savefig('Pressure_radius2.png' , format='png', dpi=1200)
    plt.show()
    np.savetxt('Pressure_radius2.txt', np.transpose((radius_max,Ave_Pressure)),fmt='%.3f')      
    
    fig = plt.figure()          #Average_Pressure
    plt.plot(radius_max-radius_min, Ave_Pressure,linewidth=2, color='r')
    plt.xlabel('Thickness (um)')
    plt.ylabel('Pressure (kPa)')
    fig.savefig('Pressure_Thickness.png', format='png', dpi=1200)
    plt.show()
    np.savetxt('Pressure_Thickness.txt', np.transpose((radius_max-radius_min, Ave_Pressure)),fmt='%.3f')

    fig = plt.figure()          #Average Pressure except corner points
    plt.plot(radius_max-radius_min, Ave_Pressure_Interior,linewidth=2, color='r')
    plt.xlabel('Thickness (um)')
    plt.ylabel('Pressure (kPa)')
    fig.savefig('Pressure_Thickness2.png', format='png', dpi=1200)
    plt.show()
    np.savetxt('Pressure_Thickness2.txt', np.transpose((radius_max-radius_min,Ave_Pressure)),fmt='%.3f')    

    fig = plt.figure()          #Circumferencial stress for Vertical points
    plt.plot(radius_max, Circum_stress_Ver,linewidth=2, color='r')
    plt.xlabel('Outer radius (um)')
    plt.ylabel('Circumferential Stress (kPa)')
    fig.savefig('Circumferential_Stress_Ver.png', format='png', dpi=1200)
    plt.show()
    np.savetxt('Circumferential_Stress_Ver.txt', np.transpose((radius_max,Circum_stress_Ver)),fmt='%.3f')   

    fig = plt.figure()          #Circumferencial stress for Horiz points
    plt.plot(radius_max, Circum_stress_Hor,linewidth=2, color='r')
    plt.xlabel('Outer radius (um)')
    plt.ylabel('Circumferential Stress (kPa)')
    fig.savefig('Circumferential_Stress_Hor.png', format='png', dpi=1200)
    plt.show()
    np.savetxt('Circumferential_Stress_Hor.txt', np.transpose((radius_max,Circum_stress_Hor)),fmt='%.3f')   
############################# Time results ####################################
for t in [Simulation_t.size-1,0]:
    comm.barrier()
    if rank==0:
        Current=os.getcwd()
        if not os.path.exists('%s' %t):
            os.makedirs('%s' %t)
        os.chdir(os.path.join(Current,str(t))) 

    Output_Time=t
    Temp_Coord[:]=Current_Coord[t,:]         
    comm.barrier()
    C_tot=np.zeros((Nodes1,2,2))
    F_tot=np.zeros((Nodes1,2,2))
    F_out=np.zeros((Nodes1,2,2))
    J_tot=np.zeros((Nodes1,2,2))
    Sigma_tot=np.zeros((Nodes1,2,2))
    
    P_tot=np.zeros((Nodes1,2,2))   
    lambda_c_3=np.zeros((Nodes1))  
    lambda_c_2=np.zeros((Nodes1))  
    S_e=np.zeros((Nodes1,2,2))
    S_c_2=np.zeros((Nodes1,2,2))
    S_c_3=np.zeros((Nodes1,2,2))    

    C_cal=np.zeros((Nodes1,2,2))
    F_cal=np.zeros((Nodes1,2,2))
    J_cal=np.zeros((Nodes1,2,2))
    Sigma_cal=np.zeros((Nodes1,2,2))
    P_cal=np.zeros((Nodes1,2,2))
 
    F_chunk=np.zeros((len(x_chunk),2,2),)
    C_chunk=np.zeros((len(x_chunk),2,2),)
    J_chunk=np.zeros((len(x_chunk),2,2),)
    
    for i in range (len(x_chunk)):
        F_chunk[i]=SPH(x_chunk[i])[0]
        C_chunk[i]=SPH(x_chunk[i])[1]
        J_chunk[i]=SPH(x_chunk[i])[2]
    
    comm.Allgather([F_chunk, MPI.DOUBLE],[F_tot, MPI.DOUBLE])         
    comm.Allgather([C_chunk, MPI.DOUBLE],[C_tot, MPI.DOUBLE])
    comm.Allgather([J_chunk, MPI.DOUBLE],[J_tot, MPI.DOUBLE])
  
    P_chunk=np.zeros((len(x_chunk),2,2),)
    Sigma_chunk=np.zeros((len(x_chunk),2,2),)
    lambda_c_2_chunk=np.zeros((len(x_chunk)),)
    lambda_c_3_chunk=np.zeros((len(x_chunk)),)
    S_e_chunk=np.zeros((len(x_chunk),2,2),)
    S_c_2_chunk=np.zeros((len(x_chunk),2,2),)
    S_c_3_chunk=np.zeros((len(x_chunk),2,2),)

    for i in range (len(x_chunk)):
        P_chunk[i]=Stress_Cal(x_chunk[i])[0]
        Sigma_chunk[i]=Stress_Cal(x_chunk[i])[7]               
        lambda_c_2_chunk[i]=Stress_Cal(x_chunk[i])[2]
        lambda_c_3_chunk[i]=Stress_Cal(x_chunk[i])[3]
        S_e_chunk[i]=Stress_Cal(x_chunk[i])[4]
        S_c_2_chunk[i]=Stress_Cal(x_chunk[i])[5]
        S_c_3_chunk[i]=Stress_Cal(x_chunk[i])[6]
    
    comm.Allgather([P_chunk, MPI.DOUBLE],[P_tot, MPI.DOUBLE])
    comm.Allgather([Sigma_chunk, MPI.DOUBLE],[Sigma_tot, MPI.DOUBLE])
    comm.Allgather([lambda_c_2_chunk, MPI.DOUBLE],[lambda_c_2, MPI.DOUBLE])
    comm.Allgather([lambda_c_3_chunk, MPI.DOUBLE],[lambda_c_3, MPI.DOUBLE])
    comm.Allgather([S_e_chunk, MPI.DOUBLE],[S_e, MPI.DOUBLE])
    comm.Allgather([S_c_2_chunk, MPI.DOUBLE],[S_c_2, MPI.DOUBLE])
    comm.Allgather([S_c_3_chunk, MPI.DOUBLE],[S_c_3, MPI.DOUBLE])
   
    P_cal=deepcopy(P_tot)
    Sigma_cal=deepcopy(Sigma_tot)
    F_cal=deepcopy(F_tot)
    C_cal=deepcopy(C_tot)
    J_cal=deepcopy(J_tot)
    

    if rank==0:
        fig = plt.figure()
        if Pressure_key=="ON" or GAG2_key=="ON":
            dot_size=0.75
            
        ax = fig.add_subplot(1,1,1)
        ax.scatter(Current_Coord[Output_Time,mask_Adv,0],Current_Coord[Output_Time,mask_Adv,1], facecolor='g',s=dot_size, edgecolor='black',linewidth='0')
        ax.scatter(Current_Coord[Output_Time,mask_Elastic,0],Current_Coord[Output_Time,mask_Elastic,1], facecolor='r',s=dot_size, edgecolor='black',linewidth='0')
        ax.scatter(Current_Coord[Output_Time,mask_pool,0],Current_Coord[Output_Time,mask_pool,1], facecolor='black',s=2*dot_size, edgecolor='black',linewidth='0') 
        mask = (mask_Adv == 0) & (mask_Elastic == 0)
        ax.scatter(Current_Coord[Output_Time,mask,0],Current_Coord[Output_Time,mask,1], facecolor='b',s=dot_size, edgecolor='black',linewidth='0')
        for i in range (Elastic_radii.size):
            Elastic_nodes=[]
            for j in range (Elastic_Indice.size):
                if r_1[Elastic_Indice[j]]==Elastic_radii[i]:
                    Elastic_nodes.append(Elastic_Indice[j])
                ax.plot(Current_Coord[Output_Time,Elastic_nodes,0],Current_Coord[Output_Time,Elastic_nodes,1],linewidth=0.25, alpha=0.2,color='r')
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        plt.axes().set_aspect('equal')
        spines_to_keep = ['bottom', 'left', 'right', 'top']
        for spine in spines_to_keep:
            ax.spines[spine].set_linewidth(0.5)
        plt.tight_layout()
        plt.show()
        fig.savefig('Final_Config_%5.0f.png' %Output_Time, format='png', dpi=1200)
        plt.close()

        exclude_mask=mask_Moving+mask_Hor+mask_Ver            
        fig = plt.figure()                  #I_4 plots
        fontsize=8
        ax1 = fig.add_subplot(1,2,1)
        Colors=lambda_c_2[~exclude_mask]
        cm = plt.cm.get_cmap('brg')
        im1=ax1.scatter(Coord[~exclude_mask,0], Coord[~exclude_mask,1],c=Colors,s=dot_size,cmap=cm,edgecolor='black',linewidth='0') 
        ax1.set_title("lambda_c_2",fontsize = fontsize)
        cb1=fig.colorbar(im1, ax=ax1,fraction=0.046, pad=0.04, format='%.2f')
        cb1.ax.tick_params(labelsize=fontsize,length=1) 
        ax2 = fig.add_subplot(1,2,2)
        Colors2=lambda_c_3        
        im2=ax2.scatter(Coord[:,0], Coord[:,1],c=Colors2,s=dot_size,cmap=cm,edgecolor='black',linewidth='0') 
        cb2=fig.colorbar(im2, ax=ax2,fraction=0.046, pad=0.04, format='%.2f')
        cb2.ax.tick_params(labelsize=fontsize,length=1) 
        ax2.set_title('lambda_c_3',fontsize = fontsize)
        ax1.tick_params(labelsize=fontsize,length=3) 
        ax2.tick_params(labelsize=fontsize,length=3) 
        ax1.set_xlabel('X (um)',fontsize = fontsize)
        ax2.set_xlabel('X (um)',fontsize = fontsize)
        ax1.set_ylabel('Y (um)',fontsize= fontsize)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        fig.savefig('I4.png', format='png', dpi=1200)
        plt.close()
        
        
        fig = plt.figure()                  #Stress partitions
        fontsize=8
        ax1 = fig.add_subplot(1,3,1)
        Colors1=S_e[:,0,0]
        Colors2=S_c_2[:,0,0]
        Colors3=S_c_3[:,0,0]
        cm = plt.cm.get_cmap('brg')
        im1=ax1.scatter(Current_Coord[Output_Time,:,0], Current_Coord[Output_Time,:,1],c=Colors1,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        ax1.set_title("S_e",fontsize = fontsize)
        cb1=fig.colorbar(im1, ax=ax1,fraction=0.046, pad=0.04, format='%.2f')
        cb1.ax.tick_params(labelsize=fontsize,length=1) 
        ax2 = fig.add_subplot(1,3,2)
        im2=ax2.scatter(Current_Coord[Output_Time,:,0], Current_Coord[Output_Time,:,1],c=Colors2,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        cb2=fig.colorbar(im2, ax=ax2,fraction=0.046, pad=0.04, format='%.2f')
        cb2.ax.tick_params(labelsize=fontsize,length=1) 
        ax2.set_title('S_c_2',fontsize = fontsize)
        ax3 = fig.add_subplot(1,3,3)
        im3=ax3.scatter(Current_Coord[Output_Time,:,0], Current_Coord[Output_Time,:,1],c=Colors3,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        cb3=fig.colorbar(im3, ax=ax3,fraction=0.046, pad=0.04, format='%.2f')
        cb3.ax.tick_params(labelsize=fontsize,length=1) 
        ax3.set_title('S_c_3',fontsize = fontsize)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        fig.savefig('Stress_Partitions.png', format='png', dpi=1200)
        plt.close()        


        Sigma_rr=np.zeros((Nodes1))
        Sigma_hoop=np.zeros((Nodes1))
        Sigma_r_theta=np.zeros((Nodes1))
        P_rr=np.zeros((Nodes1))
        Stretch_rr=np.zeros((Nodes1))
        Stretch_hoop=np.zeros((Nodes1))
        Axial_Stretch_cal=np.zeros((Nodes1))
        
        for i in range (Nodes1):
            theta=np.arctan2(Coord[i,1],Coord[i,0])
            Sigma_rr[i]=Sigma_cal[i,0,0]*np.cos(theta)**2+Sigma_cal[i,1,1]*np.sin(theta)**2 + \
                        2*Sigma_cal[i,0,1]*np.sin(theta)*np.cos(theta)
            Sigma_hoop[i]=Sigma_cal[i,0,0]*np.sin(theta)**2+Sigma_cal[i,1,1]*np.cos(theta)**2 - \
                        2*Sigma_cal[i,0,1]*np.sin(theta)*np.cos(theta)    
            Stretch_rr[i]=np.sqrt(C_cal[i,0,0]*np.cos(theta)**2+C_cal[i,1,1]*np.sin(theta)**2 + \
                        2*C_cal[i,0,1]*np.sin(theta)*np.cos(theta))
            Stretch_hoop[i]=np.sqrt(C_cal[i,0,0]*np.sin(theta)**2+C_cal[i,1,1]*np.cos(theta)**2 - \
                        2*C_cal[i,0,1]*np.sin(theta)*np.cos(theta)  )
            Sigma_r_theta[i]=-Sigma_cal[i,0,0]*np.cos(theta)*np.sin(theta)+Sigma_cal[i,1,1]*np.cos(theta)*np.sin(theta)+ \
                        Sigma_cal[i,1,0]*(np.cos(theta)**2-np.sin(theta)**2)
                        
        exclude_mask=mask_Moving+mask_Hor+mask_Ver 
        fig = plt.figure()                  #Sigma_rr stress surfacre plots
        fontsize=8
        ax1 = fig.add_subplot(1,2,1)
        Colors1=Sigma_rr[~exclude_mask]
        Colors2=Sigma_hoop[~exclude_mask]
        cm = plt.cm.get_cmap('brg')
        im1=ax1.scatter(Current_Coord[Output_Time,~exclude_mask,0], Current_Coord[Output_Time,~exclude_mask,1],c=Colors1,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        ax1.set_title("Cauchy Stress_rr",fontsize = fontsize)
        cb1=fig.colorbar(im1, ax=ax1,fraction=0.046, pad=0.04, format='%.2f')
        cb1.ax.tick_params(labelsize=fontsize,length=1) 
        ax2 = fig.add_subplot(1,2,2)
        im2=ax2.scatter(Current_Coord[Output_Time,~exclude_mask,0], Current_Coord[Output_Time,~exclude_mask,1],c=Colors2,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        cb2=fig.colorbar(im2, ax=ax2,fraction=0.046, pad=0.04, format='%.2f')
        cb2.ax.tick_params(labelsize=fontsize,length=1) 
        ax2.set_title('Cauchy Stress_hoop',fontsize = fontsize)
        ax1.tick_params(labelsize=fontsize,length=3) 
        ax2.tick_params(labelsize=fontsize,length=3) 
        ax1.set_xlabel('X (um)',fontsize = fontsize)
        ax2.set_xlabel('X (um)',fontsize = fontsize)
        ax1.set_ylabel('Y (um)',fontsize = fontsize)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        fig.savefig('Sigma_Surface.png', format='png', dpi=1200)
        plt.close()
        
        fig = plt.figure()                  #Sigma_rr stress surfacre plots
        fontsize=8
        ax1 = fig.add_subplot(1,2,1)
        Colors1=Sigma_rr[~exclude_mask]
        Colors2=Sigma_hoop[~exclude_mask]
        cm = plt.cm.get_cmap('brg')
        im1=ax1.scatter(Coord[~exclude_mask,0], Coord[~exclude_mask,1],c=Colors1,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        ax1.scatter(Coord[mask_pool,0],Coord[mask_pool,1], facecolor='black',s=dot_size, edgecolor='black',linewidth='0') 
        ax1.set_title("Cauchy Stress_rr",fontsize = fontsize)
        cb1=fig.colorbar(im1, ax=ax1,fraction=0.046, pad=0.04, format='%.2f')
        cb1.ax.tick_params(labelsize=fontsize,length=1) 
        ax2 = fig.add_subplot(1,2,2)
        im2=ax2.scatter(Coord[~exclude_mask,0], Coord[~exclude_mask,1],c=Colors2,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        ax2.scatter(Coord[mask_pool,0],Coord[mask_pool,1], facecolor='black',s=dot_size, edgecolor='black',linewidth='0') 
        cb2=fig.colorbar(im2, ax=ax2,fraction=0.046, pad=0.04, format='%.2f')
        cb2.ax.tick_params(labelsize=fontsize,length=1) 
        ax2.set_title('Cauchy Stress_hoop',fontsize = fontsize)
        ax1.tick_params(labelsize=fontsize,length=3) 
        ax2.tick_params(labelsize=fontsize,length=3) 
        ax1.set_xlabel('X (um)',fontsize = fontsize)
        ax2.set_xlabel('X (um)',fontsize = fontsize)
        ax1.set_ylabel('Y (um)',fontsize = fontsize)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        fig.savefig('Sigma_Surface_Reference.png', format='png', dpi=1200)
        plt.close()
        


        exclude_mask=mask_Moving+mask_Hor+mask_Ver 
        Principal_Stretch=np.zeros(Nodes1)
        for i in range (Nodes1):
            Principal_Stretch[i]=np.sqrt(np.amax(eigvals(C_cal[i])))
        fig = plt.figure()                  #Maximum principal stretch surfacre plots
        fontsize=8
        ax1 = fig.add_subplot(1,2,1)
        Colors=(1/1)*Principal_Stretch[~exclude_mask]-0*Ini_Principal_Str[~exclude_mask]
        cm = plt.cm.get_cmap('brg')
        im1=ax1.scatter(Current_Coord[Output_Time,~exclude_mask,0], Current_Coord[Output_Time,~exclude_mask,1],c=Colors,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        #ax1.scatter(Current_Coord[t,~exclude_mask,0],Current_Coord[t,~exclude_mask,1], facecolor='black',s=dot_size, edgecolor='black',linewidth='0') 
        ax1.set_title("Maximum_Stretch, Current",fontsize = fontsize)
        cb1=fig.colorbar(im1, ax=ax1,fraction=0.046, pad=0.04, format='%.2f')
        cb1.ax.tick_params(labelsize=fontsize,length=1) 
        ax2 = fig.add_subplot(1,2,2)
        im2=ax2.scatter(Coord[~exclude_mask,0], Coord[~exclude_mask,1],c=Colors,s=dot_size,cmap=cm,edgecolor='black',linewidth='0') 
        #ax2.scatter(Coord[~exclude_mask,0],Coord[~exclude_mask,1], facecolor='black',s=2*dot_size, edgecolor='black',linewidth='0') 
        cb2=fig.colorbar(im2, ax=ax2,fraction=0.046, pad=0.04, format='%.2f')
        cb2.ax.tick_params(labelsize=fontsize,length=1) 
        ax2.set_title('Maximum_Stretch, Reference',fontsize = fontsize)
        ax1.tick_params(labelsize=fontsize,length=3) 
        ax2.tick_params(labelsize=fontsize,length=3) 
        ax1.set_xlabel('X (um)',fontsize = fontsize)
        ax2.set_xlabel('X (um)',fontsize = fontsize)
        ax1.set_ylabel('Y (um)',fontsize = fontsize)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        fig.savefig('Maximum_Stretch_Surface.png', format='png', dpi=1200)
        plt.close()


        exclude_mask=mask_Moving+mask_Hor+mask_Ver         
        fig = plt.figure()                  #hoop  stretch surfacre plots
        fontsize=8
        ax1 = fig.add_subplot(1,2,1)
        Colors=Stretch_hoop[~exclude_mask]
        cm = plt.cm.get_cmap('brg')
        im1=ax1.scatter(Current_Coord[Output_Time,~exclude_mask,0], Current_Coord[Output_Time,~exclude_mask,1],c=Colors,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        ax1.scatter(Current_Coord[Output_Time,mask_pool,0],Current_Coord[Output_Time,mask_pool,1], facecolor='black',s=dot_size, edgecolor='black',linewidth='0') 
        ax1.set_title("Hoop_Stretch, Current",fontsize = fontsize)
        cb1=fig.colorbar(im1, ax=ax1,fraction=0.046, pad=0.04, format='%.2f')
        cb1.ax.tick_params(labelsize=fontsize,length=1) 
        ax2 = fig.add_subplot(1,2,2)
        im2=ax2.scatter(Coord[~exclude_mask,0], Coord[~exclude_mask,1],c=Colors,s=dot_size,cmap=cm,edgecolor='black',linewidth='0') 
        ax2.scatter(Coord[mask_pool,0],Coord[mask_pool,1], facecolor='black',s=dot_size, edgecolor='black',linewidth='0') 
        cb2=fig.colorbar(im2, ax=ax2,fraction=0.046, pad=0.04, format='%.2f')
        cb2.ax.tick_params(labelsize=fontsize,length=1) 
        ax2.set_title('Hoop_Stretch, Reference',fontsize = fontsize)
        ax1.tick_params(labelsize=fontsize,length=3) 
        ax2.tick_params(labelsize=fontsize,length=3) 
        ax1.set_xlabel('X (um)',fontsize = fontsize)
        ax2.set_xlabel('X (um)',fontsize = fontsize)
        ax1.set_ylabel('Y (um)',fontsize = fontsize)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        fig.savefig('Hoop_Stretch_Surface.png', format='png', dpi=1200)
        plt.close()



        exclude_mask=mask_Moving+mask_Hor+mask_Ver             
        fig = plt.figure()                  #radial  stretch surfacre plots
        fontsize=8
        ax1 = fig.add_subplot(1,2,1)
        Colors=Stretch_rr[~exclude_mask]
        cm = plt.cm.get_cmap('brg')
        im1=ax1.scatter(Current_Coord[Output_Time,~exclude_mask,0], Current_Coord[Output_Time,~exclude_mask,1],c=Colors,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        ax1.scatter(Current_Coord[Output_Time,mask_pool,0],Current_Coord[Output_Time,mask_pool,1], facecolor='black',s=dot_size, edgecolor='black',linewidth='0') 
        ax1.set_title("Radial_Stretch, Current",fontsize = fontsize)
        cb1=fig.colorbar(im1, ax=ax1,fraction=0.046, pad=0.04, format='%.2f')
        cb1.ax.tick_params(labelsize=fontsize,length=1) 
        ax2 = fig.add_subplot(1,2,2)
        im2=ax2.scatter(Coord[~exclude_mask,0], Coord[~exclude_mask,1],c=Colors,s=dot_size,cmap=cm,edgecolor='black',linewidth='0') 
        ax2.scatter(Coord[mask_pool,0],Coord[mask_pool,1], facecolor='black',s=dot_size, edgecolor='black',linewidth='0') 
        cb2=fig.colorbar(im2, ax=ax2,fraction=0.046, pad=0.04, format='%.2f')
        cb2.ax.tick_params(labelsize=fontsize,length=1) 
        ax2.set_title('Radial_Stretch, Reference',fontsize = fontsize)
        ax1.tick_params(labelsize=fontsize,length=3) 
        ax2.tick_params(labelsize=fontsize,length=3) 
        ax1.set_xlabel('X (um)',fontsize = fontsize)
        ax2.set_xlabel('X (um)',fontsize = fontsize)
        ax1.set_ylabel('Y (um)',fontsize = fontsize)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        fig.savefig('Radial_Stretch_Surface.png', format='png', dpi=1200)
        plt.close()


        exclude_mask=mask_Moving+mask_Hor+mask_Ver         
        Axial_Stretch_cal[:]=(J_cal[:,0,0])/(Stretch_hoop[:]*Stretch_rr[:])
        fig = plt.figure()                  #axial  stretch surfacre plots
        fontsize=8
        ax1 = fig.add_subplot(1,2,1)
        Colors=Axial_Stretch_cal[~exclude_mask]
        cm = plt.cm.get_cmap('brg')
        im1=ax1.scatter(Current_Coord[Output_Time,~exclude_mask,0], Current_Coord[Output_Time,~exclude_mask,1],c=Colors,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        ax1.set_title("Axial_Stretch, Current",fontsize = fontsize)
        cb1=fig.colorbar(im1, ax=ax1,fraction=0.046, pad=0.04, format='%.2f')
        cb1.ax.tick_params(labelsize=fontsize,length=1) 
        ax2 = fig.add_subplot(1,2,2)
        im2=ax2.scatter(Coord[~exclude_mask,0], Coord[~exclude_mask,1],c=Colors,s=dot_size,cmap=cm,edgecolor='black',linewidth='0') 
        cb2=fig.colorbar(im2, ax=ax2,fraction=0.046, pad=0.04, format='%.2f')
        cb2.ax.tick_params(labelsize=fontsize,length=1) 
        ax2.set_title('Axial_Stretch, Reference',fontsize = fontsize)
        ax1.tick_params(labelsize=fontsize,length=3) 
        ax2.tick_params(labelsize=fontsize,length=3) 
        ax1.set_xlabel('X (um)',fontsize = fontsize)
        ax2.set_xlabel('X (um)',fontsize = fontsize)
        ax1.set_ylabel('Y (um)',fontsize = fontsize)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        fig.savefig('Axial_Stretch__Surface.png', format='png', dpi=1200)
        plt.close()


        
        Line_orientation=36*np.pi/180
        Line_data_indic=np.zeros(1,dtype=int)   #stress line plots
        All_Angles=np.array([np.arctan2(Coord[i,1],Coord[i,0]) for i in range(Nodes1)])
        for i in range (0,Nodes1):
            if np.round(np.arctan2(Coord[i,1],Coord[i,0]),2) == np.round(Line_orientation,2):
                Line_data_indic=np.append(Line_data_indic,i)
        Line_data_indic=np.delete(Line_data_indic,0)
        YYY=np.argsort(Coord[Line_data_indic,0])
        Line_data_indic=Line_data_indic[YYY]
        
        fig = plt.figure()      
        fontsize=8
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(r_1[Line_data_indic[1:]],Sigma_rr[Line_data_indic[1:]],'r')
        ax1.set_title("Cauchy Stress_rr",fontsize = fontsize)
        ax1.set_ylabel('Cauchy Stress_rr',fontsize = fontsize)
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(r_1[Line_data_indic[1:]],Sigma_hoop[Line_data_indic[1:]],'r')
        ax2.set_title("σ_hoop",fontsize = fontsize)
        ax2.set_ylabel('σ_hoop',fontsize = fontsize)
        plt.tight_layout()
        plt.show()
        fig.savefig('Line_Stress.png' , format='png', dpi=1200)
        np.savetxt('Sigma_rr_line.txt', np.transpose((r_1[Line_data_indic],Sigma_rr[Line_data_indic])),fmt='%.3f')
        np.savetxt('Sigma_hoop_line.txt', np.transpose((r_1[Line_data_indic],Sigma_hoop[Line_data_indic])),fmt='%.3f')
        plt.close()        


        Calculated_radial_stress=np.zeros((Sigma_hoop[Line_data_indic].shape[0]))
        Line_data_radii=np.zeros((len(Line_data_indic)))
        for j in range (len(Line_data_indic)):
            Line_data_radii[j]=np.sqrt(Current_Coord[t,Line_data_indic[j],0]**2+Current_Coord[t,Line_data_indic[j],1]**2)
        for j in range (Line_data_indic.shape[0]-1):
            Calculated_radial_stress[j]=-np.mean(Sigma_hoop[Line_data_indic[j:]])*(Line_data_radii[-1]-Line_data_radii[j])/Line_data_radii[j]

        fig = plt.figure()          #Radial stress lines
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot((Line_data_radii[0:]-Line_data_radii[0])/(Line_data_radii[-1]-Line_data_radii[0]),Calculated_radial_stress[0:],linewidth=1.5, color=almost_black)
        ax1.set_xlabel('Normalized current radius',fontsize=10, color=almost_black)
        ax1.set_ylabel('Radial stress (kPa)',fontsize=10, color=almost_black)
        ax1.get_xaxis().set_tick_params(direction='in', width=1,colors=almost_black)
        ax1.get_yaxis().set_tick_params(direction='in', width=1,colors=almost_black)
        plt.xticks(np.append(np.arange(0, 1, 0.2),1))
        ax1.tick_params(axis='both', which='major', labelsize=8)
        #ax1.set_xlim(xlimit_range)
        xleft, xright = ax1.get_xlim()
        ybottom, ytop = ax1.get_ylim()
        ax1.set_aspect(abs((xright-xleft)/(ybottom-ytop))*1)
        spines_to_keep = ['bottom', 'left', 'right', 'top']
        for spine in spines_to_keep:
            ax1.spines[spine].set_linewidth(0.5)
            ax1.spines[spine].set_color(almost_black)
        ax1.axvline(x=(Line_data_radii[5]-Line_data_radii[1])/(Line_data_radii[-1]-Line_data_radii[1]),linestyle='--', color='k',alpha=0.5,linewidth=0.5)

        ax2 = fig.add_subplot(1,2,2)   #Circumferential stress
        x=(Line_data_radii[1:5]-Line_data_radii[1])/(Line_data_radii[-1]-Line_data_radii[1])
        y=Sigma_hoop[Line_data_indic[1:5]]
        z=np.polyfit(x,y,deg=1)
        f=np.poly1d(z)
        Boundary=f((Line_data_radii[5]-Line_data_radii[1])/(Line_data_radii[-1]-Line_data_radii[1]))
        np.insert(Sigma_hoop[Line_data_indic[1:]],4,Boundary)
        Boundary_index=(Line_data_radii[5]-Line_data_radii[1])/(Line_data_radii[-1]-Line_data_radii[1])

        ax2.plot((Line_data_radii[1:6]-Line_data_radii[1])/(Line_data_radii[-1]-Line_data_radii[1]), \
                 np.append(Sigma_hoop[Line_data_indic[1:5]],Boundary),linewidth=1.5, color=almost_black)
        ax2.plot((Line_data_radii[5:]-Line_data_radii[1])/(Line_data_radii[-1]-Line_data_radii[1]),Sigma_hoop[Line_data_indic[5:]],linewidth=1.5, color=almost_black)

        ax2.set_xlabel('Normalized current radius',fontsize=10, color=almost_black)
        ax2.set_ylabel('Circumferential stress (kPa)',fontsize=10, color=almost_black)
        ax2.get_xaxis().set_tick_params(direction='in', width=1,colors=almost_black)
        ax2.get_yaxis().set_tick_params(direction='in', width=1,colors=almost_black)
        plt.xticks(np.append(np.arange(0, 1, 0.2),1))
        plt.yticks(np.append(np.arange(0, 600, 100),600))
        ax2.tick_params(axis='both', which='major', labelsize=8)
        #ax1.set_xlim(xlimit_range)
        xleft, xright = ax2.get_xlim()
        ybottom, ytop = ax2.get_ylim()
        ax2.set_aspect(abs((xright-xleft)/(ybottom-ytop))*1)
        spines_to_keep = ['bottom', 'left', 'right', 'top']
        for spine in spines_to_keep:
            ax2.spines[spine].set_linewidth(0.5)
            ax2.spines[spine].set_color(almost_black)
        ax2.axvline(x=(Line_data_radii[5]-Line_data_radii[1])/(Line_data_radii[-1]-Line_data_radii[1]),linestyle='--', color='k',alpha=0.5,linewidth=0.5)
        plt.tight_layout()
        fig.savefig('Transmural_stress.png', format='png', dpi=1200)
        plt.show()
        plt.close()

        fig = plt.figure()      
        fontsize=8
        ax1 = fig.add_subplot(1,3,1)
        mask=mask_Moving & ~mask_Ver & ~mask_Hor
        ax1.plot(Coord[mask,0],Sigma_rr[mask],'r')
        ax1.set_title("Radial Stress",fontsize = fontsize)
        ax1.set_ylabel('Radial Stress',fontsize = fontsize)
        ax2 = fig.add_subplot(1,3,2)
        ax2.plot(Coord[mask,0],Sigma_hoop[mask],'r')
        ax2.set_title("Hoop Stress",fontsize = fontsize)
        ax2.set_ylabel('Hoop Stress',fontsize = fontsize)
        ax3 = fig.add_subplot(1,3,3)
        ax3.plot(Coord[mask,0],Axial_Stretch_cal[mask],'r')
        ax3.set_title("Axial_Stretch",fontsize = fontsize)
        ax3.set_ylabel('Axial_Stretch',fontsize = fontsize)
        plt.tight_layout()
        plt.show()
        fig.savefig('Moving_Stress.png' , format='png', dpi=1200)
        plt.close()
       
        fig = plt.figure()                  #J surfacre plots
        fontsize=8
        ax1 = fig.add_subplot(1,2,1)
        Colors=J_tot[:,0,0]
        cm = plt.cm.get_cmap('brg')
        im1=ax1.scatter(Current_Coord[Output_Time,:,0], Current_Coord[Output_Time,:,1],c=Colors,s=dot_size/2,cmap=cm,edgecolor='black',linewidth='0') 
        ax1.set_title("J, Current",fontsize = fontsize)
        cb1=fig.colorbar(im1, ax=ax1,fraction=0.046, pad=0.04, format='%.2f')
        cb1.ax.tick_params(labelsize=fontsize,length=1) 
        ax2 = fig.add_subplot(1,2,2)
        im2=ax2.scatter(Coord[:,0], Coord[:,1],c=Colors,s=dot_size,cmap=cm,edgecolor='black',linewidth='0') 
        cb2=fig.colorbar(im2, ax=ax2,fraction=0.046, pad=0.04, format='%.2f')
        cb2.ax.tick_params(labelsize=fontsize,length=1) 
        ax2.set_title('J, Reference',fontsize = fontsize)
        ax1.tick_params(labelsize=fontsize,length=3) 
        ax2.tick_params(labelsize=fontsize,length=3) 
        ax1.set_xlabel('X (um)',fontsize = fontsize)
        ax2.set_xlabel('X (um)',fontsize = fontsize)
        ax1.set_ylabel('Y (um)',fontsize= fontsize)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.show()
        fig.savefig('J.png' , format='png', dpi=1200)
        plt.close()
        os.chdir(Current)       
#End