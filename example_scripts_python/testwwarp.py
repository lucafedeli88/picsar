from mpi4py import MPI
import sys
import os
currentdir=os.getcwd()
sys.path.append(currentdir+'/python_bin/')
print(currentdir+'/python_bin/')
import picsarpy as pxr
import numpy as np
import warp as wp

#### Input parameters (replace input file)
pxr.default_init()
##Simulation box size and duration
pxr.picsar.nx_global_grid=10
pxr.picsar.ny_global_grid=10
pxr.picsar.nz_global_grid=10
pxr.picsar.xmin=0.0
pxr.picsar.ymin=0.0
pxr.picsar.zmin=0.0
pxr.picsar.tmax=0.5

## Resolution
pxr.picsar.dx=6.5e-7
pxr.picsar.dy=6.5e-7
pxr.picsar.dz=6.5e-7

## Maxwell solver
pxr.picsar.norderx=2
pxr.picsar.nordery=2
pxr.picsar.norderz=2


## Current deposition order
pxr.picsar.nox=1
pxr.picsar.noy=1
pxr.picsar.noz=1

## Tiling parameters
pxr.picsar.ntilex=5
pxr.picsar.ntiley=5
pxr.picsar.ntilez=5

## Particles species
pxr.picsar.nspecies=2 # Set number of particle species
pxr.pdistr=2 # Random distribution for particles
# Set particle species 1 properties
pxr.set_particle_species_properties(1,"electron",1.,-1.,10,0.,0.,0.,1.3,1.3,1.3,0.,0.,0.1,0.,0.,0.)
# Set particle species 2 properties
pxr.set_particle_species_properties(2,"proton",1836.,-1.,10,0.,0.,0.,1.3,1.3,1.3,0.,0.,0.,0.,0.,0.)



#### MPI INIT
pxr.mpi_minimal_init()
pxr.mpi_initialise()
mpirank= pxr.picsar.rank


#### Init particle distributions  for each species (with tiles)
pxr.initall() #Fortran routine
#initall_py() # Python init (To be written)


#### PIC LOOP with intrinsic step function
ntsteps=10
#start=MPI.Wtime()
#pxr.step(ntsteps)
#endt=MPI.Wtime()
#if (mpirank==0):
#    print("Total simulation time with step in Fortran 90 (s) ="+str(endt-start))
def steppy(nt):
    for i in range(0,nt):
        # Push particles
        pxr.push_particles()
        pxr.particle_bcs()
        
        # Deposit currents on the grid
        pxr.depose_currents_on_grid_jxjyjz()
        pxr.current_bcs()

        # Push EM fields
        pxr.push_bfield()
        pxr.bfield_bcs()
        pxr.push_efield()
        pxr.efield_bcs()
        pxr.push_bfield()
        pxr.bfield_bcs()

        # Compute diags
        pxr.calc_diags()

        # print diags
        if (mpirank==0):
            print("Iteration number "+str(i)+"|| time (s) ="+str(pxr.picsar.dt*i))


def pntfieldswarp():
    wp.w3d.nx = pxr.picsar.nx_global
    wp.w3d.ny = pxr.picsar.ny_global
    wp.w3d.nz = pxr.picsar.nz_global
    wp.w3d.xmmin = pxr.picsar.xmin
    wp.w3d.xmmax = pxr.picsar.xmax
    wp.w3d.ymmin = pxr.picsar.ymin
    wp.w3d.ymmax = pxr.picsar.ymax
    wp.w3d.zmmin = pxr.picsar.zmin
    wp.w3d.zmmax = pxr.picsar.zmax
    
    wp.em = wp.EM3D(nxguard=pxr.picsar.nxguards,
                    nyguard=pxr.picsar.nyguards,
                    nzguard=pxr.picsar.nzguards)

    wp.em.finalize()
    del wp.em.fields.Ex; wp.em.fields.Ex = pxr.picsar.ex
    del wp.em.fields.Ey; wp.em.fields.Ey = pxr.picsar.ey
    del wp.em.fields.Ez; wp.em.fields.Ez = pxr.picsar.ez
    del wp.em.fields.Bx; wp.em.fields.Ex = pxr.picsar.bx
    del wp.em.fields.By; wp.em.fields.Ey = pxr.picsar.by
    del wp.em.fields.Bz; wp.em.fields.Ez = pxr.picsar.bz
    
pntfieldswarp()

#### PIC LOOP written in python


#start=MPI.Wtime()
steppy(ntsteps)
#endt=MPI.Wtime()
#if (mpirank==0):
#    print("Total simulation time with step in python (s) ="+str(endt-start))

wp.winon()


## MPICLOSE

#pxr.mpi_close()
