import numpy as np
import mdtraj as md
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import math
import sys
import time
from Step10_subroutine_Read_DATA import *
from Step10_subroutine_Charge_Mobility import *
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

keyword = str(sys.argv[2]) + 'K' #'700K' #'555K'
config = 1
sampling = 1
f_mag = 10**(float(sys.argv[1])) #1e-5  #electric field strength eV/sigma
print(f_mag)
f_vecs = f_mag*np.array([[1,0,0],[0,1,0],[0,0,1]])

dopants = int(sys.argv[3])  #number of dopant


traj_path = ['Trajectory_1/']
file_folder = '/home/ciwang/LAMPPS/CG_Simulation/Multiple_Snapshot/CG_NVT_' + keyword + '_8Cells/'  # 700K or 555K
ham_file = file_folder + 'ECG_Results/Diagonalization/Hamiltonian_' + keyword + '_Config_' + str(config) +'-' + str(sampling) + '.npy'
eigval_file = file_folder + 'ECG_Results/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
eigvec_file = file_folder + 'ECG_Results/Diagonalization/Eigen_Vector_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
topology_path = 'CG_topology.gro'


no_atom_per_mol = 15 #Number of CG particles in a molecule
no_frame = 12 #Number of snapshot in each trajectory folder
tmax = 1 #total number of time frame


############### Read LAMMPS DATA & Trajectory Files ###############
#### NVT 700K ###
if keyword=='700K':
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_700_8Cells/'
  data_file = traj_folder + 'CG_NVT_700K_8Cells_Eq.data'

#### NVT 555K ###
if keyword=='555K':
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_555K_8Cells_IBI66/'
  data_file = traj_folder + 'CG_NVT_555K_8Cells_Initial.data'

#### NVT 515K ###
if keyword=='515K':
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_515K_8Cells/'
  data_file = traj_folder + 'CG_NVT_515K_8Cells_Initial.data'

######## Initiate DATA class ########
lampps_data = read_data(data_file, no_atom_per_mol)
n = lampps_data.total_no_mol 
box = lampps_data.box_size

############ Read binary trajectory file & basic trajectory information ############
for folder in range(len(traj_path)):    
  traj = md.load((traj_folder+traj_path[folder]+'traj.dcd'), top=topology_path)
  total_frames = traj.n_frames
  traj_xyz = traj.xyz*10 #Convert unit from nm to angstrom
  print('\nProcessing folder: %s' %traj_path[folder][:-1])
  print("# of total frame: %s" %(total_frames))

######## Calculate center of mass ########
t = np.arange((total_frames-no_frame),total_frames)[config-1]
lampps_data.calculate_COM(traj_xyz[t,:,:])
#lampps_data.calculate_Rij(lampps_data.COM,0)
rBB_all = np.zeros((tmax,n,3),dtype = float)
for t in range(tmax):
  rBB_all[t] = lampps_data.COM


############### Read Hamiltonian, Eigenvalues and Eigenvectors  ###############
ham_all = []
ham_all.append(np.load(ham_file))
ham_all = np.array(ham_all)
#print(ham_all.shape)

eigvals_all = []
eigvals_all.append(np.load(eigval_file))
eigvals_all = np.array(eigvals_all)
#print(eigvals_all.shape)

eigvecs_all = []
eigvecs_all.append(np.load(eigvec_file))
eigvecs_all = np.array(eigvecs_all)
#print(eigvecs_all.shape)


########## David's system ##########
#ovitofile = 'ovito0-51.trj'
#trjs, steps  = get_trjs(ovitofile)
#ovito_raw, box = get_ovito(ovitofile, tmax)
#n = get_nbb(ovito_raw[0])
#rBB_all = np.zeros((tmax,n,3),dtype = float)
#for t in range(tmax):
#    rBB_all[t] = get_ellipsoids(ovito_raw[t]) #[nbb,3(x,y,z)]

#fileo = open('ham0-51.pkl', 'rb')
#ham_all = pickle.load(fileo)
#fileo.close()
#fileo = open('eigvals0-51.pkl', 'rb')
#eigvals_all = pickle.load(fileo)
#fileo.close()
#fileo = open('eigvecs0-51.pkl', 'rb')
#eigvecs_all = pickle.load(fileo)
#fileo.close()
########## David's system ##########


if n != ham_all.shape[1]:
    print("Number of beads in simulation != Dimension of hamiltonian!")
    exit


mobility = np.zeros((tmax,np.shape(f_vecs)[0]), dtype = float)
d_pop = np.zeros((tmax,np.shape(f_vecs)[0]), dtype = float)
d_mob = np.zeros((tmax,np.shape(f_vecs)[0]), dtype = float)


for t in range(tmax):
    print('Trj = {}'.format(t), flush=True)
    ham = ham_all[t]
    eigvals = eigvals_all[t]
    eigvecs = eigvecs_all[t]
    rBB = rBB_all[t]
    
    ## Calculate kij Matrix ##
    lam_coup = 0.4301307255 #eV
    g_coup = 0.05 #0.005 #unitless
    #kbT = 0.0478225 #eV, 555K
    kbT = 0.02585 #eV, 300K
    hbar = 6.58212e-16 #eV*s
    ham *= (np.ones((n,n),dtype = int) - np.identity(n)) #set diagonal = 0
    s_mat = ((eigvecs.T)**2)@ham**2@(eigvecs**2) #eV**2
    s_mat *= (np.ones((n,n),dtype = int) - np.identity(n)) #set diagonal = 0 CHECKED
    print("Percentage of the nonzero coupling btw states: %4.1f" %(100*np.nonzero(s_mat)[1].shape[0]/float(n*n)))
    
    output_folder = file_folder + 'Charge_Mobility/G' + str(g_coup).replace(".", "") + '_' + sys.argv[3] + '/'
    
    a_temp = np.array([np.sum(eigvecs**4,axis=0)]) #IPR per state CHECKED
    lam_mat = lam_coup*(a_temp.T + a_temp)
    #print('---Max---')
    #print((s_mat/lam_mat**2).max())
    #max_index = np.argwhere(s_mat/lam_mat**2==(s_mat/lam_mat**2).max())
    #print(s_mat[max_index[0][0],max_index[0][1]])
    #print(lam_mat[max_index[0][0],max_index[0][1]]**2)
    #print('---Average---')
    #print((s_mat/lam_mat**2).mean())
    #print((s_mat).mean())
    #print((lam_mat**2).mean())

    
    #print(np.amax((s_mat)*g_coup**2/lam_mat**2)) #Confirm Troisi 2014 Eq 22 is True
    #print(np.average((s_mat)*g_coup**2/lam_mat**2))
    
    e_temp = np.array([eigvals])
    e_mat = -1*(e_temp - e_temp.T) #CHECKED
    # '-1*' b/c I'm lookin at a hole in basis of electron HOMOs
    #print(e_mat.shape)

  
    cen_index = np.zeros((n),dtype = int)
    for i in range(n):
      cen_index[i] = (eigvecs[:,i]**2).argmax() #Index of centers of each MO 
    #print((eigvecs[:,0]**2).max(), eigvecs[cen_index[0],0]**2)

    
    r_mat = np.zeros((n,n,3),dtype = float)
    for i in range(n):
      #r_mat[i,:,:] = lampps_data.calculate_Rij(lampps_data.COM,i)
      r_mat[i,:,:] = lampps_data.calculate_Rij(lampps_data.COM[cen_index,:], i)

    ########## David's system ##########
    #cen_vec = np.zeros((n,3),dtype = float)
    #for i in range(n):
    #    cen_vec[i] = get_center(eigvecs[:,i]**2,rBB,box) #centers of each MO CHECKED
    #r_mat = cen_vec[np.newaxis,:,:]-cen_vec[:,np.newaxis,:]
    #r_mat -= box*np.around(r_mat/box) #r_mat(n[i],n[j],3(xyz)) CHECKED
    ########## David's system ##########
    #print(r_mat[0,1,:])
    #print(r_mat[1,0,:])

   

    for f in range(np.shape(f_vecs)[0]):
        f_mat = np.sum(r_mat*f_vecs[f],axis=2) #CHECKED on homo-1 to homo-2
       
        #This is the hopping rate matrix between energy levels, just a lot of elementwise operations
        #on the matrices previously set up. k_mat = [from state, to state]
        k_mat = g_coup**2*s_mat/hbar*np.sqrt(np.pi/kbT/lam_mat)*np.exp(-((lam_mat+e_mat-f_mat)**2)/(4*kbT*lam_mat))
        #holes flow up field
        #print('----- Compare energy difference and external field -----')
        #print(np.abs(e_mat).mean(),np.abs(e_mat).std())
        #print(np.abs(f_mat).mean(),np.abs(f_mat).std())
        print("Percentage of the nonzero rate constants: %4.1f" %(100*np.nonzero(k_mat)[1].shape[0]/float(n*n)))
        #print(e_mat.min())
        #print(e_mat.max())

    
        rounds = 150000 #max rounds
        threshold_true = 1e-15 #1e-8 #max allowable change per step
        output = 1000
        threshold = threshold_true*output #max allowable change per output
        ns = 5000
        PisFile = '' #pkl file of another run if continuing equilibration
        #starttime = time.time()
        Pis,mobility[t,f],d_pop[t,f],d_mob[t,f] = get_Pis(k_mat[-ns:,-ns:], dopants, threshold, rounds, output,eigvals[-ns:],f_mag, r_mat[-ns:,-ns:],PisFile,f,output_folder)
        mobility[t,f] *= 10**(-16) #Change unit from [angstrom**2/(V*s)] to [cm**2/(V*s)] 
        #print(time.time()-starttime)
        #dim = ['x','y','z']
        #fileo = open('Pis_t{}{}.pkl'.format(t,dim[f]),'wb')
        #pickle.dump(Pis, fileo)
        #fileo.close()
        print(mobility[t,f],flush=True)


f1 = open(output_folder + 'zout_mobility_F{:.2f}_D{}.csv'.format(np.log10(f_mag),dopants),'w')
f1.write('Mob_ave,Mob_sig,mob_x,mob_y,mob_z,sig_x,sig_y,sig_z\n')
f1.write('{},'.format(np.average(mobility)))
f1.write('{},'.format(np.std(mobility)))
for f in range(3):
    f1.write('{},'.format(np.average(mobility[:,f])))
for f in range(2):
    f1.write('{},'.format(np.std(mobility[:,f])))
f1.write('{}\n\n'.format(np.std(mobility[:,f+1])))
f1.close()

f1 = open(output_folder + 'zout_fullmobility_F{:.2f}_D{}.csv'.format(np.log10(f_mag),dopants),'w')
f1.write('TRJ,Mob_ave,Mobx,Moby,Mobz,d_mobx,d_moby,d_mobz,log(d_popx),log(d_popy),log(d_popz)\n')
for t in range(np.shape(mobility)[0]):
    f1.write('{},'.format(t))
    f1.write('{},'.format(np.average(mobility[t])))
    for f in range(np.shape(mobility)[1]):
        f1.write('{},'.format(mobility[t,f]))
    for f in range(np.shape(mobility)[1]):
        f1.write('{},'.format(d_mob[t,f]/mobility[t,f]))
    for f in range(np.shape(mobility)[1]-1):
        f1.write('{},'.format(np.log10(d_pop[t,f])))
    f1.write('{}'.format(np.log10(d_pop[t,f+1])))
    f1.write('\n')
f1.close()

