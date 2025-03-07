import mdtraj as md
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg as LA
from scipy.spatial import distance
from scipy.stats import gaussian_kde
import os,re,sys


################# Set file path & parameters #################
keyword_list = ['700K','555K']

traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/']
#traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/']
#traj_path = ['Trajectory_1/','Trajectory_2/']
traj_path = ['Trajectory_1/']

constant_onsite_energy = 'no' #sys.argv[1] #'no'
H_type = 'Gaussian_sampling' #sys.argv[2] #'Gaussian_sampling' 'ECG_mean'


topology_path = 'CG_topology.gro'

no_frame = 10 #10 #Number of snapshot in each trajectory folder for order parameter calculation
no_sampling = 10 #10 # Total number of sampling Hamiltonians by a given snapshot (mimic backmapping process)


no_atom_per_mol = 15 #Number of particles in a molecule
total_neighbor_mol =20
cutoff_neighbor_dis = 13.00

figurename_scatter = []
for keyword in keyword_list:
  if (constant_onsite_energy == 'no'):
    if (H_type == 'ECG_mean'):
      figurename_scatter.append(('Scatter_OP_Eigenvalue_' + keyword + '_H_ECG_mean.png'))
    if (H_type == 'Gaussian_sampling'):
      figurename_scatter.append(('Scatter_OP_Eigenvalue_' + keyword + '_H_Sampling.png')) 
  if (constant_onsite_energy == 'yes'):
    if (H_type == 'ECG_mean'):
      figurename_scatter.append(('Scatter_OP_Eigenvalue_' + keyword + '_H_ECG_mean_Constant_SiteEnergy.png')) 
    if (H_type == 'Gaussian_sampling'):
      figurename_scatter.append(('Scatter_OP_Eigenvalue_' + keyword + '_H_Sampling_Constant_SiteEnergy.png')) 



######################## Function: Search Keyword and Report Required Variables  ########################
def extract_value_from_file(keyword, filename):
    #print(keyword)
    with open(filename, "r") as f:
        for line in f:
            match = re.search(r"(\d+) {}\b".format(keyword), line)
            if match:
                return int(match.group(1))
    # If the keyword is not found in the file, return None or raise an error
    return None



total_IPR =[] # [morphology, snapshot(config), backmapping(sampling), states]
total_onsite_E = [] #[morphology, (no_config*no_sampling), states]
total_figure_data = [] #[morphology, (no_config*no_sampling), OP/IPR/no_mol]
for keyword in keyword_list:
  file_folder = './CG_NVT_' + keyword + '_8Cells/'  # 700K or 555K
  IPR_config = []
  onsite_E = []
  figure_data = []


  #################### Read the input *.data file ####################
  if (keyword == '700K'):
    traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_700_8Cells/'
    lammps_datafile = traj_folder + 'CG_NVT_700K_8Cells_Eq.data'
  elif (keyword == '555K'):
    traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_555K_8Cells_IBI66/'
    lammps_datafile = traj_folder + 'CG_NVT_555K_8Cells_Initial.data'
  f = open(lammps_datafile, "r")
  full_content = f.readlines()
  full_content_numpy = np.array(full_content)
  f.close()

  ########## Basic information of this system ##########
  string = "atoms"
  total_n_atoms = extract_value_from_file(string, lammps_datafile)
  total_no_mol = int(total_n_atoms/no_atom_per_mol) 
  total_frames = no_frame
  print("\n# molecules: %s" %(total_no_mol))
  print("# atoms: %s" %(total_n_atoms))
  print("# atoms per molecule: %s" %(no_atom_per_mol))

  ########## Boxsize ##########
  string = "Masses\n"
  check = np.argwhere(full_content_numpy==string).flatten()[-1]
  box_low = float(full_content[check-2].split()[0])
  box_up = float(full_content[check-2].split()[1])
  box_size = box_up - box_low
  print("Box lower boundary: %s" %box_low)
  print("Box upper boundary: %s" %box_up)
  print("Box suze: %s" %box_size)

  ########## Mass ##########
  mol_mass = []
  string = "Masses\n"
  check = np.argwhere(full_content_numpy==string).flatten()[-1]
  for m in range(no_atom_per_mol):
    mol_mass.append(float(full_content[check+2+m].split()[1]))
  mol_mass = np.array(mol_mass)
  #print(mol_mass.shape)

  config = 1
  for folder in range(len(traj_path)):
    ###################### Read binary trajectory file ######################
    traj = md.load((traj_folder+traj_path[folder]+'traj.dcd'), top=topology_path)
    total_frames = traj.n_frames
    traj_xyz = traj.xyz*10 #Convert unit from nm to angstrom
    print('\nProcessing folder: %s' %traj_path[folder][:-1])
    print("# total frame: %s" %(total_frames))

    for t in range((total_frames-no_frame),total_frames):  
      ############ Deal with periodic boundary condition: atoms ############
      traj_pbc = [] #[molecules,atoms,xyz]
      traj_init = np.reshape(traj_xyz[t,:,:], (total_no_mol,no_atom_per_mol,3)) #[molecules,atoms,xyz]
      for n in range(total_no_mol):
        ref_atom = np.full((no_atom_per_mol,3), traj_init[n,0,:]) #Regard the first atom as the reference
        check = np.absolute(traj_init[n,:,:]-ref_atom)
        traj_pbc.append(np.where((check>0.5*box_size), traj_init[n,:,:]-np.sign((traj_init[n,:,:]-ref_atom))*box_size, traj_init[n,:,:]))
        #debug_check.append(np.where((check>0.5*box_size), 99999, 0))  
      traj_pbc = np.array(traj_pbc)
      #debug_check = np.array(debug_check)


      ###################### Calculate IPR (Charge delocalization) ######################
      if (H_type == 'ECG_mean'):
        sampling_start = 0
        sampling_end = 1
      if (H_type == 'Gaussian_sampling'):
        sampling_start = 1
        sampling_end = no_sampling+1
      for sampling in range(sampling_start,sampling_end):    
        ########### Read eigenvectors of singe Hamiltonian ###########
        if (constant_onsite_energy == 'no'):
          eigen_w_file = file_folder + '/ECG_Results_Config50/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
          eigen_v_file = file_folder + '/ECG_Results_Config50/Diagonalization/Eigen_Vector_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
        if (constant_onsite_energy == 'yes'):
          eigen_w_file = file_folder + '/ECG_Results_Config50/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'_Constant_SiteEnergy.npy'
          eigen_v_file = file_folder + '/ECG_Results_Config50/Diagonalization/Eigen_Vector_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'_Constant_SiteEnergy.npy'
        eigen_w = np.load(eigen_w_file)
        eigen_v = np.load(eigen_v_file)
        eigen_w = eigen_w.real
        eigen_v = eigen_v.real
        onsite_E.append(eigen_w)
        #print(eigen_w.max(),eigen_w.min(),eigen_w.mean())

        ############ Calculate inverse participation ratio ##########
        eigen_v_prob = eigen_v**2
        IPR = np.sum(eigen_v_prob[:,:]**2,axis=0)
        IPR = 1/IPR
        #IPR_all.append(IPR)
        print(eigen_v_file)
        print("IPR mean: %s" %IPR.mean())
        print("IPR std: %s" %IPR.std())
        print("IPR max: %s" %IPR.max())
        print("IPR min: %s" %IPR.min())

        
        ######################## Move all the molecules in a cluster together  ########################
        for center in range(total_no_mol):    
          ############ Find charge carrier molecules ############
          charge_index = np.argwhere(eigen_v[:,center]**2>0.0001)
          no_charge_mol = len(charge_index)
          traj_center = np.squeeze(traj_pbc[charge_index,:,:],axis=1)
          #print(len(charge_index))
      
          ############ Find charge carrier center ############
          charge_center_index = charge_index[np.argmax(eigen_v[charge_index,center]**2)]
          #print(charge_center_index)
          #print(len(charge_index),charge_index)
          #print(np.argmax(eigen_v[charge_index,center]**2))
          #print(eigen_v[charge_index,center]**2)
      
          ############ Find neighbor molecule around charge carrier center ############
          ###### Calculate center of mass (COM) ######
          ### Create mass matrix ###
          mass_matrix = np.full((total_no_mol,no_atom_per_mol), mol_mass)
          ### Center of mass ###
          COM = np.zeros((total_no_mol,3))
          mx = mass_matrix*traj_pbc[:,:,0]
          my = mass_matrix*traj_pbc[:,:,1]
          mz = mass_matrix*traj_pbc[:,:,2]
          COM[:,0] = np.sum(mx, axis=1)/mol_mass.sum()
          COM[:,1] = np.sum(my, axis=1)/mol_mass.sum()
          COM[:,2] = np.sum(mz, axis=1)/mol_mass.sum()
          #print(COM[charge_center_index,:])
    
          ###### COM distance ######
          ##### Move the reference molecule to the center of the box #####
          COM_move = COM - np.full((total_no_mol,3), COM[charge_center_index,:]) + 0.5*box_size
          ##### Deal with periodic boundary condition if COM outside the box #####
          COM_move = np.where((COM_move>box_size), (COM_move-box_size), COM_move)
          COM_move = np.where((COM_move<0.0), (COM_move+box_size), COM_move)      
      
          ###### List neighbor molecules ######
          vec = COM_move - np.full((total_no_mol,3), COM_move[charge_center_index,:])
          dis= LA.norm(vec,axis=1)
          ### Cut-off based on number of molecules ###
          #charge_neighbor_index = np.argsort(dis)[:total_neighbor_mol].reshape(-1, 1)
          ### Cutoff based on separating distance ###
          ind = np.argwhere(dis[np.argsort(dis)]<cutoff_neighbor_dis)
          charge_neighbor_index = np.argsort(dis)[:ind[-1][0]].reshape(-1, 1) #Exclude the center molecule
          if (charge_neighbor_index.size==0): #Deal with empty list
            charge_neighbor_index = np.argsort(dis)[:15].reshape(-1, 1)
            print('No molecule is in the cutoff radius.')
            print(charge_neighbor_index)
            print(center,dis[np.argsort(dis)][:2],dis[np.argsort(dis)[1]])
          #print(ind)
          #print(np.sort(dis)[ind[-1]])
          #print(charge_center_index,charge_neighbor_index)
          #print(charge_index)
          #print('----------')
          
          
          ##### Find charge carrier form an aggregate #####
          if (len(charge_index)<15):
            #####  List the molecular index of local environment ##### 
            #print(charge_index)
            #print(charge_neighbor_index)
            charge_index = charge_neighbor_index
            no_charge_mol = len(charge_index)
            #print(no_charge_mol)
            traj_center = np.squeeze(traj_pbc[charge_index,:,:],axis=1)
            #####  Calculate center of mass (COM) ##### 
            ### Create mass matrix ###
            mass_matrix = np.full((no_charge_mol,no_atom_per_mol), mol_mass)
            ### Center of mass ###
            COM = np.zeros((no_charge_mol,3))
            mx = mass_matrix*np.squeeze(traj_pbc[charge_index,:,0],axis=1)
            my = mass_matrix*np.squeeze(traj_pbc[charge_index,:,1],axis=1)
            mz = mass_matrix*np.squeeze(traj_pbc[charge_index,:,2],axis=1)
            #print(np.squeeze(traj_pbc[charge_index,:,0],axis=1).shape)
            COM[:,0] = np.sum(mx, axis=1)/mol_mass.sum()
            COM[:,1] = np.sum(my, axis=1)/mol_mass.sum()
            COM[:,2] = np.sum(mz, axis=1)/mol_mass.sum()
            #print(COM)
            #print('-----')
            #traj_center = np.squeeze(traj_pbc[charge_index,:,:],axis=1) 
        
            ##### Deal with periodic boundary condition (PDB): atoms #####
            for j in range(no_charge_mol):
              for k in range(3):
                if((COM[j,k]-COM[0,k])<-0.5*box_size):
                  traj_center[j,:,k] = traj_center[j,:,k] + box_size
                if((COM[j,k]-COM[0,k])>0.5*box_size):
                  traj_center[j,:,k] = traj_center[j,:,k] - box_size
        


            ######################## Estimate local orderparameter for molecular cluster  ########################
            #####  Calculate center of mass (COM) after PDB process ##### 
            COM = np.zeros((no_charge_mol,3))
            mx = mass_matrix*traj_center[:,:,0]
            my = mass_matrix*traj_center[:,:,1]
            mz = mass_matrix*traj_center[:,:,2]
            #print(np.squeeze(traj_pbc[charge_index,:,0],axis=1).shape)
            COM[:,0] = np.sum(mx, axis=1)/mol_mass.sum()
            COM[:,1] = np.sum(my, axis=1)/mol_mass.sum()
            COM[:,2] = np.sum(mz, axis=1)/mol_mass.sum()
            #print(COM)
            #print('=========')  
        
            inertia_tensor = np.zeros((3,3))
            unit_vec = np.zeros((no_charge_mol,3))
            for i in range(no_charge_mol):
              ##### Calculate moment of inertia tensor #####
              inertia_tensor[0,0] = (mol_mass*(((traj_center[i,:,1]-COM[i,1])**2)+((traj_center[i,:,2]-COM[i,2])**2))).sum() #Ixx
              inertia_tensor[1,1] = (mol_mass*(((traj_center[i,:,0]-COM[i,0])**2)+((traj_center[i,:,2]-COM[i,2])**2))).sum() #Iyy
              inertia_tensor[2,2] = (mol_mass*(((traj_center[i,:,0]-COM[i,0])**2)+((traj_center[i,:,1]-COM[i,1])**2))).sum() #Izz
              inertia_tensor[0,1] = -(mol_mass*(traj_center[i,:,0]-COM[i,0])*(traj_center[i,:,1]-COM[i,1])).sum() #Iyx
              inertia_tensor[0,2] = -(mol_mass*(traj_center[i,:,0]-COM[i,0])*(traj_center[i,:,2]-COM[i,2])).sum() #Izx
              inertia_tensor[1,2] = -(mol_mass*(traj_center[i,:,1]-COM[i,1])*(traj_center[i,:,2]-COM[i,2])).sum() #Izy
              inertia_tensor[1,0] = inertia_tensor[0,1] #Ixy
              inertia_tensor[2,0] = inertia_tensor[0,2] #Ixz
              inertia_tensor[2,1] = inertia_tensor[1,2] #Iyz
              ##### Diagonize moment of inertia tensor & obtained the director axis #####
              IT_eigen_w, IT_eigen_v = np.linalg.eig(inertia_tensor)
              unit_vec[i,:] = IT_eigen_v[:,np.argmin(IT_eigen_w)] 
              #print(unit_vec[i,:])
              #Warning: The normalized eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].


            ############ Calculate order parameter tensor ############
            Q = np.zeros((3,3))
            Q[0,0] = (3*unit_vec[:,0]*unit_vec[:,0]-1).sum() #xx
            Q[0,1] = (3*unit_vec[:,0]*unit_vec[:,1]).sum()   #xy
            Q[0,2] = (3*unit_vec[:,0]*unit_vec[:,2]).sum()   #xz
            Q[1,0] = (3*unit_vec[:,1]*unit_vec[:,0]).sum()   #yx
            Q[1,1] = (3*unit_vec[:,1]*unit_vec[:,1]-1).sum() #yy
            Q[1,2] = (3*unit_vec[:,1]*unit_vec[:,2]).sum()   #yz
            Q[2,0] = (3*unit_vec[:,2]*unit_vec[:,0]).sum()   #zx
            Q[2,1] = (3*unit_vec[:,2]*unit_vec[:,1]).sum()   #zy
            Q[2,2] = (3*unit_vec[:,2]*unit_vec[:,2]-1).sum() #zz
            Q = Q/(2*no_charge_mol)
        
            ############ Diagonize order parameter tensor & obtain order parameters ############
            Q_eigen_w, Q_eigen_v = np.linalg.eig(Q)
            #order_parameter.append(np.sort(eigen_w))
            figure_data.append([Q_eigen_w.max().real,eigen_w[center],no_charge_mol])
            #print(no_charge_mol,Q_eigen_w.max())
          
      config = config+1 

  figure_data = np.array(figure_data)
  total_figure_data.append(figure_data)
  print(figure_data.shape)
  #print(figure_data[:,2].max())

total_figure_data=np.array(total_figure_data)
print(total_figure_data.shape)

for l in range(len(keyword_list)):
  plt.figure(figsize = (10, 10), dpi=250)
  ##### Calculate the point density #####
  x, y = total_figure_data[l][:,0], total_figure_data[l][:,1]
  xy = np.vstack([x,y])
  z = gaussian_kde(xy)(xy) 
    
  ##### Sort the points by density, so the densest points are plotted last #####
  idx = z.argsort()
  x, y, z = x[idx], y[idx], z[idx]
  plt.scatter(x, y, c=z, s=25, cmap='viridis')   #edgecolor=''
  #plt.scatter(figure_data[:,1], figure_data[:,0])   #edgecolor=''

  plt.xlabel('Order parameter', fontsize=36)
  plt.xticks(np.arange(0.0, 1.1, step=0.1), fontsize=28, fontname="Arial", rotation=45) #rotation=90

  plt.ylabel('CT state energy [eV]', fontsize=36)
  plt.ylim(-8.2, -6.4)
  plt.yticks(np.arange(-8.2, -6.3, step=0.2), fontsize=28, fontname="Arial")
  plt.tick_params(axis='both', which='major', length=12)  # Adjust the length for major ticks
  plt.savefig(figurename_scatter[l], format="png", bbox_inches="tight")
  #plt.show()
  plt.close()



