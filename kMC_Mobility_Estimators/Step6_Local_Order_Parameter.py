import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from scipy.spatial import distance
from scipy.stats import gaussian_kde
import os,re,sys



################# Set file path & parameters #################
keyword_list = ['700K','555K','515K']
traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/']
#traj_path = ['Trajectory_1/','Trajectory_2/']
#traj_path = ['Trajectory_1/']

topology_path = 'CG_topology.gro'

no_frame = 2 #10 #Number of snapshot in each trajectory folder for order parameter calculation
no_atom_per_mol = 15 #Number of particles in a molecule
total_neighbor_mol =20
cutoff_dis = np.arange(10.0, 50., step=2.0)
cutoff_dis = np.concatenate((cutoff_dis,np.array([50,55,60,70,80,90,100],dtype=float)))
dis_sampling = np.full(27,9000)
dis_sampling[-7:] = 500

figurename = 'Local_OP_Cutoff_R.png'
outfilename = 'Local_OP_Cutoff_R.npy'
bins = np.arange(0.0, 1.05, step=0.05)
labels = ['Isotropic', 'Smectic A', 'Smectic E']
colors = ['#548235','#1F4E79','#FFC000']
markers = ['D', 'o', '^']
figure_dpi= 250
#print(bins)

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
"""
figure_data = [] #[morphology, snapshot(config), cutoff, mean/std]
for keyword in keyword_list:
  file_folder = './CG_NVT_' + keyword + '_8Cells/'  # 700K or 555K


  #################### Read the input *.data file ####################
  if (keyword == '700K'):
    traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_700_8Cells/'
    lammps_datafile = traj_folder + 'CG_NVT_700K_8Cells_Eq.data'
  elif (keyword == '555K'):
    traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_555K_8Cells_IBI66/'
    lammps_datafile = traj_folder + 'CG_NVT_555K_8Cells_Initial.data'
  elif (keyword == '515K'):
    traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_515K_8Cells/'
    lammps_datafile = traj_folder + 'CG_NVT_515K_8Cells_Initial.data'
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
  for i in range(no_atom_per_mol):
    mol_mass.append(float(full_content[check+2+i].split()[1]))
  mol_mass = np.array(mol_mass)
  #print(mol_mass.shape)

  all_local_OP_mean_std = []
  for folder in range(len(traj_path)):
    ############ Read binary trajectory file & basic trajectory information ############
    traj = md.load((traj_folder+traj_path[folder]+'traj.dcd'), top=topology_path)
    total_frames = traj.n_frames
    traj_xyz = traj.xyz*10 #Convert unit from nm to angstrom
    print('\nProcessing folder: %s' %traj_path[folder][:-1])
    print("# total frame: %s" %(total_frames))

    for t in range((total_frames-no_frame),total_frames):  
      ############ Deal with periodic boundary condition: atoms ############
      traj_pbc = [] #[molecules,atoms,xyz]
      traj_init = np.reshape(traj_xyz[t,:,:], (total_no_mol,no_atom_per_mol,3)) #[molecules,atoms,xyz]
      for i in range(total_no_mol):
        ref_atom = np.full((no_atom_per_mol,3), traj_init[i,0,:]) #Regard the first atom as the reference
        check = np.absolute(traj_init[i,:,:]-ref_atom)
        traj_pbc.append(np.where((check>0.5*box_size), traj_init[i,:,:]-np.sign((traj_init[i,:,:]-ref_atom))*box_size, traj_init[i,:,:]))
        #debug_check.append(np.where((check>0.5*box_size), 99999, 0))  
      traj_pbc = np.array(traj_pbc)
      #debug_check = np.array(debug_check)

      local_OP_mean_std = []
      for index, cutoff_neighbor_dis in np.ndenumerate(cutoff_dis):
        local_OP = []
        ######################## Move all the molecules in a cluster together  ########################
        for center in range(dis_sampling[index[0]]):          
          ############ Define charge carrier center ############
          charge_center_index = center
      
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
          #print(COM[355,:])
    
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
          #charge_neighbor_index = np.argsort(dis)[:ind[-1][0]].reshape(-1, 1) #Exclude the center molecule
          charge_neighbor_index = np.argsort(dis)[:len(ind)].reshape(-1, 1) #Include the center molecule
          #print(ind[-1])
          #print(np.sort(dis)[ind[-1]])
          #print(np.sort(dis)[:5])
          #print(len(ind))
          #print(center,charge_neighbor_index)
          #print(charge_index)
          #print('----------')


          ############ Process molecules in local enviroment ############
          #####  List the molecular index of local environment ##### 
          charge_index = charge_neighbor_index
          no_charge_mol = len(charge_index)
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
          local_OP.append([Q_eigen_w.max().real])
          #print(Q_eigen_w.max())
        
        local_OP = np.array(local_OP).flatten()
        local_OP_mean_std.append([local_OP.mean(),local_OP.std()])
        #print(local_OP.shape)
      local_OP_mean_std = np.array(local_OP_mean_std)  
      all_local_OP_mean_std.append(local_OP_mean_std)
      print(local_OP_mean_std.shape)
  all_local_OP_mean_std = np.array(all_local_OP_mean_std)
  figure_data.append(all_local_OP_mean_std)


figure_data = np.array(figure_data) #[morphology, snapshot(config), cutoff, mean/std]
np.save(outfilename,figure_data)
print(figure_data.shape)
"""

#figure_data = np.load('./Figures/Local_OP_Cutoff_R.npy')
figure_data = np.load('./Local_OP_Cutoff_R.npy')

"""
for q in range(figure_data.shape[0]):
  hist, bins = np.histogram(figure_data[q,:], bins=bins)
  fig = plt.figure(figsize=(20, 5), dpi=figure_dpi)
  plt.hist(bins[:-1], bins, weights=hist, color="#1F4E79", alpha=0.6)
  plt.show()
  plt.close()  
"""

#################### Plot order parameter as a function of cutoff radius ####################
fig = plt.figure(figsize=(10, 10), dpi=figure_dpi)

########## Calculate mean & std of order parameter #########
op_mean = np.mean(figure_data[:,:,:,0], axis=1)
op_std = np.mean(figure_data[:,:,:,1], axis=1)

for l in range(len(keyword_list)):                
  plt.scatter(cutoff_dis, op_mean[l,:], label=labels[l], color=colors[l], marker=markers[l], s=80)


  ########## Set up error bar #########
  plt.errorbar(
    cutoff_dis,
    op_mean[l,:],
    yerr = op_std[l,:],
    #marker = '.',
    #drawstyle = 'steps-mid',
    color=colors[l],
    alpha=0.5,
    ecolor=colors[l],
    lw=1.5, capsize=4, capthick=1.5 
    #lw=2.5, capsize=4, capthick=2.5
    )

plt.xlim(5,102)
plt.xticks(np.arange(10, 110, step=10), fontsize=28, fontname="Arial")
plt.ylim(0,1)
plt.yticks(np.arange(0, 1.1, step=0.1), fontsize=28, fontname="Arial")

plt.xlabel('Cutoff radius [$\AA$]', fontsize=38)
plt.ylabel('Order parameter', fontsize=38)
plt.tick_params(axis='both', which='major', length=12)  # Adjust the length for major ticks
plt.tick_params(labelsize = 28)

#plt.legend(frameon=False, fontsize=38, loc='upper right')
plt.legend(frameon=False, fontsize=38, loc='center right')

plt.savefig(figurename, dpi=figure_dpi,format="png", bbox_inches="tight")
#plt.show()
plt.close()


