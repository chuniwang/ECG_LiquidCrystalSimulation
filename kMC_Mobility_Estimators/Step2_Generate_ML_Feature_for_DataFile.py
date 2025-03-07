import mdtraj as md
import numpy as np
import numpy.linalg as LA
from scipy.spatial import distance
import sys,re

############ Set file path & parameters ############
#morphology = '700K' #'700K' #'555K'
morphology = sys.argv[1]

#### NVT 700K ###
if morphology=='700K': 
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_700_8Cells/'
  lammps_datafile = traj_folder + 'CG_NVT_700K_8Cells_Eq.data'
  traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/',
               'Trajectory_6/', 'Trajectory_7/', 'Trajectory_8/', './Trajectory_9/', './Trajectory_10/']
  #traj_path = ['Trajectory_1/','Trajectory_2/']
  #traj_path = ['Trajectory_5/']
  output_folder = './CG_NVT_700K_8Cells/ML_Feature/'

#### NVT 555K ###
if morphology=='555K': 
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_555K_8Cells_IBI66/'
  lammps_datafile = traj_folder + 'CG_NVT_555K_8Cells_Initial.data'
  #traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/']
  #traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/',
  #             'Trajectory_6/', 'Trajectory_7/', 'Trajectory_8/', './Trajectory_9/']
  traj_path = ['Trajectory_10/']
  output_folder = './CG_NVT_555K_8Cells/ML_Feature/'

#### NVT 515K ###
if morphology=='515K':
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_515K_8Cells/'
  lammps_datafile = traj_folder + 'CG_NVT_515K_8Cells_Initial.data'
  #traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/']
  #traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/',
  #             'Trajectory_6/', 'Trajectory_7/', 'Trajectory_8/', './Trajectory_9/', './Trajectory_10/']
  traj_path = ['Trajectory_10/']
  output_folder = './CG_NVT_515K_8Cells/ML_Feature/'


topology_path = 'CG_topology.gro'
no_frame = 12 #10 #Number of snapshot in each trajectory folder for order parameter calculation
no_atom_per_mol = 15 #Number of particles in a molecule

snapshot_index = 109 #1 #####Need Check






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


#################### Function: generate ML fearture-distance matrix for HOMO energy prediction ####################
def E_distance_matrix(no_atom_per_mol,mol_xyz):
  CM = np.zeros((no_atom_per_mol, no_atom_per_mol), float)
  ########## Calculate the off-diagonal element of distance matrix ##########
  for i in range(no_atom_per_mol):
    for j in range(i+1, no_atom_per_mol):
      # Calculate pairwise distance
      dst = distance.euclidean(mol_xyz[i], mol_xyz[j])
      CM[i,j] = 1/dst
  ########## Indices for the upper-triangle of distance matrix ##########
  iu_idx = np.triu_indices(no_atom_per_mol, 1)
  return CM[iu_idx]


################ Function: generate ML fearture-distance matrix for coupling prediction ################
def V_distance_matrix(no_atom_per_mol,mol_xyz):
  CM = np.zeros((no_atom_per_mol, no_atom_per_mol), float)
  ########## Calculate the off-diagonal element of distance matrix ##########
  for i in range(no_atom_per_mol):
    for j in range(i+1, no_atom_per_mol):
      # Calculate pairwise distance
      dst = distance.euclidean(mol_xyz[i], mol_xyz[j])
      CM[i,j] = 1/dst
      CM[j,i] = CM[i,j]
  ########## Indices for the upper-triangle of distance matrix ##########
  iu_idx = np.triu_indices(no_atom_per_mol, 1)
  ########## Distance matrix excluding inter-bead elements ##########
  CM_inter = []
  for i in range(int(no_atom_per_mol/2)):
    CM_inter = np.concatenate([CM_inter, CM[i, int(no_atom_per_mol/2):no_atom_per_mol]])
  CM_inter = np.array(CM_inter)
  return CM_inter


#################### Read the input *.data file ####################
f = open(lammps_datafile, "r")
full_content = f.readlines()
full_content_numpy = np.array(full_content)
f.close()

########## Basic information of this system ##########
string = "atoms"
total_n_atoms = extract_value_from_file(string, lammps_datafile)
total_no_mol = int(total_n_atoms/no_atom_per_mol) 
total_frames = no_frame
print("# molecules: %s" %(total_no_mol))
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




for folder in range(len(traj_path)):
  ############ Read binary trajectory file & basic trajectory information ############
  traj = md.load((traj_folder+traj_path[folder]+'traj.dcd'), top=topology_path)
  total_frames = traj.n_frames
  traj_xyz = traj.xyz*10 #Convert unit from nm to angstrom
  print('\nProcessing folder: %s' %traj_path[folder][:-1])
  print("# total frame: %s" %(total_frames))


  for t in range((total_frames-no_frame),total_frames):  
    HOMO_E_feature = []
    Coupling_feature = []
    Coupling_list = []
    
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
    
    ############ Feature for HOMO energy prediction ############
    for i in range(total_no_mol):#####Need Check
      HOMO_E_feature.append(E_distance_matrix(no_atom_per_mol,traj_pbc[i,:,:]))#####Need Check
      
      
    ############ Calculate center of mass (COM) ############
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
    #print(COM.shape)
    
    ############ COM distance ############
    for i in range(total_no_mol):  #####Need Check
      ##### Move the reference molecule to the center of the box #####
      COM_move = COM - np.full((total_no_mol,3), COM[i,:]) + 0.5*box_size
      traj_mol_1 = traj_pbc[i,:,:] - np.full((no_atom_per_mol,3), COM[i,:]) + 0.5*box_size

      ##### Deal with periodic boundary condition #####
      COM_move = np.where((COM_move>box_size), (COM_move-box_size), COM_move)
      COM_move = np.where((COM_move<0.0), (COM_move+box_size), COM_move)

      ##### Calculate the distance between CG beads #####
      vec = COM_move - np.full((total_no_mol,3), COM_move[i,:])
      dis= LA.norm(vec,axis=1)
      
      ############ Feature for coupling prediction ############
      ##### Selecte molecules within cutoff radius for coupling prediction #####
      #dis_list = np.argwhere(((dis<6.0) & (dis!=0))).flatten()
      dis_list = np.argwhere(((dis<7.0) & (dis!=0))).flatten()
      
      for k in range(dis_list.shape[0]):
        ##### Deal with periodic boundary condition for selected molecules #####
        traj_mol_2 = traj_pbc[dis_list[k],:,:] - np.full((no_atom_per_mol,3), COM[i,:]) + 0.5*box_size
        traj_mol_2 = np.where((traj_mol_2>box_size), (traj_mol_2-box_size), traj_mol_2)
        traj_mol_2 = np.where((traj_mol_2<0.0), (traj_mol_2+box_size), traj_mol_2)
        #check_dis = ((traj_mol_2[0,:] - COM_move[dis_list[k],:])**2).sum()**0.5
        #print(check_dis)
        #if (check_dis>(0.5*box_size)):
          #print('There is a bug!')
      
        ##### Feature for coupling prediction of selected pairs #####
        Coupling_list.append([t,i,dis_list[k]])
        pair_xyz = np.concatenate((traj_mol_1,traj_mol_2))
        Coupling_feature.append(V_distance_matrix(2*no_atom_per_mol,pair_xyz))
      #print(i,dis_list,dis_list.shape[0])
      #print(dis[dis_list])
      #print('---')
 
      
    ############ Save files ############
    HOMO_E_feature = np.array(HOMO_E_feature)
    Coupling_list = np.array(Coupling_list)
    Coupling_feature = np.array(Coupling_feature)

    np.save(output_folder+'Feature_MO_Energy_'+ morphology + '_Config_' + str(snapshot_index) +'.npy', HOMO_E_feature)
    np.save(output_folder+'Feature_Coupling_' + morphology + '_Config_' + str(snapshot_index) +'.npy', Coupling_feature)
    np.save(output_folder+'Coupling_List_'+ morphology + '_Config_' + str(snapshot_index) +'.npy', Coupling_list)

    print(Coupling_list.shape)
    print(Coupling_feature.shape)
    print(HOMO_E_feature.shape) 
    print('ML features based on snapshot-%s are generated.' %str(snapshot_index))
    
    snapshot_index = snapshot_index + 1



