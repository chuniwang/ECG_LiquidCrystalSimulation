import mdtraj as md
import numpy as np
import numpy.linalg as LA
from scipy.spatial import distance
import sys,re

############ Set file path & parameters ############
#temperature = '700'
#input_filename = './CG_NVT_' + temperature +'K_8Cells_UnwrapPBC.data'
#input_filename = './Feature_Analysis/Data_File/IBI_555K_Final_Unwrap.data'
#input_filename = './Feature_Analysis/Data_File/CG_555K_8Cells_Compress_Box_Unwrap.data'
#input_filename = './Feature_Analysis/Data_File/CG_555K_to_700K_Final_Unwrap.data'
no_frame = 1 #Number of snapshots for order parameter calculation
no_atom_per_mol = 15 #Number of particles in a molecule




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
class read_data():
  def __init__(self, input_filename, no_atom_per_mol):
    self.no_atom_per_mol = no_atom_per_mol
    ########## Open *.data file ##########  
    f = open(input_filename, "r")
    full_content = f.readlines()
    full_content_numpy = np.array(full_content)
    f.close()

    ########## Basic information of this system ##########
    string = "atoms"
    self.total_n_atoms = extract_value_from_file(string, input_filename)
    self.total_no_mol = int(self.total_n_atoms/self.no_atom_per_mol) 
    self.total_frames = no_frame
    print("# molecules: %s" %(self.total_no_mol))
    print("# particles: %s" %(self.total_n_atoms))
    print("# particles per molecule: %s" %(self.no_atom_per_mol))

    ########## Boxsize ##########
    string = "Masses\n"
    check = np.argwhere(full_content_numpy==string).flatten()[-1]
    box_low = float(full_content[check-2].split()[0])
    box_up = float(full_content[check-2].split()[1])
    self.box_size = box_up - box_low
    print("Box lower boundary: %s" %box_low)
    print("Box upper boundary: %s" %box_up)
    print("Box suze: %s" %self.box_size)

    ########## Mass ##########
    mol_mass = []
    string = "Masses\n"
    check = np.argwhere(full_content_numpy==string).flatten()[-1]
    for i in range(self.no_atom_per_mol):
      mol_mass.append(float(full_content[check+2+i].split()[1]))
    self.mol_mass = np.array(mol_mass)
    #print(mol_mass.shape)

    ########## XYZ ##########
    #self.traj_xyz = np.zeros((self.total_frames,self.total_n_atoms,3))
    #string = "Atoms  # full\n"
    #check = np.argwhere(full_content_numpy==string).flatten()[-1]
    #for i in range(self.total_n_atoms):
      #self.traj_xyz[0,i,0] = float(full_content[check+2+i].split()[4])
      #self.traj_xyz[0,i,1] = float(full_content[check+2+i].split()[5])
      #self.traj_xyz[0,i,2] = float(full_content[check+2+i].split()[6]):q
    #print(traj_xyz[0,:2,:])

  def calculate_COM(self,traj_xyz):
    self.traj_xyz = traj_xyz
    ############ Deal with periodic boundary condition: atoms ############
    traj_pbc = [] #[molecules,atoms,xyz]
    traj_init = np.reshape(self.traj_xyz[:,:], (self.total_no_mol,self.no_atom_per_mol,3)) #[molecules,atoms,xyz]
    for i in range(self.total_no_mol):
      ref_atom = np.full((self.no_atom_per_mol,3), traj_init[i,0,:]) #Regard the first atom as the reference
      check = np.absolute(traj_init[i,:,:]-ref_atom)
      traj_pbc.append(np.where((check>0.5*self.box_size), traj_init[i,:,:]-np.sign((traj_init[i,:,:]-ref_atom))*self.box_size, traj_init[i,:,:]))
      #debug_check.append(np.where((check>0.5*box_size), 99999, 0))  
    self.traj_pbc = np.array(traj_pbc)
    #debug_check = np.array(debug_check)
    
    ############ Calculate center of mass (COM) ############
    ### Create mass matrix ###
    mass_matrix = np.full((self.total_no_mol,self.no_atom_per_mol), self.mol_mass)
    ### Center of mass ###
    self.COM = np.zeros((self.total_no_mol,3))
    mx = mass_matrix*self.traj_pbc[:,:,0]
    my = mass_matrix*self.traj_pbc[:,:,1]
    mz = mass_matrix*self.traj_pbc[:,:,2]
    self.COM[:,0] = np.sum(mx, axis=1)/self.mol_mass.sum()
    self.COM[:,1] = np.sum(my, axis=1)/self.mol_mass.sum()
    self.COM[:,2] = np.sum(mz, axis=1)/self.mol_mass.sum()
    #print(self.COM.shape)

  def calculate_Rij(self, COM, mol_index):
    i = mol_index
    ##### Move the reference molecule to the center of the box #####
    COM_move = COM - np.full((self.total_no_mol,3), COM[i,:]) + 0.5*self.box_size
    #vec = COM_move - np.full((self.total_no_mol,3), COM_move[i,:])
    #dis= LA.norm(vec,axis=1)
    #print(dis.min(),dis.max())
    
    ##### Deal with periodic boundary condition if COM outside the box #####
    COM_move = np.where((COM_move>self.box_size), (COM_move-self.box_size), COM_move)
    COM_move = np.where((COM_move<0.0), (COM_move+self.box_size), COM_move)
    #print(COM_move.shape)
    ##### Calculate the distance between CG beads #####
    vec = COM_move - np.full((self.total_no_mol,3), COM_move[i,:])
    #print(vec[:,0].min(),vec[:,0].max())
    #dis= LA.norm(vec,axis=1)
    #print(dis[:3])
    #print(dis.min(),dis.max())
    
    return vec


"""
counter_t = 0
Coupling_list = []
for t in range(total_frames):
    HOMO_E_feature = []
    Coupling_feature = []
    
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
    for i in range(total_no_mol):
      HOMO_E_feature.append(E_distance_matrix(no_atom_per_mol,traj_pbc[i,:,:]))
      
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
    for i in range(total_no_mol):
      ##### Move the reference molecule to the center of the box #####
      COM_move = COM - np.full((total_no_mol,3), COM[i,:]) + 0.5*box_size

      ##### Deal with periodic boundary condition #####
      COM_move = np.where((COM_move>box_size), (COM_move-box_size), COM_move)
      COM_move = np.where((COM_move<0.0), (COM_move+box_size), COM_move)

      ##### Calculate the distance between CG beads #####
      vec = COM_move - np.full((total_no_mol,3), COM_move[i,:])
      dis= LA.norm(vec,axis=1)
      #dis = np.delete(dis,i) #remove intra molecular bead 
      ############ Feature for coupling prediction ############
      ##### Decide the molecular feature for coupling prediction #####
      #dis_list = np.argwhere(((dis<6.0) & (dis!=0))).flatten()
      dis_list = np.argwhere(((dis<7.0) & (dis!=0))).flatten()
      for k in range(dis_list.shape[0]):
        Coupling_list.append([t,i,dis_list[k]])
        ##### Feature for coupling prediction of selected pairs #####
        pair_xyz = np.concatenate((traj_pbc[i,:,:],traj_pbc[dis_list[k],:,:]))
        Coupling_feature.append(V_distance_matrix(2*no_atom_per_mol,pair_xyz))
      #print(i,dis_list)
      #print(dis[dis_list])
      #print('---')
      
      
############ Save files ############
HOMO_E_feature = np.array(HOMO_E_feature)
Coupling_list = np.array(Coupling_list)
Coupling_feature = np.array(Coupling_feature)

np.save('Feature_MO_Energy_'+ input_filename[29:-5] +'.npy', HOMO_E_feature)
np.save('Feature_Coupling_'+ input_filename[29:-5] +'.npy', Coupling_feature)
np.save('Coupling_List_'+ input_filename[29:-5] +'.npy', Coupling_list)

print(Coupling_list.shape)
print(Coupling_feature.shape)
print(HOMO_E_feature.shape)  
"""


