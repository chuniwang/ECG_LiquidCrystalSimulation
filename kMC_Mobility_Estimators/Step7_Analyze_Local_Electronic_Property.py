import mdtraj as md
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg as LA
from scipy.spatial import distance
from scipy.stats import gaussian_kde
import os,re,sys
import csv

################# Set file path & parameters #################
keyword_list = ['700K','555K', '515K']
keyword = '515K'
file_folder = './CG_NVT_' + keyword + '_8Cells/'  # 700K or 555K

constant_onsite_energy = sys.argv[3] #'no' 

#traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/']
traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/',
             'Trajectory_6/', 'Trajectory_7/', 'Trajectory_8/', './Trajectory_9/', './Trajectory_10/']
folder = int(sys.argv[1]) #0 #int(sys.argv[1]) #BEGINS AT 0 AND ENDS AT 9! The index of the trajectory folder


topology_path = 'CG_topology.gro'

no_frame = 12 # DO NOT CHANGE! Number of snapshot in each trajectory folder for order parameter calculation.
#no_frame = 10 # DO NOT CHANGE! Number of snapshot in each trajectory folder for order parameter calculation.
frame_index = int(sys.argv[2]) #0 #int(sys.argv[1]) # BEGINS AT 0 AND ENDS AT 11!  The index among the 12 snapshots
config = 1 + folder*no_frame +frame_index

no_sampling = 20 # # Total number of sampling Hamiltonians by a given snapshot (mimic backmapping process)


no_atom_per_mol = 15 #Number of particles in a molecule
total_neighbor_mol =20
cutoff_neighbor_dis = 13.00

neighbor_criteria = 'cutoff' #'charged' #'cutoff' 

if (constant_onsite_energy == 'yes'):
  IPR_min = [0.0, 1.9, 3, 15]
  IPR_max = [1.1, 2.2, 15, 5000 ]
if (constant_onsite_energy == 'no'):
  IPR_min = [0.0, 1.9, 3]
  IPR_max = [1.1, 2.2, 15]  

#IPR_min = [0.0]
#IPR_max = [1.1]

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


################### Function: Quantify pi-pi stacking  ###################
def calculate_pipi(traj_center):
    include_mol = 15
    traj_center =traj_center[:,:include_mol,:]
    #print(traj_center.shape)
    ########### Calculate COM ########### 
    ### Create mass matrix ###
    mass_matrix = np.full((traj_center.shape[0],traj_center.shape[1]), mol_mass[:include_mol])
    ### Center of mass ###
    COM = np.zeros((traj_center.shape[0],3))
    mx = mass_matrix*traj_center[:,:,0]
    my = mass_matrix*traj_center[:,:,1]
    mz = mass_matrix*traj_center[:,:,2]
    COM[:,0] = np.sum(mx, axis=1)/mol_mass.sum()
    COM[:,1] = np.sum(my, axis=1)/mol_mass.sum()
    COM[:,2] = np.sum(mz, axis=1)/mol_mass.sum()
    #print(COM.shape)
    
    ########### Calculate moment of inertia tensor ###########
    normal_vec = np.zeros((traj_center.shape[0],3))
    unit_vec = np.zeros((traj_center.shape[0],3))
    inertia_tensor = np.zeros((3,3))
    for i in range(traj_center.shape[0]):
      ##### Calculate moment of inertia tensor #####
      inertia_tensor[0,0] = (mol_mass[:include_mol]*(((traj_center[i,:,1]-COM[i,1])**2)+((traj_center[i,:,2]-COM[i,2])**2))).sum() #Ixx
      inertia_tensor[1,1] = (mol_mass[:include_mol]*(((traj_center[i,:,0]-COM[i,0])**2)+((traj_center[i,:,2]-COM[i,2])**2))).sum() #Iyy
      inertia_tensor[2,2] = (mol_mass[:include_mol]*(((traj_center[i,:,0]-COM[i,0])**2)+((traj_center[i,:,1]-COM[i,1])**2))).sum() #Izz
      inertia_tensor[0,1] = -(mol_mass[:include_mol]*(traj_center[i,:,0]-COM[i,0])*(traj_center[i,:,1]-COM[i,1])).sum() #Iyx
      inertia_tensor[0,2] = -(mol_mass[:include_mol]*(traj_center[i,:,0]-COM[i,0])*(traj_center[i,:,2]-COM[i,2])).sum() #Izx
      inertia_tensor[1,2] = -(mol_mass[:include_mol]*(traj_center[i,:,1]-COM[i,1])*(traj_center[i,:,2]-COM[i,2])).sum() #Izy
      inertia_tensor[1,0] = inertia_tensor[0,1] #Ixy
      inertia_tensor[2,0] = inertia_tensor[0,2] #Ixz
      inertia_tensor[2,1] = inertia_tensor[1,2] #Iyz
      ##### Diagonize moment of inertia tensor & obtained the director axis #####
      IT_eigen_w, IT_eigen_v = np.linalg.eig(inertia_tensor)
      ### Select the normal vector of conjugated moiety ###
      normal_vec[i,:] = IT_eigen_v[:,np.argmax(IT_eigen_w)]
      ### Select the primary axis ###
      unit_vec[i,:] = IT_eigen_v[:,np.argmin(IT_eigen_w)]
      #print(np.argmax(IT_eigen_w))
      #Warning: The normalized eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

    ########## Calculate level of pi-pi stacking: Charged center & others ##########
    ##### Charge-center molecule #####
    cross_vec_1 = normal_vec[0,:]
    #cross_vec_1_length = LA.norm(cross_vec_1)
    COM_1 = COM[0,:]
    
    all_pipi_level =[]
    for i in range(1,traj_center.shape[0]):
      ##### Adjacent molecules ######
      cross_vec_2 = normal_vec[i,:]
      #cross_vec_2 = normal_vec[i,:]/LA.norm(cross_vec_2)
      #cross_vec_2_length = LA.norm(cross_vec_2)
      
      ##### Calculate vector between COM #####
      COM_vec = COM[i,:] - COM_1
      COM_dis = LA.norm(COM_vec)
      
      ##### Calculate level of pi-pi stacking #####
      pipi_level = ((np.dot(COM_vec,cross_vec_1))**2) * ((np.dot(COM_vec,cross_vec_2))**2) * ((np.dot(cross_vec_1,cross_vec_2))**2)
      pipi_level = pipi_level * 1 * np.exp(-(COM_dis-2.0))
      all_pipi_level.append(pipi_level)
      #print(pipi_level,COM_dis)
    all_pipi_level = np.array(all_pipi_level)

    
    ########## Calculate level of pi-pi stacking: others except for the charged center ##########
    network_pipi_level =[]    
    ##### Molecule-1 #####
    for i in range(1,traj_center.shape[0]):
      cross_vec_1 = normal_vec[i,:]
      #cross_vec_1_length = LA.norm(cross_vec_1)
      COM_1 = COM[i,:]
    
      ##### Molecule-2 ######
      for j in range(i+1,traj_center.shape[0]):
        cross_vec_2 = normal_vec[j,:]
        #cross_vec_2 = normal_vec[j,:]/LA.norm(cross_vec_2)
        #cross_vec_2_length = LA.norm(cross_vec_2)
        ##### Calculate vector between COM #####
        COM_vec = COM[j,:] - COM_1
        COM_dis = LA.norm(COM_vec)
    
        ##### Calculate level of pi-pi stacking #####
        pipi_level = ((np.dot(COM_vec,cross_vec_1))**2) * ((np.dot(COM_vec,cross_vec_2))**2) * ((np.dot(cross_vec_1,cross_vec_2))**2)
        pipi_level = pipi_level * 1 * np.exp(-(COM_dis-2.0))
        network_pipi_level.append(pipi_level)
        #print(pipi_level,COM_dis)
    network_pipi_level = np.array(network_pipi_level)
    
    ########## Calculate local order parameter: long axis ##########
    ##### Calculate order parameter tensor #####
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
    Q = Q/(2*traj_center.shape[0])

    ##### Diagonize order parameter tensor & obtain order parameters #####
    Q_eigen_w, Q_eigen_v = np.linalg.eig(Q)
    order_para = Q_eigen_w.max()
    #order_parameter.append(np.sort(eigen_w))
    
    ########## Calculate local order parameter: conjugated ring normal vectors ##########
    ##### Calculate order parameter tensor #####
    Q = np.zeros((3,3))
    Q[0,0] = (3*normal_vec[:,0]*normal_vec[:,0]-1).sum() #xx
    Q[0,1] = (3*normal_vec[:,0]*normal_vec[:,1]).sum()   #xy
    Q[0,2] = (3*normal_vec[:,0]*normal_vec[:,2]).sum()   #xz
    Q[1,0] = (3*normal_vec[:,1]*normal_vec[:,0]).sum()   #yx
    Q[1,1] = (3*normal_vec[:,1]*normal_vec[:,1]-1).sum() #yy
    Q[1,2] = (3*normal_vec[:,1]*normal_vec[:,2]).sum()   #yz
    Q[2,0] = (3*normal_vec[:,2]*normal_vec[:,0]).sum()   #zx
    Q[2,1] = (3*normal_vec[:,2]*normal_vec[:,1]).sum()   #zy
    Q[2,2] = (3*normal_vec[:,2]*normal_vec[:,2]-1).sum() #zz
    Q = Q/(2*traj_center.shape[0])

    ##### Diagonize order parameter tensor & obtain order parameters #####
    Q_eigen_w_pi, Q_eigen_v_pi = np.linalg.eig(Q)
    order_para_pi = Q_eigen_w_pi.max()
    #order_parameter.append(np.sort(eigen_w))
    
    #if (all_pipi_level.shape[0]<=5):
    #  return all_pipi_level[:5].mean(), order_para
    #else:
    #  return all_pipi_level.mean(), order_para
    return all_pipi_level.mean(), network_pipi_level.mean(), order_para, order_para_pi
    #return all_pipi_level.max(), order_para


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
for m in range(no_atom_per_mol):
    mol_mass.append(float(full_content[check+2+m].split()[1]))
mol_mass = np.array(mol_mass)
#print(mol_mass.shape)


###################### Read binary trajectory file ######################
traj = md.load((traj_folder+traj_path[folder]+'traj.dcd'), top=topology_path)
total_frames = traj.n_frames
traj_xyz = traj.xyz*10 #Convert unit from nm to angstrom
print('\nProcessing folder: %s' %traj_path[folder][:-1])
print("# total frame: %s" %(total_frames))


t = np.arange((total_frames-no_frame),total_frames)[frame_index]
#print(t)
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






for sampling in range(1,no_sampling+1):
  all_homo_diff_mean = []
  all_adjacent_v_mean = []
  all_network_v_mean = []
  all_order_para_part = []
  all_order_para_pi_part = []
  all_pi_level_part = []
  all_network_pi_level_part = []
  all_IPR_part = []
  all_no_mole = []

  IPR_all = []
  figure_data = []  
  output_data = [] 
  ############ Read Hamiltonian ##########
  ham = np.load((file_folder + 'ECG_Results/Diagonalization/Hamiltonian_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy')) 
    
  ########### Read eigenvectors of singe Hamiltonian ###########
  if (constant_onsite_energy == 'no'):
    eigen_w_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
    eigen_v_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Vector_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
  if (constant_onsite_energy == 'yes'):
    eigen_w_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'_Constant_SiteEnergy.npy'
    eigen_v_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Vector_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'_Constant_SiteEnergy.npy'
  eigen_w = np.load(eigen_w_file)
  eigen_v = np.load(eigen_v_file)
  eigen_w = eigen_w.real
  eigen_v = eigen_v.real
  if (constant_onsite_energy == 'no'):
    print_text = keyword + '_Config_' + str(config) +'-' + str(sampling)  
  if (constant_onsite_energy == 'yes'):
    print_text = keyword + '_Config_' + str(config) +'-' + str(sampling) +'_Constant_SiteEnergy'
  print('\nProcessing: %s' %print_text)
  #print(eigen_w.max(),eigen_w.min())


  ############ Calculate inverse participation ratio (IPR) ##########
  eigen_v_prob = eigen_v**2
  IPR = np.sum(eigen_v_prob[:,:]**2,axis=0)
  IPR = 1/IPR
  IPR_all.append(IPR)
  print("IPR mean: %s" %IPR.mean())
  print("IPR std: %s" %IPR.std())
  print("IPR max: %s" %IPR.max())
  print("IPR min: %s" %IPR.min())

  for j in range(len(IPR_min)):
    ######################## Analyze local electronic properties  ########################
    adjacent_v_mean = []
    network_v_mean = []
    homo_diff_mean = []
    order_para_part = []
    order_para_pi_part = []
    pi_level_part = []
    network_pi_level_part = []
    IPR_part = []
    no_density = []
    no_adjacent_mol = []
    ########## Find charged states with specific IPR values ##########
    print('Analyze charge delocalization with IPR value betwee %s ~ %s' %(IPR_min[j],IPR_max[j]))
    IPR_index = np.where((IPR>=IPR_min[j]) & (IPR<IPR_max[j]))[0]
    #IPR_index = np.where((IPR>=1.99) & (IPR<=2.01))[0]
    #IPR_index = np.where((IPR>=4.99) & (IPR<=8.01))[0]
    #IPR_index = np.where((IPR>=1.00) & (IPR<=1.01))[0]
    #IPR_index = np.where((IPR>=500) & (IPR<=2000))[0]
    print(IPR_index.shape)
    
    
    for ind, charge_state in np.ndenumerate(IPR_index[:]):
      ########## Find charged molecules ##########
      if (neighbor_criteria =='charged'):
        charge_size = np.round(IPR[charge_state]).astype(int)
        charge_index = np.argsort(eigen_v[:,charge_state]**2)[-charge_size:]
      elif (neighbor_criteria =='cutoff'):
        charge_index = np.where(eigen_v[:,charge_state]**2>0.0001)[0]
        charge_size = len(charge_index)
      charge_center_index = charge_index[np.argmax(eigen_v[charge_index,charge_state]**2)]      
      #print(charge_center_index,eigen_v[charge_center_index,charge_state]**2)
      #print('Size: %s' %charge_size)
      #print(eigen_v[charge_index,charge_state]**2)
      
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
      charge_neighbor_index = np.argsort(dis)[:ind[-1][0]]
      #print(charge_center_index)
      #print(charge_neighbor_index)
      #print(np.sort(dis)[:ind[-1][0]])
      no_density.append(len(charge_neighbor_index)) 
      if (charge_neighbor_index.size<=2): #Deal with empty list
        charge_neighbor_index = np.argsort(dis)[:12]
        print('Only two molecules are within the cutoff radius.')
        #print('No molecule is in the cutoff radius.')
        #print(charge_neighbor_index)
        #print(charge_neighbor_index,np.sort(dis)[:12]) 
      
           
      ########## Calcuate HOMO energy difference (absolute values) ##########
      ##### Only consider the charged molecules #####
      if((charge_size != 1) & (neighbor_criteria =='charged') ):
        HOMO_E = ham[charge_index,charge_index]
        HOMO_diff = np.abs(HOMO_E[:,np.newaxis] -HOMO_E)
        #print(HOMO_diff)
        HOMO_diff = HOMO_diff[np.triu_indices(charge_size,k=1)]
        #print(HOMO_diff)
        #print(HOMO_diff.mean(),eigen_w[charge_state])
      ##### Molecules of COM within the cutoff distance #####
      elif(neighbor_criteria =='cutoff' ): 
        charge_index = charge_neighbor_index
        charge_size = len(charge_neighbor_index)
        HOMO_E = ham[charge_index,charge_index]
        HOMO_diff = np.abs(HOMO_E[:,np.newaxis] -HOMO_E)
        #print(HOMO_diff)
        HOMO_diff = HOMO_diff[np.triu_indices(charge_size,k=1)]
        #print(HOMO_diff)
        #print(charge_size,HOMO_diff.mean())
      homo_diff_mean.append(HOMO_diff.mean())  
      
      ########## Extract the couping values of charged molecules ##########
      ##### Coupling of the adjacent molecules #####
      nonzero_index = np.nonzero(ham[charge_index[0],charge_index[:]])[0][1:]
      if (len(nonzero_index)!=0):
        #print(charge_state)
        adjacent_v_mean.append(np.abs(ham[charge_index[0],charge_index[nonzero_index]]).mean())
      elif(len(nonzero_index)==0):
        adjacent_v_mean.append(0.00)
      #print(charge_index[0])
      #print(ham[charge_index[0],charge_index[nonzero_index]])
      
      ##### Coupling of the network molecules #####
      network_v = []
      for m in range(1,charge_size):
        for n in range(m+1,charge_size):
         if (ham[charge_index[m],charge_index[n]]!=0):
           network_v.append(np.abs(ham[charge_index[m],charge_index[n]]))
      if not network_v: #Deal with empty list
        network_v_mean.append(0.00)
      else:
        network_v = np.array(network_v)
        network_v_mean.append(network_v.mean())
      #print(HOMO_diff[np.triu_indices(charge_size,k=1)])
      #print(ham[charge_index[-2:][0],charge_index[-2:][1]])
      
      ############ Analyze structre correlations ############
      traj_center = traj_pbc[charge_index,:,:]
      ##### Deal with periodic boundary condition (PDB): atoms #####
      for j in range(len(charge_neighbor_index)):
        for k in range(3):
          if((COM[charge_neighbor_index[j],k]-COM[charge_neighbor_index[0],k])<-0.5*box_size):
            traj_center[j,:,k] = traj_center[j,:,k] + box_size
          if((COM[charge_neighbor_index[j],k]-COM[charge_neighbor_index[0],k])>0.5*box_size):
            traj_center[j,:,k] = traj_center[j,:,k] - box_size
      ##### Calculate pi-pi level #####
      #pipi_level_mean,pipi_level_std = calculate_pipi(traj_center)
      #figure_data.append([IPR[charge_state].real,pipi_level_mean])
      pipi_level_max, network_pipi_level ,order_para, order_para_pi = calculate_pipi(traj_center)
      figure_data.append([IPR[charge_state].real,pipi_level_max,order_para])
      #print(IPR[charge_state].real,pipi_level_max)
      IPR_part.append(IPR[charge_state].real)
      order_para_part.append(order_para)
      order_para_pi_part.append(order_para_pi)
      pi_level_part.append(pipi_level_max)
      if (np.isnan(network_pipi_level)):
        print('Empty array.')
      else:
        network_pi_level_part.append(network_pipi_level)

    homo_diff_mean = np.array(homo_diff_mean)
    adjacent_v_mean = np.array(adjacent_v_mean)
    network_v_mean = np.array(network_v_mean)
    IPR_part = np.array(IPR_part)
    order_para_part = np.array(order_para_part)
    order_para_pi_part = np.array(order_para_pi_part)
    pi_level_part = np.array(pi_level_part)
    network_pi_level_part = np.array(network_pi_level_part)
    no_density = np.array(no_density)
    print('Averaged number of molecules with in the cutoff %s angstrom: %s %s' %(cutoff_neighbor_dis, no_density.mean(), no_density.std()))
    all_no_mole.append([no_density.mean(), no_density.std()])
    #print(adjacent_v_mean.shape)
    #print(network_v_mean.shape)
    #print(homo_diff_mean.shape)
    
    all_homo_diff_mean.append(homo_diff_mean)
    all_adjacent_v_mean.append(adjacent_v_mean)
    all_network_v_mean.append(network_v_mean)
    all_IPR_part.append(IPR_part)
    all_order_para_part.append(order_para_part)
    all_order_para_pi_part.append(order_para_pi_part)
    all_pi_level_part.append(pi_level_part)
    all_network_pi_level_part.append(network_pi_level_part)
    

  print('\nHOMO energy:')
  for i in range(len(all_homo_diff_mean)):
    print(all_homo_diff_mean[i].mean(),all_homo_diff_mean[i].std())



  #print(all_adjacent_v_mean)
  print('\nAdjacent coupling:')
  for i in range(len(all_adjacent_v_mean)):
    print(all_adjacent_v_mean[i].mean(),all_adjacent_v_mean[i].std())

  print('\nNetwork coupling:')
  for i in range(len(all_network_v_mean)):
    print(all_network_v_mean[i].mean(),all_network_v_mean[i].std())

  print('\nOrder parameter-Primary axis:')
  for i in range(len(all_order_para_part)):
    print(all_order_para_part[i].mean(),all_order_para_part[i].std())

  print('\nOrder parameter-Pi axis:')
  for i in range(len(all_order_para_pi_part)):
    print(all_order_para_pi_part[i].mean(),all_order_para_pi_part[i].std())

  print('\npi-pi level:')
  for i in range(len(all_pi_level_part)):
    print(all_pi_level_part[i].mean(),all_pi_level_part[i].std())  

  print('\nNetwork pi-pi level:')
  for i in range(len(all_network_pi_level_part)):
    print(all_network_pi_level_part[i].mean(),all_network_pi_level_part[i].std()) 
    
  print('\nIPR:')
  for i in range(len(all_IPR_part)):
    print(all_IPR_part[i].mean(),all_IPR_part[i].std())   
    
  print('\n# molecules with in the cutoff:')
  for i in range(len(all_IPR_part)):
    print(all_no_mole[i][0],all_no_mole[i][1]) 

  figure_data = np.array(figure_data)
  #print(figure_data.shape)
  
  ########################Save results  ########################
  if (constant_onsite_energy == 'no'):
    csv_file = file_folder + '/Local_Electronic/Local_Electronic_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.csv'
  if (constant_onsite_energy == 'yes'):
    csv_file = file_folder + '/Local_Electronic/Local_Electronic_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'_ConstantSite.csv'
  for i in range(len(IPR_max)):
    output_data.append([all_homo_diff_mean[i].mean(),all_homo_diff_mean[i].std(),
                      all_adjacent_v_mean[i].mean(),all_adjacent_v_mean[i].std(),
                      all_network_v_mean[i].mean(),all_network_v_mean[i].std(),
                      all_order_para_part[i].mean(),all_order_para_part[i].std(),
                      all_pi_level_part[i].mean(),all_pi_level_part[i].std(),
                      all_network_pi_level_part[i].mean(),all_network_pi_level_part[i].std(),
                      all_no_mole[i][0],all_no_mole[i][1],
                      all_order_para_pi_part[i].mean(),all_order_para_pi_part[i].std(),
                      all_IPR_part[i].mean(),all_IPR_part[i].std()
                      ])


  ##### Open the CSV file in write mode #####
  with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    
    ##### Write the header row (optional) #####
    writer.writerow(['HOMO-diff-mean', 'HOMO-diff-std', 
                     'Adjacent-V-mean', 'Adjacent-V-std',
                     'Network-V-mean', 'Network-V-std',
                     'OP_primary-mean', 'OP_primary-std',
                     'pi-pi-mean', 'pi-pi-std',
                     'Network-pi-pi-mean', 'Network-pi-pi-std',
                     'no_mol-mean', 'no_mol-std',
                     'OP_pi-mean', 'OP_pi-std',
                     'IPR-mean', 'IPR-std'
                     ])

    ##### Write the data rows #####
    for row in output_data:
        writer.writerow(row)   


