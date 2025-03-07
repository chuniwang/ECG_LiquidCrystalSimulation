"""
This script evaluates the order parameters with different definations of director axis:
    (1) User specifies the director axis based on a vector of two atoms
    (2) The director axis is determined by the molecular moment of inertia tensor
        (search Wiki or see http://www.kwon3d.com/theory/moi/iten.html or https://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node64.html )
"""

import mdtraj as md
import numpy as np
import numpy.linalg as LA
import sys,re


############ Set file path & parameters ############
morphology = '700K' #'700K' #'555K'

#### NVT 700K ###
if morphology=='700K': 
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_700_8Cells/'
  lammps_datafile = traj_folder + 'CG_NVT_700K_8Cells_Eq.data'
  traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/']
  output_folder = './CG_NVT_700K_8Cells/'

#### NVT 555K ###
if morphology=='555K': 
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_555K_8Cells_IBI66/'
  lammps_datafile = traj_folder + 'CG_NVT_555K_8Cells_Initial.data'
  traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/']
  output_folder = './CG_NVT_555K_8Cells/'

#### NVT 515K ###
if morphology=='515K':
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_515K_8Cells/'
  lammps_datafile = traj_folder + 'CG_NVT_515K_8Cells_Initial.data'
  traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/']
  output_folder = './CG_NVT_515K_8Cells/'


topology_path = 'CG_topology.gro'
no_frame = 10 #Number of snapshot in each trajectory folder for order parameter calculation

director_type = 1 #Defination of the director axis, 0: vector of two atoms; 1: eigenvector of moment of inertia tensor
director_atom = [3,11] #atom index 4 & atom index 12
offset_dis = 0.0 * 0.1 #unit in nm


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



order_parameter = []
order_parameter_max = []
for folder in range(len(traj_path)):
  ############ Read binary trajectory file & basic trajectory information ############
  traj = md.load((traj_folder+traj_path[folder]+'traj.dcd'), top=topology_path)
  total_no_mol = traj.n_residues
  no_atom_per_mol = int(traj.n_atoms/total_no_mol)
  total_frames = traj.n_frames
  print('\nProcessing folder: %s' %traj_path[folder][:-1])
  print("# molecules: %s" %(total_no_mol))
  print("# atoms per molecule: %s" %(no_atom_per_mol))
  print("# total frame: %s" %(total_frames))


  ############ Read atom mass ############
  ########## Read the input *.data file ##########
  f = open(lammps_datafile, "r")
  full_content = f.readlines()
  full_content_numpy = np.array(full_content)
  f.close()
  ########## Mass ##########
  mol_mass = []
  string = "Masses\n"
  check = np.argwhere(full_content_numpy==string).flatten()[-1]
  for i in range(no_atom_per_mol):
    mol_mass.append(float(full_content[check+2+i].split()[1]))
  mol_mass = np.array(mol_mass)


  counter_t = 0
  mol_director = np.zeros((no_frame,total_no_mol,3,3)) #Three director axies obtained from molecular moment of inertia tensor
  for t in range((total_frames-no_frame),total_frames):    
    #print("Analyze the %s-th frame" %(t))
    box_size = traj.unitcell_lengths[t,0]
    #box = np.zeros(3)
    #box[0] = traj.xyz[t,:,0].max()-traj.xyz[t,:,0].min()
    #box[1] = traj.xyz[t,:,1].max()-traj.xyz[t,:,1].min()
    #box[2] = traj.xyz[t,:,2].max()-traj.xyz[t,:,2].min()
    #box_size = box.max()
    ############ Reset the coordinate: begin at (0,0,0) ############
    #shift_length = traj.xyz[t,:,:].min()
    #traj_init = np.reshape(traj.xyz[t,:,:], (total_no_mol,no_atom_per_mol,3)) - shift_length #[molecules,atoms,xyz]
    traj_init = np.reshape(traj.xyz[t,:,:], (total_no_mol,no_atom_per_mol,3)) - offset_dis #[molecules,atoms,xyz]

    ############ Deal with periodic boundary condition: atoms ############
    traj_pbc = [] #[molecules,atoms,xyz]
    for i in range(total_no_mol):
      ref_atom = np.full((no_atom_per_mol,3), traj_init[i,0,:]) #Regard the first atom as the reference
      check = np.absolute(traj_init[i,:,:]-ref_atom)
      traj_pbc.append(np.where((check>0.5*box_size), traj_init[i,:,:]-np.sign((traj_init[i,:,:]-ref_atom))*box_size, traj_init[i,:,:]))
      #debug_check.append(np.where((check>0.5*box_size), 99999, 0))  
    traj_pbc = np.array(traj_pbc)
    #debug_check = np.array(debug_check)

    
    ############ Calculate director axis unit vector ############
    if(director_type==0):
      unit_vec = traj_pbc[:,director_atom[1],:] - traj_pbc[:,director_atom[0],:] #atom index 3 & atom index 11
      unit_vec_length = LA.norm(unit_vec,axis=1)
      unit_vec = unit_vec/(np.full((3,total_no_mol),unit_vec_length).T)
      
    elif(director_type==1):
      unit_vec = np.zeros((total_no_mol,3))
      inertia_tensor = np.zeros((3,3))
      
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
      
      for i in range(total_no_mol):
        ##### Calculate moment of inertia tensor #####
        inertia_tensor[0,0] = (mol_mass*(((traj_pbc[i,:,1]-COM[i,1])**2)+((traj_pbc[i,:,2]-COM[i,2])**2))).sum() #Ixx
        inertia_tensor[1,1] = (mol_mass*(((traj_pbc[i,:,0]-COM[i,0])**2)+((traj_pbc[i,:,2]-COM[i,2])**2))).sum() #Iyy
        inertia_tensor[2,2] = (mol_mass*(((traj_pbc[i,:,0]-COM[i,0])**2)+((traj_pbc[i,:,1]-COM[i,1])**2))).sum() #Izz
        inertia_tensor[0,1] = -(mol_mass*(traj_pbc[i,:,0]-COM[i,0])*(traj_pbc[i,:,1]-COM[i,1])).sum() #Iyx
        inertia_tensor[0,2] = -(mol_mass*(traj_pbc[i,:,0]-COM[i,0])*(traj_pbc[i,:,2]-COM[i,2])).sum() #Izx
        inertia_tensor[1,2] = -(mol_mass*(traj_pbc[i,:,1]-COM[i,1])*(traj_pbc[i,:,2]-COM[i,2])).sum() #Izy
        inertia_tensor[1,0] = inertia_tensor[0,1] #Ixy
        inertia_tensor[2,0] = inertia_tensor[0,2] #Ixz
        inertia_tensor[2,1] = inertia_tensor[1,2] #Iyz
        ##### Diagonize moment of inertia tensor & obtained the director axis #####
        eigen_w, mol_director[counter_t,i,:,:] = np.linalg.eig(inertia_tensor)
        unit_vec[i,:] = mol_director[counter_t,i,:,np.argmin(eigen_w)] 
        #Warning: The normalized eigenvectors, such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    counter_t = counter_t + 1


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
    Q = Q/(2*total_no_mol)
    
    ############ Diagonize order parameter tensor & obtain order parameters ############
    eigen_w, eigen_v = np.linalg.eig(Q)
    order_parameter.append(np.sort(eigen_w))
    output = np.zeros(4)
    output[0] = eigen_w.max()
    if (eigen_v[:,np.argmax(eigen_w)][1] < 0.00):
      output[1:] = -eigen_v[:,np.argmax(eigen_w)]
    else:
      output[1:] = eigen_v[:,np.argmax(eigen_w)]
    order_parameter_max.append(output)
    #order_parameter_max.append(eigen_w.max())
    #print(t,eigen_w.max())
    

order_parameter = np.array(order_parameter)
order_parameter_max = np.array(order_parameter_max)

print("Average of order parameter: %f4" %(order_parameter_max[:,0].mean()))
print("STD of order parameter: %f4" %(order_parameter_max[:,0].std()))
print("Average of principal director: %f4  %f4  %f4"
      %(order_parameter_max[:,1].mean(),
        order_parameter_max[:,2].mean(),
        order_parameter_max[:,3].mean()))
print("STD of principal director: %f4  %f4  %f4"
      %(order_parameter_max[:,1].std(),
        order_parameter_max[:,2].std(),
        order_parameter_max[:,3].std()))

np.savetxt((output_folder + 'Order_parameter.txt'), order_parameter_max)
#print("Average of order parameter: %s" %(order_parameter_max.mean()))
#print("STD of order parameter: %s" %(order_parameter_max.std()))
#print(order_parameter.shape)
#np.savetxt((output_folder + 'Order_parameter.txt'), order_parameter)
