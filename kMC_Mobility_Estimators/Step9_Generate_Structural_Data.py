import mdtraj as md
import numpy as np
import freud
import matplotlib.pyplot as plt
import numpy.linalg as LA
from scipy.spatial import distance
from scipy.stats import gaussian_kde
import os,re,sys
import csv


############ Set file path & parameters ############
#morphology = '700K' #'700K' #'555K'
keyword = sys.argv[1] #'700K' #'555K'

traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/',
             'Trajectory_6/', 'Trajectory_7/', 'Trajectory_8/', './Trajectory_9/', './Trajectory_10/']
#traj_path = ['Trajectory_1/','Trajectory_2/']
#traj_path = ['Trajectory_1/']


#### NVT 700K ###
if keyword=='700K':
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_700_8Cells/'
  lammps_datafile = traj_folder + 'CG_NVT_700K_8Cells_Eq.data'
  output_filename = './CG_NVT_700K_8Cells/Structural_Data_' + keyword +'.npy'

#### NVT 555K ###
if keyword=='555K':
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_555K_8Cells_IBI66/'
  lammps_datafile = traj_folder + 'CG_NVT_555K_8Cells_Initial.data'
  output_filename = './CG_NVT_555K_8Cells/Structural_Data_' + keyword +'.npy'

#### NVT 515K ###
if keyword=='515K':
  traj_folder = '/home/ciwang/LAMPPS/CG_Simulation/NVT_515K_8Cells/'
  lammps_datafile = traj_folder + 'CG_NVT_515K_8Cells_Initial.data'
  output_filename = './CG_NVT_515K_8Cells/Structural_Data_' + keyword +'.npy'

topology_path = 'CG_topology.gro'
no_frame = 12 #10 #Number of snapshot in each trajectory folder for order parameter calculation
no_atom_per_mol = 15 #Number of particles in a molecule

cutoff_neighbor_dis = [13.0, 20.0, 50.0]


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


################### Function: Compute nematic order parameter  ###################
def calculate_nematic_order_para(traj_center):
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
    
    return np.sort(Q_eigen_w)[::-1], np.sort(Q_eigen_w_pi)[::-1]
    #return  order_para, order_para_pi


################### Function: Calculate pi-pi stacking strength ###################
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
    
    #print(all_pipi_level.mean(),all_pipi_level.std())
    return all_pipi_level.mean() 


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


############ Calculate density [g/cm3]############
density = (mol_mass.sum()*total_no_mol*10)/(6.02214076*box_size**3)
print("Density: %s [g/cm3]" %density)


all_order_para = []
all_order_para_pi = []
all_number_density = []
all_pi_pi = []
all_q4 = []
all_q6 = []
all_Rg = []
for folder in range(len(traj_path)):
  ############ Read binary trajectory file & basic trajectory information ############
  traj = md.load((traj_folder+traj_path[folder]+'traj.dcd'), top=topology_path)
  total_frames = traj.n_frames
  traj_xyz = traj.xyz*10 #Convert unit from nm to angstrom
  print('\nProcessing folder: %s' %traj_path[folder][:-1])
  print("# of total frame: %s" %(total_frames))

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

    
    ############ Calculate Steinhardt order parameter ############
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
    
    ##### Steinhardt order parameter #####
    ### Define Freud system ###
    freud_box = freud.box.Box.cube(box_size)
    freud_system = (freud_box,(COM-0.5*box_size))
    #print(freud_system)
    
    ### q4 ###
    L=4
    ql = freud.order.Steinhardt(L,average=True)
    #ql_sc = ql.compute(freud_system, neighbors={"num_neighbors": len(charge_neighbor_index)}).particle_order
    ql_sc = ql.compute(freud_system, {'r_max':13}).particle_order 
    all_q4.append(ql_sc)

    ### q6 ###
    L=6
    ql = freud.order.Steinhardt(L,average=True)
    #ql_sc = ql.compute(freud_system, neighbors={"num_neighbors": len(charge_neighbor_index)}).particle_order
    ql_sc = ql.compute(freud_system, {'r_max':13}).particle_order 
    all_q6.append(ql_sc)
    
    ############ Calculate Radius of gyration ############
    Rg = []
    for m in range(total_no_mol):
      R_vec = traj_pbc[m,:,:] - np.full((no_atom_per_mol,3), COM[m,:])
      R_sq = (LA.norm(R_vec,axis=1))**2
      Rg.append(((R_sq*mol_mass).sum()/(mol_mass.sum()))**0.5)
    Rg = np.array(Rg)
    all_Rg.append(Rg)


    ############ Find neighbor molecule around charge carrier center ############
    pi_pi = []
    number_density = []
    for center in range(total_no_mol):  
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
      COM_move = COM - np.full((total_no_mol,3), COM[center,:]) + 0.5*box_size
      ##### Deal with periodic boundary condition if COM outside the box #####
      COM_move = np.where((COM_move>box_size), (COM_move-box_size), COM_move)
      COM_move = np.where((COM_move<0.0), (COM_move+box_size), COM_move)      
      ###### List neighbor molecules ######
      vec = COM_move - np.full((total_no_mol,3), COM_move[center,:])
      dis= LA.norm(vec,axis=1)
      ### Cutoff based on separating distance ###
      ind = np.argwhere(dis[np.argsort(dis)]<cutoff_neighbor_dis[0])
      charge_neighbor_index = np.argsort(dis)[:ind[-1][0]]
      #print('\n')
      #print(folder,t,center)
      #print(charge_neighbor_index)
      #print(np.sort(dis)[:ind[-1][0]])
      if (charge_neighbor_index.size<=2): #Deal with empty list
        charge_neighbor_index = np.argsort(dis)[:10]
        #print('No molecule is in the cutoff radius.')
        print('There are only two molecules within the cutoff radius.')
        #print(charge_neighbor_index)
        #print(charge_neighbor_index,np.sort(dis)[:12]) 
      number_density.append(len(charge_neighbor_index))
        
      ############ Analyze structre correlations ############
      traj_center = traj_pbc[charge_neighbor_index,:,:]
      ##### Deal with periodic boundary condition (PDB): atoms #####
      for j in range(len(charge_neighbor_index)):
        for k in range(3):
          if((COM[charge_neighbor_index[j],k]-COM[charge_neighbor_index[0],k])<-0.5*box_size):
            traj_center[j,:,k] = traj_center[j,:,k] + box_size
          if((COM[charge_neighbor_index[j],k]-COM[charge_neighbor_index[0],k])>0.5*box_size):
            traj_center[j,:,k] = traj_center[j,:,k] - box_size
      ##### Calculate pi-pi stacking strength #####
      pi_pi.append(calculate_pipi(traj_center))
    pi_pi = np.array(pi_pi)
    number_density = np.array(number_density)
    all_pi_pi.append(pi_pi)
    all_number_density.append(number_density)
 
    
    ############ Calculate nematic order parameter with different characteristinc radius ############
    order_para = np.zeros((len(cutoff_neighbor_dis),total_no_mol,3))
    order_para_pi = np.zeros((len(cutoff_neighbor_dis),total_no_mol,3))
    for d_ind, d in enumerate(cutoff_neighbor_dis):
      #print(d_ind, d)
      ######## Find neighbor molecule around charge carrier center ########
      for center in range(total_no_mol):  
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
        COM_move = COM - np.full((total_no_mol,3), COM[center,:]) + 0.5*box_size
        ##### Deal with periodic boundary condition if COM outside the box #####
        COM_move = np.where((COM_move>box_size), (COM_move-box_size), COM_move)
        COM_move = np.where((COM_move<0.0), (COM_move+box_size), COM_move)      
        ###### List neighbor molecules ######
        vec = COM_move - np.full((total_no_mol,3), COM_move[center,:])
        dis= LA.norm(vec,axis=1)
        ### Cutoff based on separating distance ###
        ind = np.argwhere(dis[np.argsort(dis)]<d)
        charge_neighbor_index = np.argsort(dis)[:ind[-1][0]]
        if (charge_neighbor_index.size<=2): #Deal with empty list
          charge_neighbor_index = np.argsort(dis)[:10]
          
        ############ Analyze structre correlations ############
        traj_center = traj_pbc[charge_neighbor_index,:,:]
        ##### Deal with periodic boundary condition (PDB): atoms #####
        for j in range(len(charge_neighbor_index)):
          for k in range(3):
            if((COM[charge_neighbor_index[j],k]-COM[charge_neighbor_index[0],k])<-0.5*box_size):
              traj_center[j,:,k] = traj_center[j,:,k] + box_size
            if((COM[charge_neighbor_index[j],k]-COM[charge_neighbor_index[0],k])>0.5*box_size):
              traj_center[j,:,k] = traj_center[j,:,k] - box_size
        ##### Calculate nematic order parameter #####
        order_para[d_ind,center,:], order_para_pi[d_ind,center,:] = calculate_nematic_order_para(traj_center) 

    all_order_para.append(order_para)
    all_order_para_pi.append(order_para_pi)


############ Save dataset ############
all_order_para = np.array(all_order_para)
all_order_para_pi = np.array(all_order_para_pi)
all_Rg = np.array(all_Rg)
all_q4 = np.array(all_q4)
all_q6 = np.array(all_q6)
all_pi_pi = np.array(all_pi_pi)
all_number_density = np.array(all_number_density)
#print(all_q4.shape)
#print(all_order_para.shape)

config = 1
data = []
for i in range(all_Rg.shape[0]):
  for m in range(total_no_mol):
    data.append([config, all_Rg[i,m], all_number_density[i,m], 
                 all_pi_pi[i,m], all_q4[i,m], all_q6[i,m],
                 all_order_para[i,0,m,0], all_order_para[i,0,m,1], all_order_para[i,0,m,2],  # R_cutoff = 13.0
                 all_order_para[i,1,m,0], all_order_para[i,1,m,1], all_order_para[i,1,m,2],  # R_cutoff = 20.0
                 all_order_para[i,2,m,0], all_order_para[i,2,m,1], all_order_para[i,2,m,2],  # R_cutoff = 50.0
                 all_order_para_pi[i,0,m,0], all_order_para_pi[i,0,m,1], all_order_para_pi[i,0,m,2],  # R_cutoff = 13.0
                 all_order_para_pi[i,1,m,0], all_order_para_pi[i,1,m,1], all_order_para_pi[i,1,m,2],  # R_cutoff = 20.0
                 all_order_para_pi[i,2,m,0], all_order_para_pi[i,2,m,1], all_order_para_pi[i,2,m,2],  # R_cutoff = 50.0
        ])  
  config = config + 1
  
data = np.array(data)
print(data.shape)
np.save(output_filename,data)
#print(data[:3,:])


