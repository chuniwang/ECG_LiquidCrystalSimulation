import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import sys,re,os
import time
import mdtraj as md
#np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
#np.set_printoptions(precision=4)

file_keyword = sys.argv[1] #'555K'
#config_keyword = '5-1'
config_keyword = sys.argv[3]


#time_end = 1000000
#time_end = 100000
time_end = int(sys.argv[2])

kmc_folder = './CG_NVT_' + file_keyword + '_8Cells/KMC_Results/' #+ \
#kmc_folder = './'

if (file_keyword=='515K'):
  md_folder = '../NVT_' + file_keyword + '_8Cells/Trajectory_9/'
elif (file_keyword=='555K'):
  md_folder = '../NVT_' + file_keyword + '_8Cells_IBI66/Trajectory_9/'  
elif (file_keyword=='700K'):
  md_folder = '../NVT_' + file_keyword + '_8Cells/Trajectory_9/'  
#md_folder = '../../NVT_' + file_keyword + '_8Cells/Trajectory_1/'
lammps_datafile = md_folder + 'CG_NVT_' + file_keyword + '_8Cells_Eq_20ns.data'


output_path = './CG_NVT_' + file_keyword + '_8Cells/KMC_Results/' + \
                'Charge_Mobility_' +file_keyword + '_Config_' + config_keyword + \
                '_Steps' + str(time_end) + '.txt'

topology_path = 'CG_topology.gro'

no_frame = 12 #Number of snapshot in each trajectory folder for order parameter calculation
no_atom_per_mol = 15 #Number of particles in a molecule
no_kmc_step = 5000000


###### Parameter related to kij matrix ######
lam_coup = 0.4301307255 #eV
#kbT = 0.0478225 #eV, 555K
kbT = 0.02585 #eV, 300K
#hbar = 6.58212e-16 #eV*s
hbar = 6.58212e-4 #eV*ps
no_atom_per_mol = 15 #Number of CG particles in a molecule


########## Function: Generate snapshoot in *.gro format  ##########
def list_files():
  ##### Folder path #####
  folder_path = kmc_folder

  ##### Filter keyword #####
  keyword = 'Hopping_Traj_' +file_keyword + '_Config_' + config_keyword

  ##### List files in the folder #####
  files = os.listdir(folder_path)

  ##### Filter files based on the keyword #####
  filtered_files = [file for file in files if keyword in file]
  
  return filtered_files

    
########## Function: Generate snapshoot in *.gro format  ##########
def generate_snapshot(xyz):
  snapshoot_gro = open('Snapshot.gro', 'w')
  snapshoot_gro.write("Check file\n" )
  snapshoot_gro.write("%s\n" %(xyz.shape[0]))
  counter = 1
  molecule_name = 'COM'
  print_atom_symbol = 'Cl'
  for m in range(xyz.shape[0]):
      snapshoot_gro.write('%5i%-5s%4s%6i%8.3f%8.3f%8.3f \n' %((m+1), molecule_name, print_atom_symbol, counter,\
                                                              xyz[m,0], xyz[m,1], xyz[m,2]))
      counter = counter + 1
  snapshoot_gro.write('%10.5f%10.5f%10.5f \n' %(box_size, box_size, box_size)) 


########## Function: Linear regression  ##########
def linear_func(x, pars):
    #a,b = pars
    a = pars
    y = a*x #+ b
    return y

def obj_function(pars):
    return ((y_true-linear_func(x, pars))**2).sum()



def identify_initial_site(kmc_file):
  initial_site = int(kmc_file.split('_')[-1][:-4])
  print('Initial site index: %s' %initial_site)
  return initial_site


def plot_scatter(x,y,coefficients): 
  ########## Plot figures ##########
  #filename = 'Compare_CG_Potential_' + str(search_index[0]) + '_' + str(search_index[1]) + '.png'
  #CG_text = 'CG-' + str(search_index[0]) + ' to ' + 'CG-' + str(search_index[1])
  #plt.figure(figsize = (20, 8), dpi=250)
  #plt.figure(figsize = (6, 6), dpi=250)
  #plt.xlim([0,20])
  #plt.ylim([-1,1])
  
  fit_y = coefficients[0]*x #+ coefficients[1] 

  plt.plot(x, y)
  plt.plot(x,fit_y)
  #plt.plot(phase_1[:,0], phase_1[:,1], color=colors[0], label=label_name[0])
  #plt.plot(phase_2[:,0], phase_2[:,1], color=colors[1], label=label_name[1])

    
  plt.xlabel('Time [ns]', fontsize=20, fontname="Arial")
  plt.ylabel('MSD [$\AA^{2}$]', fontsize=20, fontname="Arial")
  #plt.xticks(np.arange(0, 21, step=2), fontsize=28, fontname="Arial")
  #plt.yticks(fontsize=28, fontname="Arial")
  #plt.text(10, -0.75, CG_text, dict(size=28))
  #plt.xticks(np.arange(0, 21, step=2), fontsize=24, fontname="Arial")
  #plt.yticks(fontsize=24, fontname="Arial")
  #plt.tick_params(axis='both', which='major', length=12)  # Adjust the length for major ticks
  #plt.legend(fontsize=28,frameon=False)
  #filename = 'Compare_RDFs_Format.png'
  plt.savefig(('Test.png'),format="png", bbox_inches="tight")
  plt.show()
  plt.close()  


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


def read_md_info(lammps_datafile):
  #################### Read the input *.data file ####################
  f = open(lammps_datafile, "r")
  full_content = f.readlines()
  full_content_numpy = np.array(full_content)
  f.close()
  
  ########## Basic information of this system ##########
  string = "atoms"
  total_n_atoms = extract_value_from_file(string, lammps_datafile)
  total_no_mol = int(total_n_atoms/no_atom_per_mol) 
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
  print("Box size: %s" %box_size)
  
  ########## Mass ##########
  mol_mass = []
  string = "Masses\n"
  check = np.argwhere(full_content_numpy==string).flatten()[-1]
  for i in range(no_atom_per_mol):
    mol_mass.append(float(full_content[check+2+i].split()[1]))
  mol_mass = np.array(mol_mass)
  
  return box_size, mol_mass, total_no_mol



############ Read binary trajectory file & basic trajectory information ############
###### Read MD basic information ######
lammps_datafile = md_folder + 'CG_NVT_' + file_keyword + '_8Cells_Eq_20ns.data'
box_size, mol_mass, total_no_mol = read_md_info(lammps_datafile)

###### Read MD configuration ######
traj_path = md_folder + 'traj.dcd' 
traj = md.load(traj_path, top=topology_path)
traj_xyz = traj.xyz*10 #Convert unit from nm to angstrom
total_frames = traj.n_frames

if ((int(config_keyword[:-2])%12)==0):
  t = 12 + 7
else:
  t = (int(config_keyword[:-2])%12) + 7
print((int(config_keyword[:-2])%12),t)
#t = int(config_keyword[:-2]) + 7

############ Deal with periodic boundary condition: atoms ############
traj_pbc = [] #[molecules,atoms,xyz]
traj_init = np.reshape(traj_xyz[t,:,:], (total_no_mol,no_atom_per_mol,3)) #[molecules,atoms,xyz]
for i in range(total_no_mol):
  ref_atom = np.full((no_atom_per_mol,3), traj_init[i,0,:]) #Regard the first atom as the reference
  check = np.absolute(traj_init[i,:,:]-ref_atom)
  traj_pbc.append(np.where((check>0.5*box_size), traj_init[i,:,:]-np.sign((traj_init[i,:,:]-ref_atom))*box_size, traj_init[i,:,:]))
  #debug_check.append(np.where((check>0.5*box_size), 99999, 0))  
traj_pbc = np.array(traj_pbc)


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

### Deal with periodic boundary condition: COM ###
COM = np.where((COM>box_size), (COM-box_size), COM)
COM = np.where((COM<0), (COM+box_size), COM)


##### Using each site as the coordainte origin #####
pbc_dis = [] # [origin-particle, particle, xyz] asymmetric matrix
for m in range(total_no_mol):
  ### Move the reference molecule to the center of the box ###
  PBC_move = COM - np.full((total_no_mol,3), COM[m,:]) + 0.5*box_size
  
  ### Deal with periodic boundary condition ###
  PBC_move_move = np.where((PBC_move>box_size), (PBC_move-box_size), PBC_move)
  PBC_move = np.where((PBC_move<0.0), (PBC_move+box_size), PBC_move)
  
  ### Calculate distance ###
  pbc_vec = PBC_move - np.full((total_no_mol,3), PBC_move[m,:])
  pbc_dis.append(np.abs(pbc_vec)) 

pbc_dis = np.array(pbc_dis)
#print(pbc_dis.shape)



############### Read Kinetic Monte Carlo results  ###############
all_mobility = []
###### List all the files using the same CG configuration ######
filtered_files = list_files()
print(len(filtered_files))

for kmc_file in filtered_files:
  ###### Read hopping index ######
  kmc_traj = np.load(kmc_folder+kmc_file)
  #print(kmc_traj.shape,kmc_traj[:10,1].astype(int))
  ###### Read the index of initial site ######
  initial_site = identify_initial_site(kmc_file)
  #time_interval = kmc_traj[1:,0] - kmc_traj[:-1,0] 
  #print(time_interval)
  
  
  ##### Move the reference molecule to the center of the box #####
  COM_move = COM - np.full((total_no_mol,3), COM[initial_site,:]) + 0.5*box_size

  ##### Deal with periodic boundary condition #####
  COM_move = np.where((COM_move>box_size), (COM_move-box_size), COM_move)
  COM_move = np.where((COM_move<0.0), (COM_move+box_size), COM_move)

  ##### Using the initial hopping site as the coordainte origin #####
  #vec = COM_move - np.full((total_no_mol,3), COM_move[initial_site,:])
  origin_dis = [] # [xyz, particle-1, particle-2] a symmetric matrix
  for k in range(3):
    origin_dis.append(np.abs(COM_move[np.newaxis,:,k]-COM_move[np.newaxis,:,k].T))
  origin_dis = np.array(origin_dis)
  #print(origin_dis[2,:,initial_site])
  #print(origin_dis[:,initial_site,:]==origin_dis[:,:,initial_site])
  #print(origin_dis.shape)


  ############ Calculate the displacement during the KMC process ############
  kmc_traj_index = kmc_traj[:,1].astype(int)
  kmc_traj_time = kmc_traj[:,0]/1000 #change unit from ps to ns
  no_hopping_steps = kmc_traj.shape[0]
  #print(no_hopping_steps,kmc_traj_index)

  cross_check = np.full(3, 0, dtype=int)
  hop_dis = [np.full(3, 0.00000)]

  for t in range(time_end-1):
    t0_site = kmc_traj_index[t]
    t1_site = kmc_traj_index[t+1]
    #print(t0_site,t1_site)
  
    temp_dis = []
    for k in range(3):
      ##### Hopping inside the box #####
      if (round(origin_dis[k,t0_site,t1_site],4)<= round(pbc_dis[t0_site,t1_site,k],4) and cross_check[k]==0): 
        temp_dis.append(origin_dis[k,initial_site,t1_site])  
        cross_check[k] = 0  
    
      ##### Hopping across the box #####
      elif (round(origin_dis[k,t0_site,t1_site],4) > round(pbc_dis[t0_site,t1_site,k],4)):
        #print('Across the box')
        ### Move forward ###
        if (cross_check[k]==0):
          temp_dis.append(hop_dis[t][k] + pbc_dis[t0_site,t1_site,k])
          ### To positive ###
          if (COM_move[t1_site,k]<COM_move[t0_site,k]):
            cross_check[k] += 1
          ### To negative ###
          elif (COM_move[t1_site,k]>COM_move[t0_site,k]):
            cross_check[k] += -1  
          
        ### Move backward ###
        elif (cross_check[k]!=0):
          temp_dis.append(hop_dis[t][k] - pbc_dis[t0_site,t1_site,k])
          ### To positive ###
          if (COM_move[t1_site,k]<COM_move[t0_site,k]):
            cross_check[k] += -1
          ### To negative ###
          elif (COM_move[t1_site,k]>COM_move[t0_site,k]):
            cross_check[k] += 1  
          
        
      ##### Hopping outside the original box but not across another box #####
      elif (round(origin_dis[k,t0_site,t1_site],4)<= round(pbc_dis[t0_site,t1_site,k],4) and cross_check[k]!=0): 
        ### Move forward to positive ###
        if (COM_move[t1_site,k]>COM_move[t0_site,k] and cross_check[k]>0):
          temp_dis.append(hop_dis[t][k] + origin_dis[k,t0_site,t1_site])
        ### Move forward to negative ###
        elif (COM_move[t1_site,k]<COM_move[t0_site,k] and cross_check[k]<0):
          temp_dis.append(hop_dis[t][k] + origin_dis[k,t0_site,t1_site])  
        ### Move backward to positive ###
        elif (COM_move[t1_site,k]<COM_move[t0_site,k] and cross_check[k]>0): 
          temp_dis.append(hop_dis[t][k] - origin_dis[k,t0_site,t1_site])
        ### Move backward to negative ###
        elif (COM_move[t1_site,k]>COM_move[t0_site,k] and cross_check[k]<0): 
          temp_dis.append(hop_dis[t][k] - origin_dis[k,t0_site,t1_site])  
          #print(k,t0_site,t1_site,origin_dis[k,t0_site,t1_site])     
    
    hop_dis.append(np.array(temp_dis))  

  hop_dis = np.array(hop_dis)
  MSD = hop_dis**2

  ############ Calculate charge diffusivity & charge mobility ############
  mobility = [int(initial_site)]
  ##### Perform linear regression ####
  for k in range(3):
    #initial_guess = [100.0, 0]
    initial_guess = [100.0]
    x = kmc_traj_time[:time_end] 
    y_true = MSD[:time_end,k]
    fit = minimize(obj_function, initial_guess, method='Nelder-Mead', options={'maxiter':50000}, tol=0.0000001)

    diffusivity = fit.x[0]/2.0
    mobility.append(diffusivity/kbT)

    #plot_scatter(kmc_traj_time[:time_end], MSD[:time_end,k], fit.x) 

  mobility = np.array(mobility)
  #print(mobility) #unit in angstrom**2/ns
  all_mobility.append(mobility)

all_mobility = np.array(all_mobility)
np.savetxt(output_path,all_mobility)
#print(all_mobility)
