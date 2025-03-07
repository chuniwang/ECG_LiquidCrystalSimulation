import numpy as np
import random
import sys
import time
np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

file_keyword = '555K'
#config_keyword = '9-1'
config_keyword = sys.argv[2]
initial_site = int(sys.argv[1])

ham_file = '/home/ciwang/LAMPPS/CG_Simulation/Multiple_Snapshot/CG_NVT_' + file_keyword + \
           '_8Cells/ECG_Results/Diagonalization/Hamiltonian_' + file_keyword + '_Config_' + \
           config_keyword + '.npy'
#ham_file = '../../NVT_' + file_keyword + '' + '_8Cells/Hamiltonian_' + file_keyword + '_Config_1-1.npy'

output_path = './CG_NVT_' + file_keyword + '_8Cells/KMC_Results/' + \
                'Hopping_Traj_' +file_keyword + '_Config_' + config_keyword + \
                '_Site_' + str(initial_site) + '.npy'

#no_kmc_step = 5000000
no_kmc_step = 100000


###### Parameter related to kij matrix ######
lam_coup = 0.4301307255 #eV
#kbT = 0.0478225 #eV, 555K
kbT = 0.02585 #eV, 300K
#hbar = 6.58212e-16 #eV*s
hbar = 6.58212e-4 #eV*ps
no_atom_per_mol = 15 #Number of CG particles in a molecule


def cal_kij(ham):
  ###### Reorganization Energy ######
  lam_mat = lam_coup
  
  ###### Coupling Square ######
  n = ham.shape[0]
  s_mat = ham**2 #eV**2
  s_mat *= (np.ones((n,n),dtype = int) - np.identity(n)) #set diagonal = 0 CHECKED
  #print(s_mat.mean(),s_mat.min(),s_mat.max())
  #print("Percentage of the nonzero coupling btw states: %4.1f" %(100*np.nonzero(s_mat)[1].shape[0]/float(n*n)))
  
  ###### Energy Difference ######
  E_HOMO = np.diag(ham)
  e_temp = E_HOMO[np.newaxis,:]
  e_mat = -1*(e_temp - e_temp.T) #CHECKED
  #print(e_temp.shape)
  
  ###### Hopping rate matrix between sites ######
  k_mat = s_mat/hbar*np.sqrt(np.pi/kbT/lam_mat)*np.exp(-((e_mat-lam_mat)**2)/(4*kbT*lam_mat))
  #print('----- Compare energy difference and external field -----')
  #print(np.abs(e_mat).mean(),np.abs(e_mat).std())
  #print("Percentage of the nonzero rate constants: %4.1f" %(100*np.nonzero(k_mat)[1].shape[0]/float(n*n)))
  
  return k_mat


def rejection_free_kmc(initial_site, kij):
  total_time = 0.000
  hopping_traj = []
  for t in range(no_kmc_step):
    hopping_traj.append([total_time,initial_site])
    ###### Find possible hopping sites with nonzero rate  ######
    nonzero_index = np.nonzero(kij[:,initial_site])
    #print(nonzero_index)
    
    ###### Calculate the cumulative function & total rate  ######
    kij_cumulatvie = np.cumsum(kij[nonzero_index,initial_site]) # [cumulate function, initial site]
    total_rate = kij_cumulatvie[-1]
    #print(kij_cumulatvie,total_rate)
    
    ###### Determine the hoppin site by the cumulative function & random process ######
    ### Get a uniform random number ###
    random_value = random.random()
    #print(random_value)
    
    ### Decide the hopping site ###
    hopping_index = np.argwhere(kij_cumulatvie >= total_rate*random_value)[0]
    #print(total_rate*random_value,hopping_index)
    
    ###### Estimate hopping time interval ######
    ### Get a uniform random number ###
    random_value = random.random()
    #print(random_value)
    
    ### Calculate hopping time ###
    delta_t = -np.log(random_value)/total_rate
    total_time += delta_t
    #print(delta_t, total_time)
    
    ###### Assign new charge center  ######
    initial_site = nonzero_index[0][hopping_index][0]
    #print(initial_site)
    #print('\n')

  hopping_traj = np.array(hopping_traj)
  #print(hopping_traj[-1,0])
  np.save(output_path, hopping_traj)

############### Read Hamiltonian  ###############
ham = np.load(ham_file)
n = ham.shape[0]
############### Calculate hopping rate matrix, cumulatvie functions, total rate  ###############
kij = cal_kij(ham)  #k_ij [final,initial]
#kij_cumulatvie = np.cumsum(kij, axis=0) # [cumulate function, initial site]
#total_rate = kij_cumulatvie[-1,:]
#print(np.nonzero(kij)[0].shape[0])
#print("\nTotal number of the nonzero rate constants: %s" %(np.nonzero(kij)[0].shape[0]))
print("\nHopping initial site: %s" %(initial_site))

############### Read Hamiltonian  ###############
start_time = time.time() # Start the timer
rejection_free_kmc(initial_site, kij)

end_time = time.time() # End the timer

# Calculate the elapsed time
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

