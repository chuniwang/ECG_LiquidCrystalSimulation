import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,sys



################# Set file path & parameters #################
##### Input files #####
keyword = sys.argv[1] #Provide number index of each snapshot
file_folder = './CG_NVT_' + sys.argv[2] + '_8Cells/'  #type 700K or 555K
MO_energy_filename = file_folder + '/ECG_Results/HOMO_Energy_' + sys.argv[2] + '_Config_' + keyword +'.txt'
coupling_filename = file_folder + '/ECG_Results/Coupling_' + sys.argv[2] + '_Config_' + keyword +'.txt'
coupling_index_filename = file_folder + '/ML_Feature/Coupling_List_' + sys.argv[2] + '_Config_' + keyword + '.npy'

no_sampling = 20 #20 #10
bin_size = 0.02 # Bin size for histogram !!! unit in eV !!!
figure_dpi = 250


for i in range(no_sampling):
  ################# Define output files #################
  Hamiltonian_filename = file_folder + '/ECG_Results/Diagonalization/Hamiltonian_' + sys.argv[2] + '_Config_' + keyword +'-' + str(i+1) + '.npy'
  eigen_w_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Value_' + sys.argv[2] + '_Config_' + keyword +'-' + str(i+1) + '.npy'
  eigen_v_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Vector_' + sys.argv[2] + '_Config_' + keyword +'-' + str(i+1) + '.npy'
  IPR_file = file_folder + '/ECG_Results/Diagonalization/IPR_' + sys.argv[2] + '_Config_' + keyword +'-' + str(i+1) + '.npy'
  figurename_histo = file_folder + '/ECG_Results/Figures/DOS_' + sys.argv[2] + '_Config_' + keyword +'-' + str(i+1) + '.png'
  
  ################# Set the random seed #################
  random_seed = int(i+1)
  np.random.seed(random_seed)

  ################# Read HOMO energy #################
  HOMO_mean = np.loadtxt(MO_energy_filename)[:,0] # unit in eV
  HOMO_std = np.loadtxt(MO_energy_filename)[:,1] # unit in eV
  HOMO = np.random.normal(HOMO_mean,HOMO_std)
  no_mol = HOMO_mean.shape[0]
  #print(HOMO_mean[:5])
  #print(HOMO[:5])


  ################# Read electronic couplings & pair list #################
  coupling_mean = np.loadtxt(coupling_filename)[:,0] # meV in linear scale with phase
  coupling_std = np.loadtxt(coupling_filename)[:,1] # meV in log scale
  coupling_log = np.random.normal(np.log10(np.abs(coupling_mean)),coupling_std)
  coupling = 10**(coupling_log)*np.sign(coupling_mean)
  coupling /= 1000 #change unit from meV to eV
  coupling_info = np.load(coupling_index_filename)
  #print(coupling[:5])
  #print(coupling_mean[:5]/1000)
  #print(coupling_info.shape)



  ################# Define electronic Hamiltonian #################
  H = np.zeros((no_mol,no_mol), dtype=float)
  ######## Diagonal elements ########
  np.fill_diagonal(H, HOMO)
  #print(H[:5,:5])
  ######## Off-diagonal elements ########
  for i in range(coupling.shape[0]):
    H[coupling_info[i,1],coupling_info[i,2]] = coupling[i]
    H[coupling_info[i,2],coupling_info[i,1]] = coupling[i]
  #print(H[:5,:5])
  np.save(Hamiltonian_filename, H)



  if not os.path.exists(eigen_v_file):
    ######## Diagonalization ########
    eigen_w, eigen_v = np.linalg.eig(H)
    idx = np.argsort(eigen_w)
    eigen_w = eigen_w[idx]
    eigen_v = eigen_v[:,idx]
    eigen_w = eigen_w.real
    eigen_v = eigen_v.real
    np.save(eigen_w_file,eigen_w)
    np.save(eigen_v_file,eigen_v)
    #print(eigen_w.max(),eigen_w.min())
    #print(eigen_v.shape)
  else:
    eigen_w = np.load(eigen_v_file)
    eigen_v = np.load(eigen_v_file)
    eigen_w = eigen_w.real
    eigen_v = eigen_v.real
    #print(eigen_w.max(),eigen_w.min())


  ############ Calculate inverse participation ratio ##########
  eigen_v_prob = eigen_v**2
  IPR = np.sum(eigen_v_prob[:,:]**2,axis=0)
  IPR = 1/IPR
  print("IPR mean: %s" %IPR.mean())
  print("IPR std: %s" %IPR.std())
  print("IPR max: %s" %IPR.max())
  print("IPR min: %s" %IPR.min())
  np.save(IPR_file,IPR)

"""
#################### Plot Probability Density Distribution ####################
##### Specify bining boundary #####
up_bound = np.array([HOMO.max(), eigen_w.max()])
up_bound = up_bound.max()
low_bound = np.array([HOMO.min(), eigen_w.min()])
low_bound = low_bound.min()
bins = np.arange(round(low_bound,1), round(up_bound,1), step=bin_size)

##### Histograming #####
HOMO_hist, bins = np.histogram(HOMO, bins=bins, density=True)
HOMO_hist = HOMO_hist*100/HOMO_hist.sum()
eigen_hist, bins = np.histogram(eigen_w, bins=bins, density=True)
eigen_hist = eigen_hist*100/eigen_hist.sum()


##### Make histogram plot #####
plt.figure(figsize=(10, 10), dpi=figure_dpi)
x_min = -8.0 #round(low_bound)
x_max = -6.8 #round(up_bound) + 0.2
y_max = np.array([HOMO_hist.max(), eigen_hist.max()]).max()
if (y_max > 75):
  y_max = 100.0
else:
  y_max = y_max * 1.35
y_max = 13

plt.hist(bins[:-1], bins, weights=HOMO_hist, color="#1F4E79", alpha=0.8, label='Predicted HOMO')
plt.hist(bins[:-1], bins, weights=eigen_hist, color="#FFC000", alpha=0.6, label='Eigenstates')
#plt.plot(E, gaussian_form, linewidth=2, color="indianred", linestyle="--")
#plt.plot(E, P_E, linewidth=2, color="black", linestyle="--", label='P(E|R),Testing')

plt.xticks(np.arange(x_min, x_max+1, step=0.2), fontsize=24, fontname="Arial") #rotation=90
plt.yticks(fontsize=24, fontname="Arial")
plt.xlabel('HOMO Energy [eV]', fontsize=28, fontname="Arial")
plt.ylabel('Density of States', fontsize=28, fontname="Arial")
plt.legend(frameon=False, fontsize=24, loc='upper left')
plt.xlim(x_min, x_max)
plt.ylim(0, y_max)
plt.savefig((figurename_histo), dpi=figure_dpi,format="png", bbox_inches="tight")
#plt.show()
plt.close()
"""
