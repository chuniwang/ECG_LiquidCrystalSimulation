import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from scipy.stats import gaussian_kde
from scipy.spatial import distance
import os,re,sys



################# Set file path & parameters #################
no_snapshot = 120 #40 # Total number of snapshots of CG trajectroy 
no_sampling = 20 #10 # Total number of sampling Hamiltonians by a given snapshot (mimic backmapping process)

keyword_list = ['700K','555K','515K']
keyword_list = ['515K','555K','700K']

make_IPR_histo = 'no'
make_IPR_DOS_scatter = 'no'

constant_onsite_energy = 'yes' #sys.argv[1] #'no'
H_type = 'Gaussian_sampling' #'ECG_mean'
read_IPR = 'yes'


if (constant_onsite_energy == 'no'):
  if (H_type == 'ECG_mean'):
    figurename_histo = 'Histo_IPR_H_ECG_mean.png'
  if (H_type == 'Gaussian_sampling'):
    figurename_histo = 'Histo_IPR_H_Sampling.png'
if (constant_onsite_energy == 'yes'):
  if (H_type == 'ECG_mean'):
    figurename_histo = 'Histo_IPR_H_ECG_mean_Constant_SiteEnergy.png'
  if (H_type == 'Gaussian_sampling'):
    figurename_histo = 'Histo_IPR_H_Sampling_Constant_SiteEnergy.png'

figurename_scatter = []
for keyword in keyword_list:
  if (constant_onsite_energy == 'no'):
    if (H_type == 'ECG_mean'):
      figurename_scatter.append(('Scatter_IPR_DOS_' + keyword + '_H_ECG_mean.png'))
    if (H_type == 'Gaussian_sampling'):
      figurename_scatter.append(('Scatter_IPR_DOS_' + keyword + '_H_Sampling.png')) 
  if (constant_onsite_energy == 'yes'):
    if (H_type == 'ECG_mean'):
      figurename_scatter.append(('Scatter_IPR_DOS_' + keyword + '_H_ECG_mean_Constant_SiteEnergy.png')) 
    if (H_type == 'Gaussian_sampling'):
      figurename_scatter.append(('Scatter_IPR_DOS_' + keyword + '_H_Sampling_Constant_SiteEnergy.png')) 


labels = ['Isotropic', 'Smectic A', 'Smectic E']
labels = ['Smectic E', 'Smectic A', 'Isotropic']
colors = ['#548235','#1F4E79','#FFC000']
colors = ['#FFC000','#1F4E79','#548235']
markers = ['D', 'o']
figure_dpi = 250


total_IPR =[] # [morphology, snapshot(config), backmapping(sampling), states]
total_onsite_E = [] #[morphology, (no_config*no_sampling), states]
for keyword in keyword_list:
  file_folder = './CG_NVT_' + keyword + '_8Cells/'  # 700K or 555K
  IPR_config = []
  onsite_E = []
  for config in range(1,(no_snapshot+1)):
    IPR_sampling = []
    if (H_type == 'ECG_mean'):
      sampling_start = 0
      sampling_end = 1
    if (H_type == 'Gaussian_sampling'):
      sampling_start = 1
      sampling_end = no_sampling+1
    #for sampling in range(no_sampling):
    #for sampling in range(1,(no_sampling+1)):
    for sampling in range(sampling_start,sampling_end): 
      if (read_IPR != 'yes'):  
        ###################### Calculate IPR (Charge delocalization) ######################
        ########### Read eigenvectors of singe Hamiltonian ###########
        if (constant_onsite_energy == 'no'):
          eigen_w_file = file_folder + 'ECG_Results/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
          eigen_v_file = file_folder + 'ECG_Results/Diagonalization/Eigen_Vector_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
          #eigen_w_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
          #eigen_v_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Vector_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
        if (constant_onsite_energy == 'yes'):
          eigen_w_file = file_folder + 'ECG_Results/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'_Constant_SiteEnergy.npy'
          eigen_v_file = file_folder + 'ECG_Results/Diagonalization/Eigen_Vector_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'_Constant_SiteEnergy.npy'
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

      if (read_IPR == 'yes'):
        if (constant_onsite_energy == 'yes'):
          IPR_filename = file_folder + 'ECG_Results/Diagonalization/IPR_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'_Constant_SiteEnergy.npy' 
        if (constant_onsite_energy == 'no'): 
          IPR_filename = file_folder + 'ECG_Results/Diagonalization/IPR_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy' 
        IPR = np.load(IPR_filename)
        
        ########### Read eigenvectors of singe Hamiltonian ###########
        #if (constant_onsite_energy == 'no'):
        #  eigen_w_file = file_folder + 'ECG_Results/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.npy'
        #if (constant_onsite_energy == 'yes'):
        #  eigen_w_file = file_folder + 'ECG_Results/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'_Constant_SiteEnergy.npy'
        #eigen_w = np.load(eigen_w_file)
        #eigen_w = eigen_w.real
        #onsite_E.append(eigen_w)

      IPR_sampling.append(IPR)
      
    IPR_sampling = np.array(IPR_sampling)
    IPR_config.append(IPR_sampling)
  IPR_config = np.array(IPR_config)
  total_IPR.append(IPR_config)
  #onsite_E = np.array(onsite_E)
  #total_onsite_E.append(onsite_E)   
total_IPR = np.array(total_IPR)
#total_onsite_E = np.array(total_onsite_E)
print(total_IPR.shape)      
#print(total_onsite_E.shape)
np.save('Total_IPR.npy',total_IPR)

if (make_IPR_histo == 'yes'):
  ###################### Generate IPR histogram ######################
  print('Makeing charge delocalization (IPR) histogram...')
  fig = plt.figure(figsize=(20, 5), dpi=figure_dpi)
  ################# Define bin values#################
  print('You may need to customize the bin values!') 
  x = np.logspace(np.log10(20),np.log10(3500), 40)
  logbins = np.concatenate((np.arange(20), x), axis=0)
  
  for l in range(len(keyword_list)):
    IPR_all = total_IPR[l,:,:,:]
    ################# Histograming over all sampled IPR #################
    all_IPR_hist = []
    for i in range(no_snapshot):
      IPR_hist, bins = np.histogram(IPR_all[i,:,:], bins=logbins)#, density=True)
      IPR_hist = IPR_hist/no_sampling #*100/IPR_hist.sum() 
      all_IPR_hist.append(IPR_hist)
    IPR_hist_mean = np.mean(all_IPR_hist,axis=0)
    IPR_hist_std = np.std(all_IPR_hist,axis=0)
    #print(IPR_hist_mean.shape, IPR_hist_std.shape)
    #print(IPR_hist_std)
    long_range_delocalization_percentage = 1-np.array(all_IPR_hist)[:,:11].sum()/(no_snapshot*no_sampling*9000)
    print('The percentage of charge delocalizing across %f molecules: %f' %(bins[11],long_range_delocalization_percentage))



    ################# Generate histogram figure #################
    ########## Set up error bar #########
    bin_centers = 0.5*(logbins[1:] + logbins[:-1])
    plt.errorbar(
      bin_centers,
      IPR_hist_mean,
      yerr = np.log10(IPR_hist_std),
      #marker = '.',
      drawstyle = 'steps-mid',
      color=colors[l],
      alpha=0.8,
      ecolor=colors[l],
      #lw=1.5, capsize=3, capthick=1.5 
      lw=2.5, capsize=4, capthick=2.5
      )

    plt.hist(logbins[:-1], logbins, weights=IPR_hist_mean, color=colors[l], alpha=0.5, label=labels[l])

  plt.xlim(0.9, 3500)
  plt.ylim(0.01, 12000) # Without on-site disorder (log scale)
  #plt.ylim(0.1, 8500) # Without on-site disorder (linear scale)
  #plt.ylim(0.1, 5500) # With on-site disorder 
  plt.xscale('log')
  plt.yscale('log')

  plt.xlabel('Charge delocalization (IPR)', fontsize=35)
  plt.ylabel('Count', fontsize=35)

  plt.xticks(fontsize=28, fontname="Arial")
  plt.yticks(fontsize=28, fontname="Arial")

  ##########  Set the tick line size for both x and y axes ########## 
  plt.tick_params(axis='both', which='major', length=12)  # Adjust the length for major ticks
  plt.tick_params(axis='both', which='minor', length=5)  # Set a shorter length for minor ticks
  
  plt.legend(frameon=False, fontsize=32, loc='upper right')
  plt.savefig((figurename_histo), format="png", bbox_inches="tight")
  #plt.show()
  plt.close()


if (make_IPR_DOS_scatter == 'yes'):
  ################# Generate IPR-DOS correlation scatter plot #################
  print('Makeing IPR-DOS correlation scatter plot...') 
  for l in range(len(keyword_list)):
    plt.figure(figsize = (10, 10), dpi=250)

    ##### Calculate the point density #####
    #x, y = all_eigen_w.flatten(), IPR_all.flatten()
    x, y = total_onsite_E[l,:,:].flatten(), total_IPR[l,:,:,:].flatten()
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    ##### Sort the points by density, so the densest points are plotted last #####
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=25, cmap='viridis')   #edgecolor=''
    
    plt.xlabel('Energy [eV]', fontsize=35)
    plt.ylabel('Charge delocalization (IPR)', fontsize=35)
    if (H_type == 'Gaussian_sampling'):
      plt.xticks(np.arange(-8.2, -6.3, step=0.2), fontsize=24, fontname="Arial", rotation=45) #rotation=90
      plt.xlim(-8.2, -6.4)
    if (H_type == 'ECG_mean'):
      plt.xticks(np.arange(-7.9, -6.8, step=0.1), fontsize=24, fontname="Arial", rotation=45)
      plt.xlim(-7.9, -6.8)
    #matplotlib.rc('ytick', labelsize=20)

    #####  Set the parameters of y axes ##### 
    plt.yscale("log")  
    plt.tick_params(axis='y', which='major', length=12)  # Adjust the length for major ticks
    plt.tick_params(axis='y', which='minor', length=5)  # Set a shorter length for minor ticks    
    ytick_values = np.array([1,2,10,100,1000])
    plt.yticks(ytick_values, fontsize=24, fontname="Arial") #rotation=90
    plt.tick_params(labelsize = 24)
    plt.ylim(0.9, 1000)
  
    plt.savefig(figurename_scatter[l], format="png", bbox_inches="tight")
    #plt.show()
    plt.close()    



