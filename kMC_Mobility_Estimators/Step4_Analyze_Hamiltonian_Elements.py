import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,sys
from statsmodels.stats.weightstats import ztest as ztest
import scipy.stats as stats
import csv



################# Set file path & parameters #################
no_snapshot = 50 #40 # Total number of snapshots of CG trajectroy 
no_sampling = 10 #10 # Total number of sampling Hamiltonians by a given snapshot (mimic backmapping process)

keyword_list = ['700K','555K']
keyword_list = ['700K']

make_IPR_histo = 'no'
make_IPR_DOS_scatter = 'yes'

constant_onsite_energy = 'no' #sys.argv[1] #'no'
H_type = 'ECG_mean' #sys.argv[2] #'Gaussian_sampling' 'ECG_mean'
figure_dpi = 250


for keyword in keyword_list:
  file_folder = './CG_NVT_' + keyword + '_8Cells/ECG_Results/Diagonalization/'  # 700K or 555K
  p_value_HOMO = []
  p_value_coupling = []
  ECG_mean_hist_HOMO = []
  ECG_mean_hist_coupling = []
  G_sampling_hist_HOMO = []
  G_sampling_hist_coupling = []
  ################# Read ECG dataset #################
  ##### HOMO energy #####
  if keyword == '555K':
    HOMO_dataset = np.load('/home/ciwang/Electronic_Coupling/ML_Dataset/' + keyword + '/Label_Energy_Sample_132081.npy')[:,0]
    Coupling_dataset = np.load('/home/ciwang/Electronic_Coupling/ML_Dataset/' + keyword + '/Label_Coupling_Sample_92160.npy')
    #HOMO_dataset = np.load('/home/ciwang/Electronic_Coupling/ML_Dataset/' + keyword + '/Label_Energy_1_10240.npy')[:,0]
  if keyword == '700K':
    HOMO_dataset = np.load('/home/ciwang/Electronic_Coupling/ML_Dataset/' + keyword + '/Label_Energy_Sample_182098.npy')[:,0]
    Coupling_dataset = np.load('/home/ciwang/Electronic_Coupling/ML_Dataset/' + keyword + '/Label_Coupling_Sample_128000.npy')
  HOMO_dataset = HOMO_dataset * 27.211386245988 #change unit from hatree to eV 
  Coupling_dataset = np.log10(abs(Coupling_dataset*1000)) #change unit from eV to meV

  for config in range(1,(no_snapshot+1)):
    ################# Read Hamiltonian based on ECG predicted mean value #################
    ham_filename = 'Hamiltonian_' + keyword + '_Config_' + str(config) + '-0.npy'
    H_ECG_mean = np.load((file_folder+ham_filename))
    print(ham_filename)
    ########## HOMO (onsite) energy ##########
    HOMO_ECG_mean = np.diag(H_ECG_mean) 
    #print(ztest(HOMO_dataset, HOMO_ECG_mean, value=0))
    #print(stats.ttest_ind(HOMO_dataset, HOMO_ECG_mean, equal_var=False))
    #print('-----')
    ##### Calcuate p value #####
    p_value_HOMO.append([(keyword + '_Config_' + str(config) + '-0'), '-',
                       ztest(HOMO_dataset, HOMO_ECG_mean, value=0)[1]
                      ])

    ########## Electronic coupling ##########
    Coupling_ECG_mean = H_ECG_mean[np.triu(np.ones((9000, 9000), dtype=bool),k=1)].flatten()
    Coupling_ECG_mean = Coupling_ECG_mean[Coupling_ECG_mean !=0]
    Coupling_ECG_mean = np.log10(abs(Coupling_ECG_mean*1000)) #change unit from eV to meV
    ##### Calcuate p value #####
    p_value_coupling.append([(keyword + '_Config_' + str(config) + '-0'), '-',
                       ztest(Coupling_dataset, Coupling_ECG_mean, value=0)[1],
                       10**(np.sort(Coupling_ECG_mean)[-1]),
                       10**(np.sort(Coupling_ECG_mean)[-2]),
                       10**(np.sort(Coupling_ECG_mean)[-3])
                      ])
    #print(ztest(Coupling_dataset, Coupling_ECG_mean, value=0)[1])
    #print(ztest(np.log10(Coupling_dataset), np.log10(Coupling_ECG_mean), value=0)[1])
    #print(10**(np.sort(Coupling_ECG_mean)[-5:]))

    for sampling in range(1,(no_sampling+1)):
      ################# Read Hamiltonian sampling from Gaussian distribution  #################
      ham_filename = 'Hamiltonian_' + keyword + '_Config_' + str(config) + '-' + str(sampling) +'.npy'
      print(ham_filename)
      H = np.load((file_folder+ham_filename))
      ########## HOMO (onsite) energy ##########
      HOMO_G_sampling = np.diag(H)
      #print(stats.ttest_ind(HOMO_dataset, HOMO_G_sampling, equal_var=False))
      #print(ztest(HOMO_ECG_mean, HOMO_G_sampling, value=0) )
      ##### Calcuate p value #####
      p_value_HOMO.append([(keyword + '_Config_' + str(config) + '-' + str(sampling)),
                       ztest(HOMO_ECG_mean, HOMO_G_sampling, value=0)[1],
                       ztest(HOMO_dataset, HOMO_G_sampling, value=0)[1]
                      ])

      
      ########## Electronic coupling ##########
      Coupling_G_sampling = H[np.triu(np.ones((9000, 9000), dtype=bool),k=1)].flatten()
      Coupling_G_sampling = Coupling_G_sampling[Coupling_G_sampling !=0]
      Coupling_G_sampling = np.log10(abs(Coupling_G_sampling*1000)) #change unit from eV to meV
      ##### Calcuate p value #####
      p_value_coupling.append([(keyword + '_Config_' + str(config) + '-' + str(sampling)),
                       ztest(Coupling_ECG_mean, Coupling_G_sampling, value=0)[1],
                       ztest(Coupling_dataset, Coupling_G_sampling, value=0)[1],
                       10**(np.sort(Coupling_G_sampling)[-1]),
                       10**(np.sort(Coupling_G_sampling)[-2]),
                       10**(np.sort(Coupling_G_sampling)[-3])
                      ])
      #print(ztest(Coupling_dataset, Coupling_G_sampling, value=0)[1])
      #print(ztest(np.log10(Coupling_dataset), np.log10(Coupling_G_sampling), value=0)[1])
      #print(10**(np.sort(Coupling_G_sampling)[-5:]))


         
      ################# Compare the HOMO distribution  #################
      output_folder = './CG_NVT_' + keyword + '_8Cells/ECG_Results/Figures/'
      figurename_histo = output_folder + 'Compare_HOMO_' + keyword + '_Config_' + str(config) + '-' + str(sampling) +'.png'
      ##### Specify bining boundary #####
      bin_size = 0.02 # bin size for histogram !!! unit in eV !!!
      up_bound = -6.6
      low_bound = -8.2
      bins = np.arange(low_bound, up_bound, step=bin_size)
      bin_centers = 0.5*(bins[1:] + bins[:-1])

      ##### Histograming #####
      dataset_hist, bins = np.histogram(HOMO_dataset, bins=bins, density=True) 
      dataset_hist = dataset_hist*100/dataset_hist.sum()
      ECG_mean_hist, bins = np.histogram(HOMO_ECG_mean, bins=bins, density=True) 
      ECG_mean_hist = ECG_mean_hist*100/ECG_mean_hist.sum()
      G_sampling_hist, bins = np.histogram(HOMO_G_sampling, bins=bins, density=True)
      G_sampling_hist = G_sampling_hist*100/G_sampling_hist.sum()
      if sampling==1:
        ECG_mean_hist_HOMO.append(ECG_mean_hist)
      G_sampling_hist_HOMO.append(G_sampling_hist)

      ##### Make histogram plot #####
      plt.figure(figsize=(10, 10), dpi=figure_dpi)
      x_min = low_bound
      x_max = up_bound
      y_max = np.array([dataset_hist.max(),ECG_mean_hist.max(), G_sampling_hist.max()]).max()
      if (y_max > 75):
        y_max = 100.0
      else:
        y_max = y_max * 1.3
      y_max = 15

      plt.hist(bins[:-1], bins, weights=ECG_mean_hist, color="#1F4E79", alpha=0.8, label='ECG_mean')
      plt.hist(bins[:-1], bins, weights=G_sampling_hist, color="#FFC000", alpha=0.6, label='Gaussian_sampling')
      plt.plot(bin_centers, dataset_hist, linewidth=3, color="black", linestyle="--",label='Training dataset')

      plt.xticks(np.arange(x_min, x_max+1, step=0.2), fontsize=24, fontname="Arial", rotation=45)
      plt.yticks(fontsize=24, fontname="Arial")
      plt.xlabel('HOMO Energy [eV]', fontsize=32, fontname="Arial")
      plt.ylabel('Density of States', fontsize=32, fontname="Arial")
      plt.legend(frameon=False, fontsize=32, loc='upper left')
      plt.xlim(x_min, x_max)
      plt.ylim(0, y_max)
      plt.savefig((figurename_histo), dpi=figure_dpi,format="png", bbox_inches="tight")
      #plt.show()
      plt.close()


      ################# Compare the electronic coupling distribution  #################
      output_folder = './CG_NVT_' + keyword + '_8Cells/ECG_Results/Figures/'
      figurename_histo = output_folder + 'Compare_Coupling_' + keyword + '_Config_' + str(config) + '-' + str(sampling) +'.png'
      ##### Specify bining boundary #####
      bin_size = 0.1 # bin size for histogram !!! unit in meV & log scale !!!
      up_bound = 4
      low_bound = -4
      bins = np.arange(low_bound, up_bound, step=bin_size)
      bin_centers = 0.5*(bins[1:] + bins[:-1])

      ##### Histograming #####
      dataset_hist_coupling, bins = np.histogram(Coupling_dataset, bins=bins, density=True) 
      dataset_hist_coupling = dataset_hist_coupling*100/dataset_hist_coupling.sum()
      ECG_mean_hist, bins = np.histogram(Coupling_ECG_mean, bins=bins, density=True) 
      ECG_mean_hist = ECG_mean_hist*100/ECG_mean_hist.sum()
      G_sampling_hist, bins = np.histogram(Coupling_G_sampling, bins=bins, density=True)
      G_sampling_hist = G_sampling_hist*100/G_sampling_hist.sum()
      if sampling==1:
        ECG_mean_hist_coupling.append(ECG_mean_hist)
      G_sampling_hist_coupling.append(G_sampling_hist)

      ##### Make histogram plot #####
      plt.figure(figsize=(10, 10), dpi=figure_dpi)
      x_min = low_bound
      x_max = up_bound
      y_max = np.array([dataset_hist_coupling.max(),ECG_mean_hist.max(), G_sampling_hist.max()]).max()
      if (y_max > 75):
        y_max = 100.0
      else:
        y_max = y_max * 1.3
      #y_max = 15

      plt.hist(bins[:-1], bins, weights=ECG_mean_hist, color="#1F4E79", alpha=0.8, label='ECG_mean')
      plt.hist(bins[:-1], bins, weights=G_sampling_hist, color="#FFC000", alpha=0.6, label='Gaussian_sampling')
      plt.plot(bin_centers, dataset_hist_coupling, linewidth=3, color="black", linestyle="--",label='Training dataset')

      plt.xticks(np.arange(x_min, x_max+1, step=1), fontsize=24, fontname="Arial") #, rotation=45)
      plt.yticks(fontsize=24, fontname="Arial")
      plt.xlabel('log$_{10}$|Coupling| [meV]', fontsize=28, fontname="Arial")
      plt.ylabel('Probability density', fontsize=28, fontname="Arial")
      plt.legend(frameon=False, fontsize=24, loc='upper left')
      plt.xlim(x_min, x_max)
      plt.ylim(0, y_max)
      plt.savefig((figurename_histo), dpi=figure_dpi,format="png", bbox_inches="tight")
      #plt.show()
      plt.close()
      

  ################# Generate average histogram  #################
  output_folder = './CG_NVT_' + keyword + '_8Cells/'
  
  ########## HOMO (onsite) energy ##########
  figurename_histo = output_folder + 'Compare_HOMO_' + keyword + '_average.png'
  ECG_mean_hist_HOMO = np.array(ECG_mean_hist_HOMO)
  ECG_mean_hist_mean = np.mean(ECG_mean_hist_HOMO,axis=0) 
  ECG_mean_hist_std = np.std(ECG_mean_hist_HOMO,axis=0) 
  G_sampling_hist_HOMO = np.array(G_sampling_hist_HOMO)
  G_sampling_hist_mean = np.mean(G_sampling_hist_HOMO,axis=0) 
  G_sampling_hist_std = np.std(G_sampling_hist_HOMO,axis=0) 
  
  ##### Specify bining boundary #####
  bin_size = 0.02 # bin size for histogram !!! unit in eV !!!
  up_bound = -6.6
  low_bound = -8.2
  bins = np.arange(low_bound, up_bound, step=bin_size)
  bin_centers = 0.5*(bins[1:] + bins[:-1])

  ##### Make histogram plot #####
  plt.figure(figsize=(10, 10), dpi=figure_dpi)
  x_min = low_bound
  x_max = up_bound
  y_max = np.array([dataset_hist.max(),ECG_mean_hist.max(), G_sampling_hist.max()]).max()
  if (y_max > 75):
    y_max = 100.0
  else:
    y_max = y_max * 1.3
  y_max = 15

  plt.hist(bins[:-1], bins, weights=ECG_mean_hist_mean, color="#1F4E79", alpha=0.8, label='ECG_mean')
  plt.hist(bins[:-1], bins, weights=G_sampling_hist_mean, color="#FFC000", alpha=0.6, label='Gaussian_sampling')

  ########## Set up error bar #########
  plt.errorbar(
      bin_centers,
      ECG_mean_hist_mean,
      yerr = ECG_mean_hist_std,
      #marker = '.',
      drawstyle = 'steps-mid',
      color="#1F4E79", 
      alpha=0.8,
      ecolor="#1F4E79",
      lw=1.5, capsize=3, capthick=1.5 
      #lw=2.5, capsize=4, capthick=2.5
      )
  plt.errorbar(
      bin_centers,
      G_sampling_hist_mean,
      yerr = G_sampling_hist_std,
      #marker = '.',
      drawstyle = 'steps-mid',
      color="#FFC000", 
      alpha=0.8,
      ecolor="#FFC000",
      lw=1.5, capsize=3, capthick=1.5 
      #lw=2.5, capsize=4, capthick=2.5
      )
  plt.plot(bin_centers, dataset_hist, linewidth=3, color="black", linestyle="--",label='Training dataset')

  plt.xticks(np.arange(x_min, x_max+1, step=0.2), fontsize=24, fontname="Arial", rotation=45)
  plt.yticks(fontsize=24, fontname="Arial")
  plt.xlabel('HOMO Energy [eV]', fontsize=32, fontname="Arial")
  plt.ylabel('Density of States', fontsize=32, fontname="Arial")
  plt.legend(frameon=False, fontsize=32, loc='upper left')
  plt.xlim(x_min, x_max)
  plt.ylim(0, y_max)
  plt.savefig((figurename_histo), dpi=figure_dpi,format="png", bbox_inches="tight")
  #plt.show()
  plt.close()  
  

  ########## Electronic coupling ##########
  figurename_histo = output_folder + 'Compare_Coupling_' + keyword + '_average.png'
  ECG_mean_hist_coupling = np.array(ECG_mean_hist_coupling)
  ECG_mean_hist_mean = np.mean(ECG_mean_hist_coupling,axis=0) 
  ECG_mean_hist_std = np.std(ECG_mean_hist_coupling,axis=0) 
  G_sampling_hist_coupling = np.array(G_sampling_hist_coupling)
  G_sampling_hist_mean = np.mean(G_sampling_hist_coupling,axis=0) 
  G_sampling_hist_std = np.std(G_sampling_hist_coupling,axis=0) 
  
  ##### Specify bining boundary #####
  bin_size = 0.1 # bin size for histogram !!! unit in meV & log scale !!!
  up_bound = 4
  low_bound = -4
  bins = np.arange(low_bound, up_bound, step=bin_size)
  bin_centers = 0.5*(bins[1:] + bins[:-1])

  ##### Make histogram plot #####
  plt.figure(figsize=(10, 10), dpi=figure_dpi)
  x_min = low_bound
  x_max = up_bound
  y_max = np.array([dataset_hist.max(),ECG_mean_hist.max(), G_sampling_hist.max()]).max()
  if (y_max > 75):
    y_max = 100.0
  else:
    y_max = y_max * 1.3
  y_max = 15

  plt.hist(bins[:-1], bins, weights=ECG_mean_hist_mean, color="#1F4E79", alpha=0.8, label='ECG_mean')
  plt.hist(bins[:-1], bins, weights=G_sampling_hist_mean, color="#FFC000", alpha=0.6, label='Gaussian_sampling')

  ########## Set up error bar #########
  plt.errorbar(
      bin_centers,
      ECG_mean_hist_mean,
      yerr = ECG_mean_hist_std,
      #marker = '.',
      drawstyle = 'steps-mid',
      color="#1F4E79", 
      alpha=0.8,
      ecolor="#1F4E79",
      lw=1.5, capsize=3, capthick=1.5 
      #lw=2.5, capsize=4, capthick=2.5
      )
  plt.errorbar(
      bin_centers,
      G_sampling_hist_mean,
      yerr = G_sampling_hist_std,
      #marker = '.',
      drawstyle = 'steps-mid',
      color="#FFC000", 
      alpha=0.8,
      ecolor="#FFC000",
      lw=1.5, capsize=3, capthick=1.5 
      #lw=2.5, capsize=4, capthick=2.5
      )
  plt.plot(bin_centers, dataset_hist_coupling, linewidth=3, color="black", linestyle="--",label='Training dataset')

  plt.xticks(np.arange(x_min, x_max+1, step=1), fontsize=24, fontname="Arial") #, rotation=45)
  plt.yticks(fontsize=24, fontname="Arial")
  plt.xlabel('log$_{10}$|Coupling| [meV]', fontsize=28, fontname="Arial")
  plt.ylabel('Probability density', fontsize=28, fontname="Arial")
  plt.legend(frameon=False, fontsize=24, loc='upper left')
  plt.xlim(x_min, x_max)
  plt.ylim(0, y_max)
  plt.savefig((figurename_histo), dpi=figure_dpi,format="png", bbox_inches="tight")
  #plt.show()
  plt.close()  


  ################# Save p value  #################
  ########## HOMO (onsite) energy ##########
  ##### Specify the CSV file path #####
  csv_file = './CG_NVT_' + keyword + '_8Cells/p_value_HOMO_' + keyword + '.csv'

  ##### Open the CSV file in write mode #####
  with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    
    ##### Write the header row (optional) #####
    writer.writerow(['Hamiltonian', 'p-ECG_mean', 'p-ML_dataset'])

    ##### Write the data rows #####
    for row in p_value_HOMO:
        writer.writerow(row)    


  ########## Electronic coupling ##########
  ##### Specify the CSV file path #####
  csv_file = './CG_NVT_' + keyword + '_8Cells/p_value_Coupling_' + keyword + '.csv'

  ##### Open the CSV file in write mode #####
  with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    
    ##### Write the header row (optional) #####
    writer.writerow(['Hamiltonian', 'p-ECG_mean', 'p-ML_dataset', 'Top1', 'Top2', 'Top3'])

    ##### Write the data rows #####
    for row in p_value_coupling:
        writer.writerow(row)   

