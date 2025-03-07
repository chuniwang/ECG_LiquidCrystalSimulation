import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,sys



################# Set file path & parameters #################
##### Input files #####
keyword = sys.argv[1] #Provide number index of each snapshot
file_folder = './CG_NVT_' + keyword + '_8Cells/'  #type 700K or 555K
output_filename = file_folder + 'Electronic_Data_' + keyword +'.npy'

no_sampling = 3
no_config = 120
no_state = 9000

data = np.zeros((no_config,no_sampling,no_state,4)) #[config, sampling, CT-state, [center, IPR, eigenvalue, center-probability]]

for config in range(no_config):
  for i in range(no_sampling):
    ################# Read files #################
    eigen_w_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Value_' + keyword + '_Config_' + str(config+1) +'-' + str(i+1) + '.npy'
    eigen_v_file = file_folder + '/ECG_Results/Diagonalization/Eigen_Vector_' + keyword + '_Config_' + str(config+1) +'-' + str(i+1) + '.npy'

    eigen_w = np.load(eigen_w_file)
    eigen_v = np.load(eigen_v_file)
    eigen_w = eigen_w.real
    eigen_v = eigen_v.real
    #print(eigen_w.max(),eigen_w.min())

    ############ Calculate inverse participation ratio (IPR) ##########
    eigen_v_prob = eigen_v**2
    IPR = np.sum(eigen_v_prob[:,:]**2,axis=0)
    IPR = 1/IPR
    print("\nIPR mean: %s" %IPR.mean())
    print("IPR std: %s" %IPR.std())
    print("IPR max: %s" %IPR.max())
    print("IPR min: %s" %IPR.min())
    
    
    for charge_state in range(no_state):
      charge_index = np.where(eigen_v[:,charge_state]**2>0.0001)[0]
      charge_center_index = charge_index[np.argmax(eigen_v[charge_index,charge_state]**2)]      
      #print(charge_center_index,eigen_v[charge_center_index,charge_state]**2)
      data[config,i,charge_state,0] = charge_center_index         # CT center
      data[config,i,charge_state,1] = IPR[charge_center_index]      # IPR
      data[config,i,charge_state,2] = eigen_w[charge_center_index]  # CT energy
      data[config,i,charge_state,3] = eigen_v[charge_center_index,charge_state]**2  # CT center probability
      #print(data[config,i,charge_state,:])

################# Read files #################
print(data.shape)
np.save(output_filename,data)
