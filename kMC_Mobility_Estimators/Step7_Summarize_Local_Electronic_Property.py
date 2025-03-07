import numpy as np
import csv


################# Set file path & parameters #################
keyword_list = ['700K','555K']
keyword = '515K'
file_folder = './CG_NVT_' + keyword + '_8Cells/'  # 700K or 555K

total_config = 100
no_sampling = 20 #10 # Total number of sampling Hamiltonians by a given snapshot (mimic backmapping process)
H_type = 'Gaussian_sampling' #'ECG_mean'
#H_type = 'ECG_mean' #sys.argv[2] #'Gaussian_sampling' 'ECG_mean'

if (H_type == 'ECG_mean'):
        sampling_start = 0
        sampling_end = 1
if (H_type == 'Gaussian_sampling'):
        sampling_start = 1
        sampling_end = no_sampling+1

output_file = file_folder + '/Summary_Local_Electronic_' + keyword + '_' + H_type + '.npy'
        
        
# Initialize an empty list to store the data
all_data = []   
for config in range(1,total_config+1):
 for sampling in range(sampling_start,sampling_end): 
    #print(sampling)
    # Specify the path to your CSV file
    #csv_file = file_folder + 'Local_Electronic/Local_Electroinc_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.csv'
    csv_file = file_folder + 'Local_Electronic/Local_Electronic_' + keyword + '_Config_' + str(config) +'-' + str(sampling) +'.csv'

    data = [] 
    # Open the CSV file for reading
    with open(csv_file, 'r') as file:
      csv_reader = csv.reader(file)
    
      # Skip the header row (if present)
      next(csv_reader)
    
      # Read each row of data and append it to the list
      for row in csv_reader:
        data.append(row)

    # Convert the data list to a NumPy array
    data = np.array(data)
    all_data.append(data)

all_data = np.array(all_data,dtype=float)
all_data[:,:,:6] *=1000
print(all_data.shape)
#print(all_data)
for i in range(3):
  print(np.std(all_data[:,i,:],axis=0).shape)
  print(np.std(all_data[:,i,:],axis=0))
np.save(output_file, all_data)



