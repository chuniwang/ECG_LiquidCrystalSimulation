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
keyword_list = ['515K', '555K', '700K']
#keyword_list = ['515K']
colors = ['#FFC000','#1F4E79','#548235' ]
labels = ['Smectic', 'Nematic', 'Isotropic']

traj_path = ['Trajectory_1/', 'Trajectory_2/', 'Trajectory_3/', './Trajectory_4/', './Trajectory_5/',
             'Trajectory_6/', 'Trajectory_7/', 'Trajectory_8/', './Trajectory_9/', './Trajectory_10/']
#traj_path = ['Trajectory_1/','Trajectory_2/']
traj_path = ['Trajectory_1/']

figurename_histo_1 = 'Stenhardt_OP_Distribution_q4.png'
figurename_histo_2 = 'Stenhardt_OP_Distribution_q6.png'
figurename_scatter = ['Stenhardt_OP_Scatter_Smectic.png', 'Stenhardt_OP_Scatter_Nematic.png','Stenhardt_OP_Scatter_Isotropic.png']
outputfilename = ['Stenhardt_OP_Smectic.npy', 'Stenhardt_OP_Nematic.npy', 'Stenhardt_OP_Isotropic.npy']

make_histo = 'no'
make_scatter = 'yes'
no_frame = 12 #10 #Number of snapshot in each trajectory folder for order parameter calculation

figure_dpi = 500
topology_path = 'CG_topology.gro'
no_atom_per_mol = 15 #Number of particles in a molecule
cutoff_neighbor_dis = 13.0
no_compute_particle = 15

##### Specify bining boundary #####
bin_size = 0.02 # bin size for histogram !!! unit in eV !!!
up_bound = 1.02
low_bound = 0.0
bins = np.arange(low_bound, up_bound, step=bin_size)
bin_centers = 0.5*(bins[1:] + bins[:-1])


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

LC_q4 = []
LC_q6 = []
LC_q4_hist = []
LC_q6_hist = []
for keyword in keyword_list:
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
  print('\n----- %s -----' %keyword)
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


  q4 = []
  q6 = []
  all_q4_hist = []
  all_q6_hist = []
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
      mass_matrix = np.full((total_no_mol,no_compute_particle), mol_mass[:no_compute_particle])
      ### Center of mass ###
      COM = np.zeros((total_no_mol,3))
      mx = mass_matrix*traj_pbc[:,:no_compute_particle,0]
      my = mass_matrix*traj_pbc[:,:no_compute_particle,1]
      mz = mass_matrix*traj_pbc[:,:no_compute_particle,2]
      COM[:,0] = np.sum(mx, axis=1)/mol_mass.sum()
      COM[:,1] = np.sum(my, axis=1)/mol_mass.sum()
      COM[:,2] = np.sum(mz, axis=1)/mol_mass.sum()
      ### Deal with periodic boundary condition: COM ###
      COM = np.where((COM>box_size), (COM-box_size), COM)
    
      ##### Steinhardt order parameter #####
      ### Define Freud system ###
      freud_box = freud.box.Box.cube(box_size)
      freud_system = (freud_box,(COM-0.5*box_size))
      #print(freud_system)
    
      ### Compute q4 ###
      L=4
      ql = freud.order.Steinhardt(L,average=True)
      #ql_sc = ql.compute(freud_system, neighbors={"num_neighbors": 6}).particle_order
      #ql_sc = ql.compute(freud_system, neighbors={"num_neighbors": len(charge_neighbor_index)}).particle_order
      ql_sc = ql.compute(freud_system, {'r_max':cutoff_neighbor_dis}).particle_order 
      mean_sc = np.nanmean(ql_sc)
      std_sc = np.nanstd(ql_sc)
      #print(ql_sc.shape)
      q4.append(ql_sc)
      #print("The Q{} values computed for simple cubic are {:.3f} +/- {:.3e}".format(L, mean_sc, std_sc))

      ### Compute q4 distribution ###
      q4_hist, bins = np.histogram(ql_sc, bins=bins, density=True) 
      q4_hist = q4_hist*100/q4_hist.sum()  
      all_q4_hist.append(q4_hist)

      ### Compute q6 ###
      L=6
      ql = freud.order.Steinhardt(L,average=True)
      #ql_sc = ql.compute(freud_system, neighbors={"num_neighbors": 6}).particle_order
      #ql_sc = ql.compute(freud_system, neighbors={"num_neighbors": len(charge_neighbor_index)}).particle_order
      ql_sc = ql.compute(freud_system, {'r_max':cutoff_neighbor_dis}).particle_order 
      mean_sc = np.nanmean(ql_sc)
      std_sc = np.nanstd(ql_sc)
      #print(ql_sc.shape)
      q6.append(ql_sc)
      print("The Q{} values computed for simple cubic are {:.3f} +/- {:.3e}".format(L, mean_sc, std_sc))
    
      ### Compute q6 distribution ###
      q6_hist, bins = np.histogram(ql_sc, bins=bins, density=True) 
      q6_hist = q6_hist*100/q6_hist.sum()
      all_q6_hist.append(q6_hist)

  q4 = np.array(q4)
  q6 = np.array(q6)
  all_q4_hist = np.array(all_q4_hist)
  all_q6_hist = np.array(all_q6_hist)
  LC_q4.append(q4)
  LC_q6.append(q6)
  LC_q4_hist.append(all_q4_hist)
  LC_q6_hist.append(all_q6_hist)

LC_q4 = np.array(LC_q4)
LC_q6 = np.array(LC_q6)
LC_q4_hist = np.array(LC_q4_hist)
LC_q6_hist = np.array(LC_q6_hist)
print(LC_q4.shape)
print(LC_q4_hist.shape)


##################### Save files #####################
for phase in range(len(keyword_list)):
  data = np.zeros((LC_q4.shape[1],LC_q4.shape[2],2))
  data[:,:,0] = LC_q4[phase,:,:]
  data[:,:,1] = LC_q6[phase,:,:]
  np.save(outputfilename[phase],data)
  print(outputfilename[phase],data.shape)


if (make_histo == 'yes'):
  ##################### Setup histogram plot #####################
  x_min = low_bound
  x_max = 0.5 #up_bound
  y_max = 40
  
  ###################### Calculate q4 distribution ######################
  plt.figure(figsize=(10, 10), dpi=figure_dpi)
  for phase in range(len(keyword_list)):
    q4_hist_mean = np.mean(LC_q4_hist[phase,:,:],axis=0)
    q4_hist_std = np.std(LC_q4_hist[phase,:,:],axis=0) 
    #q6_hist_mean = np.mean(LC_q6_hist[phase,:,:],axis=0)
    #q6_hist_std = np.std(LC_q6_hist[phase,:,:],axis=0)

    ##################### Make histogram plot #####################
    plt.hist(bins[:-1], bins, weights=q4_hist_mean, color=colors[phase], alpha=0.6, label=labels[phase])

    ########## Set up error bar #########
    plt.errorbar(
      bin_centers,
      q4_hist_mean,
      yerr = q4_hist_std,
      #marker = '.',
      drawstyle = 'steps-mid',
      color=colors[phase], 
      alpha=0.8,
      ecolor=colors[phase],
      lw=1.5, capsize=3, capthick=1.5 
      #lw=2.5, capsize=4, capthick=2.5
      )

  plt.xticks(np.arange(0, 0.55, step=0.05), fontsize=24, fontname="Arial", rotation=45)
  plt.yticks(fontsize=24, fontname="Arial")
  plt.xlabel('$\overline{q}_4$', fontsize=32, fontname="Arial")
  plt.ylabel('Probability [%]', fontsize=32, fontname="Arial")
  plt.legend(frameon=False, fontsize=32, loc='upper right')
  plt.xlim(x_min, x_max)
  plt.ylim(0, y_max)
  plt.savefig((figurename_histo_1), dpi=figure_dpi,format="png", bbox_inches="tight")
  #plt.show()
  plt.close()  

  ###################### Calculate q6 distribution ######################
  plt.figure(figsize=(10, 10), dpi=figure_dpi)
  for phase in range(len(keyword_list)):
    q6_hist_mean = np.mean(LC_q6_hist[phase,:,:],axis=0)
    q6_hist_std = np.std(LC_q6_hist[phase,:,:],axis=0)

    ##################### Make histogram plot #####################
    plt.hist(bins[:-1], bins, weights=q6_hist_mean, color=colors[phase], alpha=0.6, label=labels[phase])

    ########## Set up error bar #########
    plt.errorbar(
      bin_centers,
      q6_hist_mean,
      yerr = q6_hist_std,
      #marker = '.',
      drawstyle = 'steps-mid',
      color=colors[phase], 
      alpha=0.8,
      ecolor=colors[phase],
      lw=1.5, capsize=3, capthick=1.5 
      #lw=2.5, capsize=4, capthick=2.5
      )

  plt.xticks(np.arange(0, 0.55, step=0.05), fontsize=24, fontname="Arial", rotation=45)
  plt.yticks(fontsize=24, fontname="Arial")
  plt.xlabel('$\overline{q}_6$', fontsize=32, fontname="Arial")
  plt.ylabel('Probability [%]', fontsize=32, fontname="Arial")
  plt.legend(frameon=False, fontsize=32, loc='upper right')
  plt.xlim(x_min, x_max)
  plt.ylim(0, y_max)
  plt.savefig((figurename_histo_2), dpi=figure_dpi,format="png", bbox_inches="tight")
  #plt.show()
  plt.close()  


if (make_scatter == 'yes'):
  for i in range(len(keyword_list)):
      ##################### Setup histogram plot #####################
      plt.figure(figsize = (10, 10), dpi=figure_dpi)
      ##### Create a boolean mask for NaN values #####
      nan_mask = np.isnan(LC_q4[i,:,:].flatten())
      ##### Calculate the point density #####
      x, y = LC_q4[i,:,:].flatten()[~nan_mask], LC_q6[i,:,:].flatten()[~nan_mask]
      xy = np.vstack([x,y])
      z = gaussian_kde(xy)(xy) 
        
      ##### Sort the points by density, so the densest points are plotted last #####
      idx = z.argsort()
      x, y, z = x[idx], y[idx], z[idx]
      plt.scatter(x, y, c=z, s=25, cmap='viridis')   #edgecolor=''
      #plt.scatter(figure_data[:,1], figure_data[:,0])   #edgecolor=''

      plt.xlabel('$\overline{q}_4$', fontsize=36)
      plt.xticks(np.arange(0, 0.6, step=0.05), fontsize=28, fontname="Arial", rotation=45)
      plt.xlim(0, 0.55)

      plt.ylabel('$\overline{q}_6$', fontsize=36)
      plt.ylim(0, 0.55)
      plt.yticks(np.arange(0, 0.6, step=0.05), fontsize=28, fontname="Arial")
      #plt.tick_params(axis='both', which='major', length=12)  # Adjust the length for major ticks
      plt.savefig(figurename_scatter[i], format="png", bbox_inches="tight")
      #plt.show()
      plt.close()
