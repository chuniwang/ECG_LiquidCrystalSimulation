import os,gc,sys,time
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
#matplotlib.use('agg')  #agg backend is for writing to file, not for rendering in a window
import matplotlib.pyplot as plt
import sys



##### Setting about input & output files #####
keyword = sys.argv[1] 
input_path = './' # Folder contains information of model performation 
output_dir = './'   # Fodler saves the visualized results 
#big_dir = './ML_Dataset/700K/Label_Energy_Sample_138730.npy' # Folder keeps ML dataset 
big_dir = '/home/ciwang/Electronic_Coupling/ML_Dataset/700K/Label_Energy_Sample_182098.npy' # Folder keeps ML dataset 
figure_dpi = 250
if (len(keyword)>1):
  figurename_scatter = 'Scatter_HOMO' + keyword + '.png'
  figurename_histo = 'Histogram_Energy' + keyword + '.png'
else:
  figurename_scatter = 'Scatter_HOMO.png'
  figurename_histo = 'Histogram_Energy.png'
figurename_loss = 'Loss.png'
figurename_noise = 'Likelihood_Noise.png'
bin_size = 0.02 # bin size for histogram !!! unit in eV !!!



#################### Read Information of Model Performance ####################
if (len(keyword)>1):
  data_out = np.loadtxt((input_path+'Predict_Mean_Testing' + keyword + '.txt'))
else:
  data_out = np.loadtxt((input_path+'Predict_Mean_Testing.txt'))
DKL_loss = np.load((input_path+"DKL_Loss.npy"))
train_y = np.load(big_dir)[:,0]
likelihood_noise = np.loadtxt((input_path+'Likelihood_Noise.txt'))

k_fold = DKL_loss.shape[0]
n_steps = DKL_loss.shape[1]
#print(DKL_loss.shape)

train_mean = train_y.mean()
#print(train_y.shape)



#################### Estimate Model Performance ####################
correlation_matrix = np.corrcoef(data_out[:,0], data_out[:,1])
r_2 = np.corrcoef(data_out[:,0], data_out[:,1])[0, 1]**2
MAE = np.mean(np.abs(data_out[:,0] - data_out[:,1]))
RMSE = np.power(np.mean((data_out[:,0] - data_out[:,1])**2), 0.5)
        
print('\n ------------------------------ Model Evaluation (Unit In Hartree) ------------------------------ \n')
print('Mean & standard deviation of testing dataset: %8.5f +- %8.5f\n' %(data_out[:,0].mean(),data_out[:,0].std()))
print('Averge over GPR predicted mean of testing dataset: %8.5f\n' %(data_out[:,1].mean()))
print('Averge over GPR predicted standard deviation of testing dataset: %8.5f\n' %(data_out[:,2].mean()))
print('R square: %10.7f\n' %(r_2))
print('MAE: %10.7f\n' %(MAE))
print('RMSE: %10.7f\n' %(RMSE))


#################### Scater Plot: Test Dataset ####################
plt.figure(1, figsize=(9.5, 9), dpi=figure_dpi)
x = np.linspace(-1000,1000,100)
y = x
plt.plot(x, y, color="darkgrey", linewidth=1.0, linestyle="--")

##### Calculate the point density #####
x, y = data_out[:,0], data_out[:,1]
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

##### Sort the points by density, so the densest points are plotted last #####
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x * 27.211386245988, y * 27.211386245988, c=z, s=25, cmap='jet')   #edgecolor=''
#cb = plt.colorbar()
#cb.set_label('Gaussian Kernel Density', fontsize=30)

plt.text(-6.6, -7.7, "$R^2$: %2.5f\nMAE: %2.4f\nRMSE: %2.4f" %(r_2, MAE, RMSE), fontsize=40)
#plt.text(-0.244 * 27.211386245988, -0.288 * 27.211386245988, "$R^2$: %2.5f" %(r_2), fontsize=40)

#plt.title("TEST: Electronic Coupling, V [meV]", fontsize=35)
plt.xlabel('$\mathregular{HOMO^{QC}}$ [eV]', fontsize=35)
plt.ylabel('$\mathregular{HOMO^{ML}}$ [eV]', fontsize=35)
plt.xlim(-0.24 * 27.211386245988, -0.30 * 27.211386245988)
plt.ylim(-0.24 * 27.211386245988, -0.30 * 27.211386245988)

plt.xticks(np.arange(6.6, 8.0, step=0.2)*(-1), fontsize=24, fontname="Arial") #rotation=90
plt.tick_params(labelsize = 24)
#plt.grid(True)
plt.tight_layout()
plt.savefig((output_dir+figurename_scatter), dpi=figure_dpi,format="png")
#plt.show()
plt.close()


#################### Plot Profiles of Loss Functions ####################
plt.figure(figsize=(20,10), dpi=figure_dpi)
x_min = 0
x_max = n_steps
l_train = DKL_loss[:,:,0]
l_val = DKL_loss[:,:,1]
epoach = np.arange(n_steps,dtype=int)
for k in range(k_fold):
  label_name = str(k) + "-fold: Training"
  plt.plot(epoach, l_train[k,:], label=label_name)
  label_name = str(k) + "-fold: Validating"
  plt.scatter(epoach, l_train[k,:], label=label_name)
plt.legend(frameon=False, fontsize=24, loc='upper right')
plt.xlabel('Epoach', fontsize=28, fontname="Arial")
plt.ylabel('Loss', fontsize=28, fontname="Arial")
plt.xticks(fontsize=24, fontname="Arial")
plt.yticks(fontsize=24, fontname="Arial")
plt.savefig((output_dir+figurename_loss), dpi=figure_dpi,format="png")
#plt.show()
plt.close()


#################### Plot Profiles of Likelihood Noise ####################
plt.figure(figsize=(20,10), dpi=figure_dpi)
x_min = 0
x_max = n_steps
epoach = np.arange(n_steps,dtype=int)
for k in range(k_fold):
  label_name = str(k) + "-fold"
  #plt.plot(epoach, np.log10(likelihood_noise[:,k]), label=label_name)
  plt.plot(epoach, likelihood_noise[:,k], label=label_name)
plt.legend(frameon=False, fontsize=24, loc='upper right')
plt.xlabel('Epoach', fontsize=28, fontname="Arial")
plt.ylabel('Likelihood noise', fontsize=28, fontname="Arial")
plt.xticks(fontsize=24, fontname="Arial")
plt.yticks(fontsize=24, fontname="Arial")
plt.savefig((output_dir+figurename_noise), dpi=figure_dpi,format="png")
plt.show()
plt.close()


#################### Plot Probability Density Distribution ####################
##### Specify bining boundary #####
up_bound = np.array([(train_y.max()), data_out[:,0].max(), data_out[:,1].max()])
up_bound = up_bound.max() * 27.211386245988 #change unit from hartree to eV
low_bound = np.array([(train_y.min()), data_out[:,0].min(), data_out[:,1].min()])
low_bound = low_bound.min() * 27.211386245988 #change unit from hartree to eV
bins = np.arange(round(low_bound,1), round(up_bound,1), step=bin_size)

##### Histograming #####
train_hist, bins = np.histogram(((train_y)*27.211386245988), bins=bins, density=True)
test_hist, bins = np.histogram((data_out[:,0]*27.211386245988), bins=bins, density=True)
pred_hist, bins = np.histogram((data_out[:,1]*27.211386245988), bins=bins, density=True)

##### Calculate P(E) based on the predicted mean & noise #####
no_data = data_out[:,1].shape[0]
E = np.linspace((low_bound-5),(up_bound+5),200)
P_E = np.zeros(E.shape[0])
for i in range(E.shape[0]):
  P_i = np.zeros(no_data)
  for n in range(no_data):
    mu = data_out[n,1] * 27.211386245988
    sigma = data_out[n,2]* 27.211386245988
    P_i[n] = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp( - (E[i] - mu)**2 / (2 * sigma**2))
  P_E[i] = P_i.sum()/no_data


##### Make histogram plot #####
plt.figure(figsize=(10, 10), dpi=figure_dpi)
x_min = round(low_bound) - 0.2
x_max = round(up_bound) + 0.2
y_max = np.array([train_hist.max(),test_hist.max(),pred_hist.max()]).max()
#y_max = np.array([train_hist.max(),test_hist.max()]).max()
if (y_max > 75):
  y_max = 100.0
else:
  y_max = y_max * 1.3

plt.hist(bins[:-1], bins, weights=train_hist, color="b", alpha=0.6, label='Training data')
plt.hist(bins[:-1], bins, weights=test_hist, color="grey", alpha=0.8, label='Testing data')
plt.hist(bins[:-1], bins, weights=pred_hist, color="green", alpha=0.5, label='Predicted meam')
#plt.plot(E, gaussian_form, linewidth=2, color="indianred", linestyle="--")
plt.plot(E, P_E, linewidth=4, color="black", linestyle="--",label='P(E|R), ML prediction on testing data')

plt.xticks(np.arange(x_min, x_max+1, step=0.2), fontsize=24, fontname="Arial") #rotation=90
plt.yticks(fontsize=24, fontname="Arial")
plt.xlabel('HOMO Energy [eV]', fontsize=28, fontname="Arial")
plt.ylabel('Probability density', fontsize=28, fontname="Arial")
plt.legend(frameon=False, fontsize=24, loc='upper left')
plt.xlim(x_min, x_max)
plt.ylim(0, y_max)
plt.savefig((output_dir+figurename_histo), dpi=figure_dpi,format="png")
plt.show()
plt.close()
