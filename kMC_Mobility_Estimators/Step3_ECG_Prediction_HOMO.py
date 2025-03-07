import os,gc,sys,time
import numpy as np
import torch
import gpytorch
from gpytorch import settings as gpt_settings
from torch.utils.data import DataLoader, TensorDataset
from gpytorch.constraints import GreaterThan, Interval
from scipy.special import gamma
from copy import deepcopy
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from sklearn.mixture import GaussianMixture
gpt_settings.cholesky_jitter._set_value(1e-4,1e-4,None)
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('agg')  #agg backend is for writing to file, not for rendering in a window
import matplotlib.pyplot as plt


############### Define General System Variables/Parameters ###############
##### Comuting device/environment #####
GPU_index = 0
#random_seed = 1
#torch.manual_seed(random_seed)    # reproducible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if torch.cuda.is_available():
  torch.cuda.set_device(GPU_index)
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
  torch.set_default_tensor_type(torch.FloatTensor)



##### Setting about input & output files #####
filename_flag = sys.argv[1] #Provide number index of each snapshot
#filename_flag = "Feature_IBI_700K_Final_Unwrap"
file_folder = 'CG_NVT_' + sys.argv[2] + '_8Cells'  #type 700K or 555K
use_previous_model_to_train = 'yes' #'yes'
model_input = '/home/ciwang/DKL-ECG/700K/BS10K_LR002/'
#model_input = '/home/ciwang/DKL-ECG/555K/BS1K_LR0002_LS15/'
model_output = 'Model_State_GPR.pth'
#search_keyword = "Actual likelihood noise of the final model" 
search_keyword = "Actual likelihood noise"
model_likelihood = "Model_Evaluation.txt"

##### Path of ML dataset #####
#big_dir = ['/home/ciwang/Electronic_Coupling/ML_Dataset/700K', ('/home/ciwang/Electronic_Coupling/ML_Dataset/'+ filename_flag +'K/')]  #[traing, testing]
#big_dir = ['/home/ciwang/Electronic_Coupling/ML_Dataset/555K', ('/home/ciwang/Electronic_Coupling/ML_Dataset/'+ filename_flag +'K/')]  #[traing, testing]
big_dir = ['/home/ciwang/Electronic_Coupling/ML_Dataset/700K', ('./' + file_folder + '/ML_Feature') ]  #[traing, testing]
# these files should be contained in big_dir
fn_data = ['Feature_MO_Energy_CG_Improve_25_Training_COM.npy', 'Label_Energy_Sample_182098.npy',  # 700K Training
#fn_data = ['Feature_MO_Energy_CG_Improve_25_Training_COM.npy', 'Label_Energy_Sample_132081.npy', # 555K Training
           #'Feature_MO_Energy_CG_Improve_25_Testing_COM.npy', 'Label_Energy_1_10240.npy']         # 700K/555K Testing
           ('Feature_MO_Energy_'+ sys.argv[2] +'_Config_' + filename_flag +'.npy'), 'Label_Energy_1_10240.npy']         # 700K/555K Testing

output_dir = './' + file_folder + '/ECG_Results/'
test_pred_filename = 'HOMO_Energy_' + sys.argv[2] + '_Config_' + filename_flag +'.txt'

##### ML setup parameters #####
n_train = 30000   # size of training dataset
n_batch = 5000    # batch size
out_dim = 6      # latent feature size
k_fold = 5       # number of cross-validation steps
lambda_kl = 1e-4 # kl regularization term (currently turn off)
lr = 0.002       # learning rate
n_steps = 1000 # number of training epoach
n_induc = 2000   # number of GPR inducing points/grid size


#torch.cuda.set_device(GPU_index)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
#torch.set_default_tensor_type(torch.cuda.FloatTensor)
#torch.set_default_tensor_type(torch.FloatTensor)

def main():   
    print(device, flush=True)
    ############### Specify Output Files ###############
    #output_dir = "Train100K_BS5K_LR0001_LeakyReLu"
    #if not os.path.isdir(output_dir):
    #  os.mkdir(output_dir)
    #output_dir = output_dir + "/"
    
  
    ############### Define Model (Hyper)parameters ###############
    # regularization currently turned off in vdkl_experiment
    param_job = np.array([n_batch, out_dim, k_fold, lambda_kl, lr, n_steps, n_induc])
  
    ############### Machine Learning ###############
    # vdkl_experiment saves various files to a further subdirectory big_dir/CV_test
    vdkl_experiment(n_train, big_dir, fn_data, param_job,'_ur', output_dir)
    #input("Press Enter to continue...")



def init_custom(m):
    if type(m) == torch.nn.Linear:
        bound = 1/np.sqrt(m.in_features)
        torch.nn.init.normal_(m.weight, 0, bound)
        # torch.nn.init.zeros_(m.bias)

# DKL neural network structure
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, out_dim):
        super(LargeFeatureExtractor, self).__init__()
        # droprate = 0.35
        # self.add_module('linear1', torch.nn.Identity())
        self.add_module('linear1', torch.nn.Linear(data_dim, 80))
        self.add_module('drop1', torch.nn.BatchNorm1d(80))
        #self.add_module('relu1', torch.nn.ELU())
        self.add_module('relu1', torch.nn.LeakyReLU(negative_slope=0.01))
        self.add_module('linear2', torch.nn.Linear(80,40))
        self.add_module('drop2', torch.nn.BatchNorm1d(40))
        #self.add_module('relu2', torch.nn.ELU())
        self.add_module('relu2', torch.nn.LeakyReLU(negative_slope=0.01))
        self.add_module('linear3', torch.nn.Linear(40, 20))
        self.add_module('drop3', torch.nn.BatchNorm1d(20))
        #self.add_module('relu3', torch.nn.ELU())
        self.add_module('relu3', torch.nn.LeakyReLU(negative_slope=0.01))
        self.add_module('linear4', torch.nn.Linear(20, 10))
        self.add_module('drop4', torch.nn.BatchNorm1d(10))
        #self.add_module('relu4', torch.nn.ELU())
        self.add_module('relu4', torch.nn.LeakyReLU(negative_slope=0.01))
        self.add_module('linear5', torch.nn.Linear(10, out_dim))
        self.add_module('drop5', torch.nn.BatchNorm1d(out_dim))
        #self.add_module('relu5', torch.nn.ELU())
        self.add_module('relu5', torch.nn.LeakyReLU(negative_slope=0.01))
        self.apply(init_custom)
        # self.add_module('relu5', torch.nn.Sigmoid())

# DKL output -> mean in latent space
class VariationalMean(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(VariationalMean, self).__init__()
        # droprate = 0.35
        # self.add_module('linear1', torch.nn.Identity())
        self.add_module('linear1', torch.nn.Linear(data_dim, data_dim))
        # self.add_module('relu1', torch.nn.Tanh())
        # self.add_module('linear2', torch.nn.Linear(data_dim, data_dim))
        self.apply(init_custom)
        with torch.no_grad():
            self[0].weight = torch.nn.Parameter(torch.eye(data_dim))
        # self.add_module('relu5', torch.nn.Sigmoid())

# DKL output -> variance in latent space
class VariationalVar(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(VariationalVar, self).__init__()
        # droprate = 0.35
        # self.add_module('linear1', torch.nn.Identity())
        self.add_module('linear1', torch.nn.Linear(data_dim, data_dim))
        self.add_module('relu1', torch.nn.Sigmoid())
        self.add_module('linear2', torch.nn.Linear(data_dim, 1))
        #self.add_module('linear2', torch.nn.Linear(data_dim, data_dim))
        self.apply(init_custom)
        torch.nn.init.constant_(self[-1].bias,-2) # Initial bias to make inital variance reasonable
        # self.add_module('relu5', torch.nn.Sigmoid())

# Variational layer after DKL projection
class VAELayer(torch.nn.Module):
    def __init__(self, data_dim):
        super(VAELayer, self).__init__()
        
        self.mu = VariationalMean(data_dim)
        self.logvar = VariationalVar(data_dim)
        
    
    # Layer between the feature extraction and the GPR
    
    def encode(self, x):
        return self.mu(x), self.logvar(x)
        # mu = self.mu(x)
        # return mu, self.logvar_lazy*torch.ones_like(mu)
    
    # Samples the latent distribution to send to GP
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu)*torch.exp(0.5*logvar)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        #print(logvar.detach().numpy().shape)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ApproximateGPLayer(gpytorch.models.ApproximateGP):
        def __init__(self, num_dim, grid_size):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=grid_size)
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, torch.randn((grid_size,num_dim)),
                variational_distribution=variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            if num_dim < 2:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            else:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=num_dim))
            
        
        def forward(self, x):
            # print(x.shape)
            mean = self.mean_module(x)
            # print(mean.shape)
            covar = self.covar_module(x)
            # print(covar.shape)
            return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLVAEModel(gpytorch.Module):
    #def __init__(self, in_dim, num_dim, grid_size=2000):
    def __init__(self, in_dim, num_dim, n_induc):
        super(DKLVAEModel, self).__init__()
        
        # DKL layer
        self.feature_extractor = LargeFeatureExtractor(in_dim, num_dim)
        # variational layer
        self.vae_layer = VAELayer(num_dim)
        # GPR on the latent projection
        self.gp_layer = ApproximateGPLayer(num_dim=num_dim, grid_size=n_induc)
        self.num_dim = num_dim

    def forward(self, x):
        lat = self.feature_extractor(x)
        feat, mu, logvar = self.vae_layer(lat)
        res = self.gp_layer(feat)
        # res is the gpr prediction on the provided points
        # mu is the mean of the latent space projection
        # logvar is the log variance of the latent space distribution
        # feat is the stochastic sample of the latent space distributions
        return res, mu, logvar, feat

def prior_exp(x):
    out_dim = len(x[0,:])
    n_0 = gamma(out_dim/2)/(2*np.pi**(out_dim/2)*gamma(out_dim))
    return n_0*torch.exp(-torch.sqrt((x**2,1)))


# This is the used KL divergence function 
# Sometimes unstable on scruggs
def kl_vae_qp(x, mu, logvar):
    out_dim = len(x[0,:])
    var = torch.exp(logvar[:,0])
    # dim_out = len(x[0])
    diff_tensor = torch.zeros((out_dim,len(x[:,0]),len(mu[:,0])))
    
    # Squared difference tensor between the given sample points x and the center of the given distributions
    for i in range(out_dim):
        diff_tensor[i] = (x[:,i:(i+1)] - mu[:,i:(i+1)].T)**2
    
    # Applies the exponential to the distance tensor
    diff_tensor = torch.exp(-.5*torch.sum(diff_tensor,0)/var)/var**(out_dim/2)
    
    #Sums the exponentials over the batch and normalizes
    pdf_x = torch.sum(diff_tensor,1)/len(x[:,0])/(2*np.pi)**(out_dim/2)
    
    # pdf_prior = prior_pdf(x)
    pdf_prior = torch.exp(-.5*torch.sum(x**2,1))/(2*np.pi)**(out_dim/2)
    # pdf_prior = prior_exp(x)
    # print(pdf_x)
    # print(pdf_prior)
    # Returns the batch sum, which is the approximate KL divergence 
    return torch.sum(torch.log( pdf_x / pdf_prior ))

# Take in train/test data, return full vdkl fit and gmm clustering analysis
# dn is working directory, f_ext added to data files to distinguish
def vdkl_trial(train_x, train_y, test_x, params, dn, f_ext, train_mean, output_dir):
    ############### Specify Hyperparameters ###############
    n_batch = int(params[0])   # batch size
    out_dim = int(params[1])   # l atent feature size
    k_fold = int(params[2])    # number of cross-validation steps
    lambda_kl = params[3]      # kl regularization term (currently turn off)
    lr = params[4]             # learning rate
    n_steps = int(params[5])   # number of training epoach
    n_induc = int(params[6])   # number of GPR inducing points/grid size
    dim_d = len(train_x[0,:])  # dimention of input feature
    n_train = len(train_x[:,0])# size of training data
    
    
    ############### Setup ML Model ###############
    n_train_sub = int(n_train*(k_fold-1)//k_fold)
    if k_fold == 1:
        n_train_sub = n_train
    
    #model_list = [DKLVAEModel(dim_d,out_dim,grid_size=n_induc) for k in range(k_fold)]
    model = DKLVAEModel(dim_d,out_dim,n_induc)
    #mll_list = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model.gp_layer, num_data=n_train_sub)
    #print(model.state_dict())
    #print(model_list[0].gp_layer.state_dict()['variational_strategy.inducing_points'])
        
    
    ##### Use the model parameters which has been trained #####
    if (use_previous_model_to_train=='yes'):
      if os.path.exists((model_input+model_output)):
        if torch.cuda.is_available():
          model.load_state_dict(torch.load(model_input+model_output)) 
        else:
          model.load_state_dict(torch.load((model_input+model_output),map_location=torch.device('cpu')))  
      else:
        raise ValueError('%s does not exit!' % (model_input+model_output))
    #print(model.state_dict())
    

    ##### Read the likelihood noise which has been trained #####
    if (use_previous_model_to_train=='yes'):
      if os.path.exists((model_input+model_likelihood)):
        f = open((model_input+model_likelihood), "r")
        full_content = f.readlines()
        full_content_numpy = np.array(full_content)
        f.close()
        for line in range(len(full_content)):
          if (full_content[line].__contains__(search_keyword)):
            likelihood_noise = float(full_content[line].split()[-1])
            #print(line,likelihood_noise)
      else:
        raise ValueError('%s does not exit!' % (model_input+model_likelihood))  
    
    hypers = {'noise': torch.tensor(likelihood_noise)}
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.initialize(**hypers)

    print("\n ------------------------------ ML Parameters ------------------------------ \n")
    print("Raw lengthe scale:")
    print(model.gp_layer.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0])
    print("Actual lengthe scale:")
    print(model.gp_layer.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0]) 
    print("\nRaw outputscale: %s" %(model.gp_layer.covar_module.raw_outputscale.item()))
    print("Actual outputscale: %s" %(model.gp_layer.covar_module.outputscale.item()))
    print("\nRaw likelihood noise: %s" %(likelihood.raw_noise.item()))
    print("Actual likelihood noise %s" %(likelihood.noise.item()))
    
    
    
    ############### Model Evaluation On Testing Dataset ###############
    model.eval()
    likelihood.eval()
    ##### Assign computing environment#####
    #if torch.cuda.is_available():
    #  test_x = test_x.cuda()
    #  test_y = test_y.cuda()
    #  train_x = train_x.cuda()
    #  train_y = train_y.cuda()
    #  model = model.cuda()
    #  likelihood = likelihood.cuda()
    
    ##### Evaluates the learned model on the test and train data ##### 
    with torch.no_grad(), gpt_settings.fast_pred_var(),gpt_settings.max_cg_iterations(2500):
        # vdkl projection of test data
        _, test_vae, _ = model.vae_layer(model.feature_extractor(test_x))
        # gpr on test data
        model_test = model.gp_layer(test_vae)
        # posterior probability (predictive distribution)
        pos_pred_2 = likelihood(model_test)
        pos_train_mean = pos_pred_2.mean.flatten()
        
        # basic prediction results on test data
        # observed value, pred mean, pred stdev
        data_out = np.zeros((test_x.shape[0],2))
        data_out[:,0] = (pos_train_mean+train_mean).cpu()#Unit in eV
        data_out[:,1] = pos_pred_2.stddev.flatten().cpu() #Unit in eV
        filename = test_pred_filename
        np.savetxt((output_dir+filename), data_out) #data_out[:,1:])
        
        
        



def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))


def vdkl_experiment(n_train, dn, fn_list, params, f_ext, output_dir):
    ############### Loading Training Dataset ###############
    # Loads the distance matrix data and normalizes the input vectors
    fname_x = dn[0]+"/"+fn_list[0]
    fname_y = dn[0]+"/"+fn_list[1]
    #data_x = torch.Tensor(np.loadtxt(fname_x,delimiter=','))
    data_x = torch.Tensor(np.load(fname_x))
    
    # standardize training data
    train_xmean = torch.mean(data_x,0)
    train_std = torch.std(data_x,0)
    data_x -= train_xmean
    data_x /= train_std
    #data_y = torch.Tensor(np.loadtxt(fname_y))
    data_y = torch.Tensor(np.load(fname_y)[:,0]) # 0:HOMO 1:LUMO 2:Band Gap
    data_y = data_y * 27.211386245988 # change unit from Hatree to eV
    train_mean = torch.mean(data_y)
    data_y -= train_mean
    
    # train_perm is the random permutation to divide training and test data
    train_perm = np.random.permutation(len(data_y))
    train_x = data_x[train_perm[:n_train],:]
    train_y = data_y[train_perm[:n_train]]
    
    
    ############### Loading Testing Dataset ###############
    fname_xt = dn[1]+"/"+fn_list[2]
    #fname_yt = dn[1]+"/"+fn_list[3]
    test_x = torch.Tensor(np.load(fname_xt))
    #test_y = torch.Tensor(np.load(fname_yt)[:,0]) # 0:HOMO 1:LUMO 2:Band Gap
    #test_y = test_y * 27.211386245988 # change unit from Hatree to eV
    
    # Apply same standardization to test set
    test_x -= train_xmean
    test_x /= train_std
    #test_y -= train_mean
    
    
    
    #################### Print Overal Model Information ##################
    print(' ------------------------------ File Information ------------------------------ \n')
    print('Path of feature file for training: %s\n' %(fn_list[0]))
    print('Path of label file for training: %s\n' %(fn_list[1]))
    print('Path of feature file for testing: %s\n' %(fn_list[2]))
    print('Path of label file for testing: %s\n' %(fn_list[3]))
    print('\n ------------------------------ Dataset Information ------------------------------ \n')
    print('Total number of data samples : %s + %s\n' %(data_x.size()[0],test_x.size()[0]))
    print('Size of trainining dataset: %s (from %s samples)\n' %(n_train,data_x.size()[0]))
    print('Size of testing/validation dataset: %s\n' %(test_x.size()[0]))
    print('Inpute feature dimension: %s\n' %(data_x.size()[1]))
    print('Mean & standard deviation of training dataset: %8.5f +- %8.5f\n' %(train_mean.cpu().item(), train_y.std().cpu().item())) 
    print('\n ------------------------------ Model Setup Parameter ------------------------------ \n')
    print('Batch size: %d\n' %(params[0])) 
    print('Latent feature dimension: %d\n' %(params[1])) 
    print('Number of cross-validation: %d\n' %(params[2]))
    print('Learning rate: %s\n' %(params[4]))
    print('Number of epoch: %s\n' %(int(params[5]))) 
    print('Number of GPR inducing points (or grid size): %s\n' %(int(params[6]))) 
    
    
    #################### Training Model ##################
    #val_trial, data_trial, feat_trial, gmm_trial = vdkl_trial(train_x, train_y, test_x, test_y, params, dn, f_ext, train_mean)
    vdkl_trial(train_x, train_y, test_x, params, dn, f_ext, train_mean, output_dir)
    


if __name__ == '__main__':
    main()

