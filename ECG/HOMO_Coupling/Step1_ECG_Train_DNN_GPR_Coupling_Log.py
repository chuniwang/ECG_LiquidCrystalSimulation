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
gpt_settings.cholesky_jitter._set_value(1e-2,1e-4,None)
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('agg')  #agg backend is for writing to file, not for rendering in a window
import matplotlib.pyplot as plt


############### Define General System Variables/Parameters ###############
##### Comuting device/environment #####
# ind_job = int(sys.argv[1])-1
# ind_job=5
GPU_index = 2
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = str(ind_job%4)
#os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_index)
#os.environ["OMP_NUM_THREADS"] = "2"
torch.cuda.set_device(GPU_index)
#random_seed = 1
#torch.manual_seed(random_seed)    # reproducible

##### Setting about input & output files #####
use_previous_model_to_train = 'no' #'yes'
model_input = 'Model_State.pth'
model_output = 'Model_State_out.pth'

##### Path of ML dataset #####
big_dir = '/home/ciwang/Electronic_Coupling/ML_Dataset/555K' 
# these files should be contained in big_dir
fn_data = ['Feature_Coupling_CG_Improve_25_Inter_Training_COM.npy', 'Label_Coupling_Sample_92160.npy',
           'Feature_Coupling_CG_Improve_25_Inter_Testing_COM.npy', 'Label_Coupling_1_10240.npy']


##### ML setup parameters #####
n_train = 90000   # size of training dataset
n_batch = 5000    # batch size
out_dim = 6      # latent feature size
k_fold = 5       # number of cross-validation steps
lambda_kl = 1e-4 # kl regularization term (currently turn off)
lr = 0.02       # learning rate
n_steps = 4000 # number of training epoach
n_induc = 2000   # number of GPR inducing points/grid size


############### Define Data Format: GPU or CPU ###############
#torch.cuda.set_device(GPU_index)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
#torch.set_default_tensor_type(torch.FloatTensor)

def main():   
    #print(device, flush=True)
    ############### Specify Output Files ###############
    output_dir = "./"
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



"""
############### Print Memory Usage ###############
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(GPU_index)
GPU_info = nvmlDeviceGetMemoryInfo(handle)
print('------------------------------ GPU Memory Usage Before Runing Script ------------------------------ \n')
print("Total: %s,  Free: %s, Used: %s, Usage percentage: %7.4f\n" 
      %(GPU_info.total, GPU_info.total, GPU_info.used, (100*GPU_info.used/GPU_info.total)))

def print_gpu_info(handle, process_step):
    GPU_info = nvmlDeviceGetMemoryInfo(handle)
    print("----- GPU Memory Usage: %s -----" %process_step)
    print("Total: %s,  Free: %s, Used: %s, Usage percentage: %7.4f" 
          %(GPU_info.total, GPU_info.total, GPU_info.used, (100*GPU_info.used/GPU_info.total)))
    print('Reserved: %s GB' %(torch.cuda.memory_reserved() / 1024**3))
    print('Max reserved: %s GB' %(torch.cuda.max_memory_reserved() / 1024**3))
    print('Allocated: %s GB' %(torch.cuda.memory_allocated() / 1024**3))
    print('Max allocated: %s GB' %(torch.cuda.max_memory_allocated() / 1024**3))
    print('\n')
    torch.cuda.reset_peak_memory_stats()
"""



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
        self.add_module('linear1', torch.nn.Linear(data_dim, 200))
        self.add_module('drop1', torch.nn.BatchNorm1d(200))
        #self.add_module('relu1', torch.nn.ELU())
        self.add_module('relu1', torch.nn.LeakyReLU(negative_slope=0.01))
        self.add_module('linear2', torch.nn.Linear(200,100))
        self.add_module('drop2', torch.nn.BatchNorm1d(100))
        #self.add_module('relu2', torch.nn.ELU())
        self.add_module('relu2', torch.nn.LeakyReLU(negative_slope=0.01))
        self.add_module('linear3', torch.nn.Linear(100, 50))
        self.add_module('drop3', torch.nn.BatchNorm1d(50))
        #self.add_module('relu3', torch.nn.ELU())
        self.add_module('relu3', torch.nn.LeakyReLU(negative_slope=0.01))
        self.add_module('linear4', torch.nn.Linear(50, 20))
        self.add_module('drop4', torch.nn.BatchNorm1d(20))
        #self.add_module('relu4', torch.nn.ELU())
        self.add_module('relu4', torch.nn.LeakyReLU(negative_slope=0.01))
        self.add_module('linear5', torch.nn.Linear(20, out_dim))
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
        with torch.no_grad(): #Comment this line if training process is not stale
            self[0].weight = torch.nn.Parameter(torch.eye(data_dim)) #Comment this line if training process is not stale
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
def vdkl_trial(train_x, train_y, test_x, test_y, params, dn, f_ext, train_mean, output_dir):
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
    model_list = [DKLVAEModel(dim_d,out_dim,n_induc) for k in range(k_fold)]
    likelihood_list = [gpytorch.likelihoods.GaussianLikelihood() for k in range(k_fold)]
    mll_list = [gpytorch.mlls.PredictiveLogLikelihood(likelihood_list[k], model_list[k].gp_layer, num_data=n_train_sub) for k in range(k_fold)]
    #print(model_list[0].state_dict())
    #print(model_list[0].gp_layer.state_dict()['variational_strategy.inducing_points'])
    
    ##### Use the model parameters which has been trained #####
    if (use_previous_model_to_train=='yes'):
      if os.path.exists(model_input):
        for model in model_list:
          model.load_state_dict(torch.load(model_input))  
      else:
        raise ValueError('%s does not exit!' % model_input)
    
    ##### Initialize the hyperarameter parameters #####
    hypers = {
        'covar_module.outputscale': torch.tensor(20 * np.std(train_y.detach().cpu().numpy())),
    }
    for model in model_list:
        model.gp_layer.initialize(**hypers)
      
    hypers = {
        'noise': torch.tensor(5 * np.std(train_y.detach().cpu().numpy())),
    }
    for likelihood in likelihood_list:
        likelihood.initialize(**hypers)
    
    ##### Define subdataset & model algorithm for cross-validation #####
    ti = time.time()
    val_data = np.zeros(k_fold)
    likelihood_noise = np.zeros((n_steps,k_fold))
    #print(model_list[0].gp_layer.covar_module.outputscale.detach().cpu().numpy())
    DKL_loss = []
    for k in range(k_fold):
        if k_fold > 1:
            inds_k = np.roll(range(n_train),k*n_train//k_fold)[:n_train_sub]
            inds_val = np.roll(range(n_train),k*n_train//k_fold)[n_train_sub:]
            train_x_k = train_x[inds_k,:]
            train_y_k = train_y[inds_k]
            train_x_val = train_x[inds_val,:]
            train_y_val = train_y[inds_val]
        else:
            train_x_k = train_x
            train_y_k = train_y
            train_x_val = train_x
            train_y_val = train_y
        # print(type(n_batch))
        #print(train_x_k.device)
        if torch.cuda.is_available():
            model_list[k] = model_list[k].cuda()
            likelihood_list[k] = likelihood_list[k].cuda()
            mll_list[k] = mll_list[k].cuda()
            #train_x_k = train_x_k.cuda()
            #train_y_k = train_y_k.cuda()
            #train_x_val = train_x_val.cuda()
            #train_y_val = train_y_val.cuda()
            
        # Define optimizer
        optimizer = torch.optim.Adam([
            {'params': model_list[k].feature_extractor.parameters()},
            {'params': model_list[k].gp_layer.parameters()},
            {'params': likelihood_list[k].parameters()},
            {'params': model_list[k].vae_layer.parameters()},
        ], weight_decay=1e-3, lr=lr)
        
        # Specify learning rate scheduler
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=700, gamma=0.5)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        # for param in model_list[k].vae_layer.parameters():
        #     print(param)
        model_list[k].train()
        likelihood_list[k].train()
        mll_list[k].train()
        l_test = torch.nn.MSELoss()
        
        
        ############### Train DKL & GPR In Unison ###############
        ##### Initialization #####
        total_batch = train_x_k.shape[0]//n_batch
        running_loss_train = np.zeros(n_steps) 
        running_loss_val = np.zeros(n_steps)
        kernel_param = np.zeros((total_batch,n_steps,out_dim+1)) # [0:out_dim]: lengthscale [-1]:outputscale
        #print(total_batch)
        ##### Training #####
        train_dataset = TensorDataset(train_x_k, train_y_k)
        for i in range(n_steps):
            minibatch_iter = DataLoader(train_dataset, batch_size=n_batch, shuffle=True, generator=torch.Generator(device='cuda'), drop_last=True)
            #minibatch_iter = DataLoader(train_dataset, batch_size=n_batch)
            #minibatch_iter = DataLoader(train_dataset, batch_size=n_batch, shuffle=True, drop_last=True)

            for batch_idx, (batch_x, batch_y) in enumerate(minibatch_iter):
                #https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad(set_to_none=True)
                if i < 50:
                    with gpt_settings.cholesky_jitter(1e-1),gpt_settings.fast_computations(covar_root_decomposition=False):
                        output, mu, logvar, x_lat = model_list[k](batch_x)
                    #output, mu, logvar, x_lat = model_list[k](batch_x)
                else:
                    output, mu, logvar, x_lat = model_list[k](batch_x)
                    # print(model.vae_layer.logvar.bias)
                    # x_lat, _, _ = model.vae_layer(model.feature_extractor(batch_x))
                    # kl_div = kl_vae_qp(x_lat,mu,logvar)
                try:
                    mll_loss = -mll_list[k](output, batch_y)
                except ValueError:
                    for param in model_list[k].vae_layer.parameters():
                        print(param)
                    print()
                    
                loss = mll_loss # + lambda_kl*kl_div
                #print(k,i,batch_idx,loss.item())
                #print("Kernel lengthscale %s" %(model_list[k].gp_layer.covar_module.base_kernel.lengthscale.detach().cpu().numpy()))
                loss.backward(retain_graph=True)
                loss = loss.item()
                    # if ((i + 1) % 5 == 0):
                        # print(f"{dn} Iter {i + 1}/{n_steps}: {loss}")
            
                with torch.no_grad():
                  running_loss_train[i] = running_loss_train[i] + loss
                  output_val, mu_val, logvar_val, x_lat = model_list[k](train_x_val)
                  running_loss_val[i] = running_loss_val[i] + (-mll_list[k](output_val, train_y_val).item())
                  #print(batch_idx,i)
                  #print(model_list[k].gp_layer.covar_module.base_kernel.lengthscale.detach().cpu().numpy())
                  kernel_param[batch_idx,i,:out_dim] = model_list[k].gp_layer.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
                  kernel_param[batch_idx,i,-1] = model_list[k].gp_layer.covar_module.outputscale.item()
                
                optimizer.step()
                
                # Release GPU memory
                del loss
                gc.collect()
                torch.cuda.empty_cache()
                del mll_loss
                gc.collect()
                torch.cuda.empty_cache()
            
            # Update learning rate
            scheduler.step()
            
            with torch.no_grad():
              running_loss_train[i] = running_loss_train[i]/total_batch  
              running_loss_val[i] = running_loss_val[i]/total_batch
              #print(i,running_loss_train[i],running_loss_val[i]
              
            likelihood_noise[i,k] = likelihood_list[k].noise.item()  
            
        
        # Release GPU memory
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        """
        # Fix dkl projection, train just gpr
        # No need for regularization, if previously used
        # Training on full dataset, could be adjusted for very large datasets
        gpr_max_data = 25000 #number of smapling data from train_x_k/train_y_k to train GPR
        #gpr_max_data = 1000 #number of smapling data from train_x_k/train_y_k to train GPR
        indices = torch.randperm((train_y_k.size()[0]))[:gpr_max_data]
        train_x_gpr = train_x_k[indices,:]
        train_y_gpr = train_y_k[indices]
        
        optimizer = torch.optim.Adam([
            {'params': model_list[k].gp_layer.parameters()},
            {'params': likelihood_list[k].parameters()},
        ], lr=lr)
        
        
        #n_steps= 1000 #150
        for i in range(n_steps):
            #https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad(set_to_none=True)
            output, mu, logvar, x_lat = model_list[k](train_x_gpr)
            loss = -mll_list[k](output, train_y_gpr)
            # print(loss)
            loss.backward(retain_graph=True)
            loss=loss.item()
            # if ((i + 1) % 10 == 0):
            #     print(f"Iter {i + 1}/{n_steps}: {loss}")
            
            #with torch.no_grad():
            #  running_loss_train[n_steps+i] = running_loss_train[n_steps+i] + loss
            #  output_val, mu_val, logvar_val, x_lat = model_list[k](train_x_val)
            #  running_loss_val[n_steps+i] = running_loss_val[n_steps+i] + (-mll_list[k](output_val, train_y_val).item())
            
            optimizer.step()
            
            del loss
            gc.collect()
            torch.cuda.empty_cache()
        
        """
        DKL_loss.append(np.vstack((running_loss_train,running_loss_val)).T)
        filename = "Kernel_Params_fold-" + str(k) + ".npy"
        np.save((output_dir+filename), kernel_param)     
        
        
        model_list[k].eval()
        likelihood_list[k].eval()
        with torch.no_grad():
          output_test, mu_test, logvar_test, x_lattest = model_list[k](train_x_val)
          val_loss = np.sqrt(l_test(likelihood_list[k](output_test).mean.flatten(),train_y_val).item())
          val_data[k] = val_loss
    
    DKL_loss = np.array(DKL_loss)
    filename = "DKL_Loss.npy"
    np.save((output_dir+filename), DKL_loss)
    
    filename = "Likelihood_Noise.txt"
    np.savetxt((output_dir+filename), likelihood_noise)
    
    # Select best iteration of cross-validation
    tf = time.time()
    k_min = np.argmin(val_data)
    model = model_list[k_min]
    likelihood = likelihood_list[k_min]
    cwd = os.getcwd()
    print("\n ------------------------------ Cross-Validation Performance ------------------------------ \n")
    print('Total training time is %f min \n' %((tf-ti)/60.0))
    print("MSE:%s \n" %val_data)
    print("Best model index: %s\n" %k_min)
    print("Save model parameters in %s\n" %((cwd+output_dir+model_output)))
    print("Raw lengthe scale:")
    print(model.gp_layer.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0])
    print("Actual lengthe scale:")
    print(model.gp_layer.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0]) 
    print("\nRaw outputscale: %s" %(model.gp_layer.covar_module.raw_outputscale.item()))
    print("Actual outputscale: %s" %(model.gp_layer.covar_module.outputscale.item()))
    print("\nRaw likelihood noise: %s" %(likelihood.raw_noise.item()))
    print("Actual likelihood noise %s" %(likelihood.noise.item()))
    
    torch.save(model.state_dict(), (output_dir+model_output))
    #print(model.state_dict())
    
    
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
  
    # torch.save(model.feature_extractor.state_dict(),dn+'/CV/dict_opt_fe_'+f_ext+'.pth')
    # torch.save(model.vae_layer.state_dict(),dn+'/CV/dict_opt_var_'+f_ext+'.pth')
    
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
        data_out = np.zeros((len(test_y),3))
        data_out[:,0] = (test_y+train_mean).cpu()
        data_out[:,1] = (pos_train_mean+train_mean).cpu()
        data_out[:,2] = pos_pred_2.stddev.flatten().cpu()
        # return data_out
        # np.savetxt("data/gpy/ham_homo_out/approx_fit_"+str(ind_d-5)+".csv",data_out,delimiter=',')
        # np.savetxt(dn+"/CV_test/vdkl_fit_opt_"+f_ext+".csv",data_out,delimiter=',')
        # np.savetxt(dn+"/vdkl_fit_"+str(n_batch)+".csv",data_out,delimiter=',')
        filename = 'Predict_Mean_Testing.txt'
        np.savetxt((output_dir+filename), data_out)
        
        
        #################### Estimate Model Performance ####################
        #correlation_matrix = np.corrcoef(data_out[:,0], data_out[:,1])
        r_2 = np.corrcoef(data_out[:,0], data_out[:,1])[0, 1]**2
        MAE = np.mean(np.abs(data_out[:,0] - data_out[:,1]))
        RMSE = np.power(np.mean((data_out[:,0] - data_out[:,1])**2), 0.5)
        
        print('\n ------------------------------ Model Evaluation (Unit In meV & Log10 scale) ------------------------------ \n')
        print('Mean & standard deviation of testing dataset: %8.5f +- %8.5f\n' %(data_out[:,0].mean(),test_y.std()))
        print('Averge over GPR predicted mean of testing dataset: %8.5f\n' %(data_out[:,1].mean()))
        print('Averge over GPR predicted standard deviation of testing dataset: %8.5f\n' %(data_out[:,2].mean()))
        print('R square: %10.7f\n' %(r_2))
        print('MAE: %10.7f\n' %(MAE))
        print('RMSE: %10.7f\n' %(RMSE))
        
    return val_data, data_out



def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))


def vdkl_experiment(n_train, dn, fn_list, params, f_ext, output_dir):
    ############### Loading Training Dataset ###############
    # Loads the distance matrix data and normalizes the input vectors
    fname_x = dn+"/"+fn_list[0]
    fname_y = dn+"/"+fn_list[1]
    #data_x = torch.Tensor(np.loadtxt(fname_x,delimiter=','))
    data_x = torch.Tensor(np.load(fname_x))
    
    # standardize training data
    train_xmean = torch.mean(data_x,0)
    train_std = torch.std(data_x,0)
    data_x -= train_xmean
    data_x /= train_std
    #data_y = torch.Tensor(np.loadtxt(fname_y))
    no_data = data_x.size()[0]
    data_y = torch.Tensor(np.reshape(np.load(fname_y),(no_data,1))[:,0])
    data_y = data_y * 1000 # change unit from eV to meV
    data_y = torch.log10(torch.abs(data_y)) #change to log scale
    train_mean = torch.mean(data_y)
    data_y -= train_mean
    
    # train_perm is the random permutation to divide training and test data
    train_perm = np.random.permutation(len(data_y))
    train_x = data_x[train_perm[:n_train],:]
    train_y = data_y[train_perm[:n_train]]
    
    
    ############### Loading Testing Dataset ###############
    fname_xt = dn+"/"+fn_list[2]
    fname_yt = dn+"/"+fn_list[3]
    #test_x = torch.Tensor(np.loadtxt(fname_xt,delimiter=','))
    #test_y = torch.Tensor(np.loadtxt(fname_yt))
    test_x = torch.Tensor(np.load(fname_xt))
    no_data = test_x.size()[0]
    test_y = torch.Tensor(np.reshape(np.load(fname_yt),(no_data,1))[:,0])
    test_y = test_y * 1000 # change unit from eV to meV
    test_y = torch.log10(torch.abs(test_y)) #change to log scale
    
    # Apply same standardization to test set
    test_x -= train_xmean
    test_x /= train_std
    test_y -= train_mean
    
    # test_perm = np.random.permutation(len(data_y))
    # test_x = test_x[test_perm[:n_train],:]
    # test_y = test_y[test_perm[:n_train]]
    
    
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
    print('Mean & standard deviation of training dataset: %8.5f +- %8.5f [meV]\n' %(train_mean.cpu().item(), train_y.std().cpu().item())) 
    print('\n ------------------------------ Model Setup Parameter ------------------------------ \n')
    print('Batch size: %d\n' %(params[0])) 
    print('Latent feature dimension: %d\n' %(params[1])) 
    print('Number of cross-validation: %d\n' %(params[2]))
    print('Learning rate: %s\n' %(params[4]))
    print('Number of epoch: %s\n' %(int(params[5]))) 
    print('Number of GPR inducing points (or grid size): %s\n' %(int(params[6]))) 
    
    
    #################### Training Model ##################
    val_trial, data_trial = vdkl_trial(train_x, train_y, test_x, test_y, params, dn, f_ext, train_mean, output_dir)
    

if __name__ == '__main__':
    main()


