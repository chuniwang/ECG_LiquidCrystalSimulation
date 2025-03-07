import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sys,time,os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


"""
############### Define General System Variables/Parameters ###############
##### Comuting device/environment #####
GPU_index = 3
#random_seed = 1
#torch.manual_seed(random_seed)    # reproducible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Computing device: %s" %device)
if torch.cuda.is_available():
  torch.cuda.set_device(GPU_index)
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
  torch.set_default_tensor_type(torch.FloatTensor)
"""

device = 'cpu'
torch.set_default_tensor_type(torch.FloatTensor)

###### Suffle samples ######
def Suffle_Sample_XY(X,Y,batch_size):
    torch_dataset = Data.TensorDataset(X, Y)
    loader = Data.DataLoader(torch_dataset, batch_size, shuffle=True)
    
    for step, (X_batch, y_batch) in enumerate(loader):
        X = X_batch
        Y = y_batch
        return X,Y

############ defining the network ############
class Net(nn.Module):
  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,200)
    self.fc2 = nn.Linear(200,100)
    self.fc3 = nn.Linear(100,64)
    self.fc4 = nn.Linear(64,32)
    self.fc5 = nn.Linear(32,1)
    self.batchnorm_0 = torch.nn.BatchNorm1d(input_shape)
    self.batchnorm_1 = torch.nn.BatchNorm1d(200)
    self.batchnorm_2 = torch.nn.BatchNorm1d(100)
    self.batchnorm_3 = torch.nn.BatchNorm1d(64)
  def forward(self,x):
    x = torch.relu(self.fc1(self.batchnorm_0(x)))
    x = torch.relu(self.fc2(self.batchnorm_1(x)))
    x = torch.relu(self.fc3(self.batchnorm_2(x)))
    x = torch.relu(self.fc4(self.batchnorm_3(x)))
    x = torch.sigmoid(self.fc5(x))
    return x


def main():    
    ############ Set parameters ###########
    total_sample = 128000
    training_size = 125000
    validation_size = 1000
    testing_size = 0 #512
    batch_size = 500 #2000
    device = "CPU" #"CUDA"
    #scan_learning_rate = np.array([ 0.05, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0005])
    #scan_learning_rate = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
    scan_learning_rate = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
    #scan_learning_rate = np.array([0.01, 0.05])
    total_epoch = 500
    random_seed = 1
    best_model_val_checkpoint = 0.7 #val_accuracy
    torch.manual_seed(random_seed)    # reproducible

    Feature_type = "CG_"
    Model_type = 'H4_BS500_LR'
    
    #### Training dataset #####
    Xfilename = '/home/ciwang/Electronic_Coupling/ML_Dataset/700K/Feature_Coupling_CG_Improve_25_Inter_Training_COM.npy'
    Yfilename = '/home/ciwang/Electronic_Coupling/ML_Dataset/700K/Label_Coupling_Sample_128000.npy'

    ##### Testing dataset #####
    Xfilename_test = '/home/ciwang/Electronic_Coupling/ML_Dataset/700K/Feature_Coupling_CG_Improve_25_Inter_Testing_COM.npy'
    Yfilename_test = '/home/ciwang/Electronic_Coupling/ML_Dataset/700K/Label_Coupling_1_10240.npy'



    ############ Loading data ########### 
    print('Path of feature file:', Xfilename)
    ######### Training dataset #########
    X = np.load(Xfilename)
    no_data = X.shape[0]
    ##### Feature standardization #####
    train_xmean = np.mean(X,0)
    train_xstd = np.std(X,0)
    X -= train_xmean
    X /= train_xstd
    #X = np.reshape(X.flatten(),(no_data,1,16,16))
    X = Variable(torch.from_numpy(np.array(X)))
    ##### Generate binary classification label ##### 
    Y_value = np.load(Yfilename).flatten()
    Y = np.zeros((no_data), dtype=int)
    Y = np.where((Y_value>0.0),Y+1,Y) 
    Y = Variable(torch.from_numpy(np.array(Y)))
    #####  Shuffle dataset #####
    X, Y = Suffle_Sample_XY(X, Y, total_sample) 
    print("\nWhole training dataset:\nshape of x: {}\nshape of y: {}".format(X.shape,Y.shape))
    print("Number of data for training: {} ".format(training_size))
    print('Number of posive/negative sign in training dataset: %s/%s\n' %(Y.sum().item(),(training_size-Y.sum()).item()))

    ######### Testing dataset #########
    X_test = np.load(Xfilename_test)
    no_data_test = X_test.shape[0]
    ##### Feature standardization #####
    X_test -= train_xmean
    X_test /= train_xstd
    #X_test = np.reshape(X_test.flatten(),(no_data_test,1,16,16))
    X_test = torch.Tensor(X_test)
    ##### Generate binary classification label #####
    Y_test_value = np.load(Yfilename_test).flatten()
    Y_test = np.zeros((no_data_test), dtype=int)
    Y_test = np.where((Y_test_value>0.0),Y_test+1,Y_test) #Define sign class: positve=1, negative=0
    print("Testing dataset:\nshape of x: {}\nshape of y: {}".format(X_test.shape,Y_test.shape))
    print('Number of posive/negative sign in validation dataset: %s/%s\n' %(Y_test.sum().item(),(Y_test.shape[0]-Y_test.sum()).item()))
    
    ############ Assign dataset ############
    D_in = X.shape[1]  #D_in is the dimenssion of feature
    nv = training_size + validation_size
    nt = nv + testing_size
    X_train, Y_train = X[:training_size, :].float(), Y[:training_size].float()
    #X_val, Y_val = X[training_size:nv, :].float(), Y[training_size:nv].float()    
    X_val, Y_val = X_test, Y_test
    #X_test, Y_test = X[nv:nt, :].float(), Y[nv:nt].float()
    torch_dataset = Data.TensorDataset(X_train, Y_train)

    
    ############ Setup model parameters ############
    for lr_i, learning_rate in enumerate(scan_learning_rate):
      ######### Initiate model #########
      model = Net(D_in)
      if device == 'cuda':
        model.cuda()
      ######### Define optimizer #########
      #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      print('Learning rate: %s' %learning_rate)
      ######### Define loss function #########
      #loss_fn = nn.BCEWithLogitsLoss()
      loss_fn = nn.BCELoss()
      ######### Assign batch dataset #########
      trainloader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
      print('Batch size:', batch_size)
      print("Number of batch: %s" %(len(trainloader)))
      
      
      ###### Set Output File Name #####
      LR_type = learning_rate*10000
      LR_type.tolist()
      LR_type = "%05d" %LR_type
      output_filename = Feature_type + Model_type + LR_type +'.log'
      output_filename_model = Feature_type + Model_type + LR_type +'.pth'
      output_filename_best_model = Feature_type + Model_type + LR_type +'_Best_Model.pth'
      output_filename_val_pred_Y_best = Feature_type + Model_type + LR_type +'_val_pred_Y_best.txt'
      output_filename_val_pred_Y = Feature_type + Model_type + LR_type +'_val_pred_Y_final.txt'
      
      
      ############ Training ML model ############
      print('\nTraining......')
      ti = time.time()
      model.train()
      losses = []
      val_accur = []
      best_model_val_acc = best_model_val_checkpoint
      for i in range(total_epoch):
        for j,(x_train,y_train) in enumerate(trainloader):
          ######### Forward pass: Compute predicted probability by passing x to the model #########
          output = model(x_train)
 
          #########  Calculate loss of batch dataset #########
          loss = loss_fn(output,y_train.reshape(-1,1))
 
          ######### Evaluate val_accuracy of validation/testing data set #########
          val_predicted = model(X_val)
          val_class = val_predicted.reshape(-1).detach().numpy().round()
          val_acc = (val_class == Y_val).mean()
          
          ######### Save the Bset Model #########
          if (val_acc) > best_model_val_acc:     
            torch.save(model.state_dict(), output_filename_best_model)
            best_model_val_acc = val_acc
            best_model_epoch = i+1
            np.savetxt(output_filename_val_pred_Y_best, np.stack((Y_val,val_class,val_predicted.reshape(-1).detach().numpy()), axis=-1))
            print('Save the best model at the %d epoch! Accuracy: %.3f' %((i+1), val_acc))
          
          
          ######### Reset the parameter gradients #########
          optimizer.zero_grad()
          
          ######### Backpropagation & Upgrade model parameters #########
          loss.backward()
          optimizer.step()
          
        if i%1 == 0:
           losses.append(loss)
           val_accur.append(val_acc)
           print("epoch {}\ttrain_loss : {:.5f}\t val_accuracy : {:.3f}".format((i+1),loss,val_acc))

      tf = time.time()
      print('Total training time is %f min' %((tf-ti)/60.0))
      ######### Output Information: Parameters #########
      Output_file = open(output_filename, 'w')
      Output_file.write(' ------------------------------ File Information ------------------------------ \n')
      Output_file.write('Path of feature file: %s\n' %(Xfilename))
      Output_file.write('Path of label file: %s\n' %(Yfilename))
      Output_file.write('File of ML model: %s\n' %(output_filename_model))
      Output_file.write('Device type: %s\n' %(device))
      Output_file.write('Total training time: %s [min]\n' %((tf-ti)/60.0))
      if ('random_seed' in locals().keys()): Output_file.write('Random seed: %s\n' %(random_seed))
      Output_file.write('\n ------------------------------ Dataset Information ------------------------------ \n')
      Output_file.write('Total number of sample : %s\n' %(total_sample))
      Output_file.write('Feature type: %s\n' %(Feature_type))
      Output_file.write('Feature dimension: %s\n' %(D_in))
      Output_file.write('Size of trainining dataset: %s\n' %(X_train.shape[0]))
      Output_file.write('Size of validation dataset: %s\n' %(X_val.shape[0]))
      Output_file.write('Number of posive/negative sign in training dataset: %s/%s\n' %(Y.sum().item(),(training_size-Y.sum()).item()))
      Output_file.write('Number of posive/negative sign in validation dataset: %s/%s\n' %(Y_test.sum().item(),(Y_test.shape[0]-Y_test.sum()).item()))
      
      if best_model_val_acc > best_model_val_checkpoint: 
        Output_file.write('Best model accuracy of the validation dataset : %s\n' %(best_model_val_acc))  
        Output_file.write('Best model was obtained at epoch : %s\n' %(best_model_epoch))
        print('Best model accuracy of the validation dataset : %s' %(best_model_val_acc))
      else:
        Output_file.write('There is no model with accuracy of the validation dataset higher than %s .\n' %(best_model_val_checkpoint))
        print('There is no model with accuracy of the validation dataset higher than %s .' %(best_model_val_checkpoint))
        
      Output_file.write('\n ------------------------------ Model Parameter ------------------------------ \n')
      Output_file.write('Batch size: %s\n' %(batch_size)) 
      Output_file.write('Number of batch: %s\n' %(len(trainloader))) 
      Output_file.write('Number of epoch: %s\n' %(total_epoch)) 
      Output_file.write('Initial learning rate: %s\n' %(learning_rate))  
      for idx, m in enumerate(model.named_modules()):
          if (idx==0): Output_file.write('\n %s --> %s \n' %(idx, m))
      Output_file.write('\n ------------------------------ Optimizer  ------------------------------ \n')
      Output_file.write('%s\n' %(optimizer))
      Output_file.write('%s\n' %(optimizer.state_dict()['param_groups']))
      Output_file.write('\n ------------------------------ Scheduler  ------------------------------ \n')
      #Output_file.write('Type: %s\n' %(scheduler))
      #Output_file.write('Parameters: %s\n' %(scheduler.state_dict()))
      Output_file.write('\n ------------------------------ Loss function ------------------------------ \n')
      Output_file.write('%s\n' %(loss_fn))
      Output_file.write('\n ------------------------------ Architecture of neural network  ------------------------------ \n')
    
      
      ######### Save model #########
      model.eval()
      torch.save(model.state_dict(), output_filename_model)
      np.savetxt(output_filename_val_pred_Y, np.stack((Y_val,val_class,val_predicted.reshape(-1).detach().numpy()), axis=-1))
      print('\n ----------------------------------------------------------- \n')



if __name__ == '__main__':
    main()
