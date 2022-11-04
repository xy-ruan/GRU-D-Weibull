import torch
import math
import numpy as np

class GRUD_cell(torch.nn.Module):
	def __init__(self,
			input_size,	#take an integer, the number of input features	
			hidden_size,	#take an integer, the number of neurons in hidden layer
			output_size,	#take an integer, the number of output features
			x_mean=0,	#x_mean should be a vector containing the empirical mean value for each feature
			#bias=True,	#always include bias vector for z,r,h
			#bidirectional=False,	#not applicable in this study since in practice we won't know future data points
			#dropout_type='mloss',
			dropoutratio=0,	#the dropout ratio for between timesteps of hidden layers.
			#return_hidden=False,	#always return hidden tensor
			sigmoidscaler_1=1,
			sigmoidscaler_2=1,
			dtypearg=torch.float64,
			usedevice='cuda:0'
			):
		torch.set_default_dtype(dtypearg)
		cuda_available = torch.cuda.is_available()	#return True if NVIDIA available
		device = torch.device(usedevice if cuda_available else 'cpu')
		self.device=device
		super(GRUD_cell,self).__init__()
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.output_size=output_size
		#self.return_hidden=return_hidden
		self.sigmoidscaler_1=sigmoidscaler_1
		self.sigmoidscaler_2=sigmoidscaler_2
		#x_mean=torch.tensor(x_mean,dtype=torch.float32,requires_grad=True)	#mean value will also be updated in each cycle? after test requires_grad=True is useless is register_buffer is used
		#as noted by the author, register_buffer is used because it automatically load data to GPU,if any. However, this also prohibit x_mean from updating
		#self.register_buffer('x_mean',x_mean)
		#alternatively define this way so x_mean get updated on each batch
		self.x_mean=torch.nn.Parameter(torch.tensor(x_mean,dtype=dtypearg,requires_grad=True,device=device))
		#self.bias=bias
		self.dropoutratio=dropoutratio
		self.dropoutlayer=torch.nn.Dropout(p=dropoutratio)	#dropout must be defined within __init__ so switch between eval() and train() model effectively turn off/on dropout			
		#self.dropout_type=dropout_type
		#self.bidirectional=bidirectional
		#num_directions=2 if bidirectional else 1
		#weight matrix for gamma
		self.w_dg_x=torch.nn.Linear(input_size,input_size,bias=True,device=device)
		self.w_dg_h=torch.nn.Linear(input_size,hidden_size,bias=True,device=device)
		#weight matrix for z, update gate
		self.w_xz=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.w_hz=torch.nn.Linear(hidden_size,hidden_size,bias=False,device=device)
		self.w_mz=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.b_z=torch.nn.Parameter(torch.tensor(np.ndarray(hidden_size),dtype=dtypearg,requires_grad=True,device=device))
		#weight matrix for r, reset gate
		self.w_xr=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.w_hr=torch.nn.Linear(hidden_size,hidden_size,bias=False,device=device)
		self.w_mr=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.b_r=torch.nn.Parameter(torch.tensor(np.ndarray(hidden_size),dtype=dtypearg,requires_grad=True,device=device))
		#weight matrix for hidden units
		self.w_xh=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.w_hh=torch.nn.Linear(hidden_size,hidden_size,bias=False,device=device)
		self.w_mh=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.b_h=torch.nn.Parameter(torch.tensor(np.ndarray(hidden_size),dtype=dtypearg,requires_grad=True,device=device))
		#weight matrix for linking hidden layer to final output
		self.w_hy=torch.nn.Linear(hidden_size,output_size,bias=True,device=device)
		self.batchnorm=torch.nn.BatchNorm1d(hidden_size)
		#the hidden state vector, it is the conveying belt that transfer information to the next timestep
		#it will be scaled to batch_size x hidden_size in following script
		Hidden_State=torch.zeros(hidden_size,dtype=dtypearg,device=device)
		#it is registered in buffer, which prevents it from change. so this is a "stateless" model, which assumes no relatinoship between each training sample
		self.register_buffer('Hidden_State',Hidden_State)
		#a vector containing the last observation
		#it will be scaled to [batch_size, input_size] in the following script
		X_last_obs=torch.zeros(input_size,dtype=dtypearg,device=device)
		self.register_buffer('X_last_obs',X_last_obs)
		self.reset_parameters()
	def reset_parameters(self):	#this reset all weight parameters	
		stdv=1.0/math.sqrt(self.hidden_size)
		#for weight in self.parameters():	#note this reset all weights matrix except those registered in buffer
			#torch.nn.init.uniform_(weight, -1 * stdv, stdv)
		for name,weight in self.named_parameters():
			if(name != 'x_mean'):	#avoid reset certain weights at the startup
				torch.nn.init.uniform_(weight, -1 * stdv, stdv)
	@property
	def _flat_weights(self):	#no idea what this is doing
		return list(self._parameters.values())
	def forward(self, input):
		#determine the device where 
		device=self.device	#<class 'torch.device'>
		#input has these dimensions [batch_size,"X Mask Delta",feature_size, timestep_size]
		#move input to corresponding device
		if(input.device != device):
			input=input.to(device)
		X=input[:,0,:,:]	#X has dimension [batch_size, feature_size, timestep_size]
		Mask=input[:,1,:,:]
		Delta=input[:,2,:,:]
		#
		step_size=X.size(2)	#the size of timesteps
		output=None
		h=getattr(self,'Hidden_State')	#h is a vector
		x_mean=getattr(self,'x_mean')
		x_last_obsv=getattr(self,'X_last_obs')
		#
		#an empty tensor for holding the output of each timestep
		#the dimensions are [batch_size,timestep_size,output_feature_size]
		output_tensor=torch.empty([X.size(0),X.size(2),self.output_size], 
				dtype=X.dtype, device=device)
		#an empty tensor for holding the hidden state of each timestep
		#the dimensions are [batch_size,timestep_size,hidden_size]
		hidden_tensor=torch.empty([X.size(0),X.size(2),self.hidden_size],
				dtype=X.dtype,device = device)
		#print(self.w_hh.weight)
		#iterate over timesteps
		for timestep in range(X.size(2)):
			#squeeze drop the timestep dimension and ends up with [batch_size,feature_size]
			x=X[:,:,timestep]
			m=Mask[:,:,timestep]
			d=Delta[:,:,timestep]
			#the gamma vector for filtering x. the dimension is [batch_size,feature_size]
			#each element in gamma_x is a value within (0,1]
			gamma_x=torch.exp(-1*torch.nn.functional.relu(self.w_dg_x(d)))
			#the gamma vector for filtering h. the dimension is [batch_size,hidden_size]
			gamma_h=torch.exp(-1*torch.nn.functional.relu(self.w_dg_h(d)))
			#update x_last_obsv. x_last_obsv is a vector of input_size
			#it is changed to a [batch_size,feature_size] tensor on first run
			#x_last_obsv are all 0 at the beginning, and update to the latest available data 
			x_last_obsv=torch.where(m>0,x,x_last_obsv)
			#set nan in x to 0 -- necessary to do this on the training data before input to cell
			#x[torch.isnan(x)]=0	#this kind of operation is strictly not allowed becuase it raises the RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
			x=m*x + (1-m)*(gamma_x*x_last_obsv + (1-gamma_x)*x_mean)
			#
			h=gamma_h*h	#h is initialized as a vector, and reshaped to [batch_size,hidden_size] on first run
			#z is [batch_size, hidden_size]
			z=torch.sigmoid(self.w_xz(x) + self.w_hz(h) + self.w_mz(m) + self.b_z)
			#r is [batch_size, hidden_size]
			r=torch.sigmoid(self.w_xr(x) + self.w_hr(h) + self.w_mr(m) + self.b_r)
			#h_tilde is [batch_size, hidden_size]
			h_tilde=torch.tanh(self.w_xh(x) + self.w_hh(r*h) + self.w_mh(m) + self.b_h)
			#h is [batch_size, hidden_size]
			h=(1-z)*h + z*h_tilde
			#if using batch normalization, should be added here
			h=self.batchnorm(h)
			if(self.dropoutratio > 0):	#remember to use eval() to turn off dropout during evaluation
				h=self.dropoutlayer(h)
			step_output=self.w_hy(h)	#[batch_size,output_size]
			#scaling output strategy
			#strategy 1, scale after sigmoid
			#output_tensor[:,timestep,0]=torch.sigmoid(step_output[:,0]) * self.sigmoidscaler_1
			#output_tensor[:,timestep,1]=torch.sigmoid(step_output[:,1]) * self.sigmoidscaler_2
			#strategy 2, softplus
			softplus=torch.nn.Softplus()
			output_tensor[:,timestep,0]=softplus(step_output[:,0])	#kappa
			#output_tensor[:,timestep,0]=torch.tanh(step_output[:,0])*self.sigmoidscaler_1 + 1
			output_tensor[:,timestep,1]=softplus(step_output[:,1])	#lambda
			if(self.output_size == 3):
				output_tensor[:,timestep,2]=softplus(step_output[:,2])	#lagrange point
			#strategy 3, fix k to around 3.25
			#softplus=torch.nn.Softplus()
			#output_tensor[:,timestep,0]=torch.tanh(step_output[:,0]) * 1 + 3.25
			#output_tensor[:,timestep,1]=softplus(step_output[:,1])
			#this line is always required
			hidden_tensor[:,timestep,:]=h
		#end of timestep loop
		output=(output_tensor,hidden_tensor)
		return output
	#end of forward



from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#this class takes care of creating batch input from input tensor
#it is specifically designed for CKD (or like) project which has the following input tensor
#nparray (sample,3,feature,timestep)
#targetarray (timestep, sample) i.e. the training target changes over time
#				since the time to end point decrease as time progress
#valid_timesteps_array, a 1 dimensional array containing the max allowed timesteps
#event_array, a 1 dimensional array containing the censoring status
class CKDdataset(Dataset):
	def __init__(self,nparray,targetarray,valid_timesteps_array=None,weights=None,eventarr=None):
		assert(nparray.shape[0] == targetarray.shape[1]), "nparray length and target array length not match"
		self.x=nparray
		self.y=targetarray
		self.z=None
		if(not valid_timesteps_array is None):
			assert(nparray.shape[0] == len(valid_timesteps_array)), "nparray length and valid timestep array length not match"
			self.z=valid_timesteps_array
		#end of if
		self.w=None
		if(not weights is None):
			assert(nparray.shape[0] == len(weights))
			self.w=weights
		#end of if
		self.e=None
		if(not eventarr is None):
			assert(nparray.shape[0] == len(eventarr))
			self.e=eventarr
		#end of if
	def __len__(self):
		return(self.x.shape[0])
	def __getitem__(self,idx):
		if(self.z is None):
			return(self.x[idx,:,:,:],self.y[:,idx])
		elif(not self.z is None and self.w is None):
			return(self.x[idx,:,:,:],self.y[:,idx],self.z[idx])
		elif(not self.z is None and not self.w is None and self.z is None):
			return(self.x[idx,:,:,:],self.y[:,idx],self.z[idx],self.w[idx])
		elif(not self.z is None and not self.w is None and not self.z is None):
			return(self.x[idx,:,:,:],self.y[:,idx],self.z[idx],self.w[idx],self.e[idx])
		#end of if
	#end of def
#end of class

class GRUD_model(torch.nn.Module):
	def __init__(self,
			input_size,
			hidden_size,	#this determines hidden_size of both the first and all stacked layers
			output_size,	#the output size of every layer
			num_layers=1,
			x_mean=0,	#empirical mean of each input feature
			#bias=True,
			#batch_first=False,
			#bidirectional=False,
			#dropout_type='mloss',
			dropoutratio=0,
			sigmoidscaler_1=1,
			sigmoidscaler_2=1,
			dtypearg=torch.float64,
			usedevice='cuda:0'
			):
		torch.set_default_dtype(dtypearg)
		cuda_available = torch.cuda.is_available()
		device = torch.device(usedevice if cuda_available else 'cpu')
		self.device=device
		super(GRUD_model,self).__init__()
		#first layer is the GRU-D
		self.gru_d=GRUD_cell(input_size=input_size,
					hidden_size=hidden_size,
					output_size=output_size,
					x_mean=x_mean,
					dropoutratio=dropoutratio,
					sigmoidscaler_1=sigmoidscaler_1,
					sigmoidscaler_2=sigmoidscaler_2,
					dtypearg=dtypearg,
					usedevice=usedevice
					)
		self.num_layers=num_layers
		self.hidden_size=hidden_size
		self.output_size=output_size
		self.sigmoidscaler_1=sigmoidscaler_1
		self.sigmoidscaler_2=sigmoidscaler_2
		#stack other layers as regular GRU layer
		if(self.num_layers > 1):
			self.gru_layers=torch.nn.GRU(input_size=hidden_size,
				hidden_size=hidden_size,
				batch_first=True,	#necessary because the output from gru_d is [batch_size,timestep,feature], whereas torch.nn.GRU takes [timestep,batch,feature] by default
				num_layers=self.num_layers-1,
				dropout=dropoutratio,
				device=device
				)	#I manually confirmed torch.nn.GRU dropout is sensitive to switch between eval() and training()
			#this is for converting the last layer hidden output to actual output
			self.hidden_to_output=torch.nn.Linear(hidden_size, output_size, bias=True, device=device)	
	#end of __init__ 
	def _flat_weights(self):
		return list(self._parameters.values())
	def forward(self,input):
		#pass through the first gru_d layer
		#note real_output is [batch_size,timestep,output_feature]
		#hidden is [batch_size,timestep,hidden_size]
		#determine the device where 
		if(hasattr(self,'device')):
			device=self.device	#<class 'torch.device'>
			#move input to corresponding device
			if(input.device != device):
				input=input.to(device)
		(real_output,hidden)=self.gru_d(input)
		if(self.num_layers > 1):
			#pass through the rest of regular gru layers
			#output contains the final layer output for all samples and all timesteps
			#output has shape [batch_size,timestep,hidden_size]
			#hidden has shape [num_layers, batch_size, hidden_size]
			#in this example output[:,timestep-1,:] == hidden, should be all true
			(output,hidden)=self.gru_layers(hidden)
			#covert output of last layer hidden output [batch_size, timestep, hidden_size] to real output [batch_size, timestep, output_feature]
			inter_output=self.hidden_to_output(output)
			#strategy 1
			#s=real_output.shape
			#output_tensor=torch.empty([s[0],s[1],s[2]],device=device)
			#output_tensor[:,:,0]=torch.sigmoid(inter_output[:,:,0]) * self.sigmoidscaler_1
			#output_tensor[:,:,1]=torch.sigmoid(inter_output[:,:,1]) * self.sigmoidscaler_2
			#strategy 2
			s=real_output.shape
			output_tensor=torch.empty([s[0],s[1],s[2]],device=device)
			softplus=torch.nn.Softplus()
			output_tensor[:,:,0]=softplus(inter_output[:,:,0])
			output_tensor[:,:,1]=softplus(inter_output[:,:,1])
			if(s[2] == 3):
				output_tensor[:,:,2]=softplus(inter_output[:,:,2])
			#strategy 3
			#output_tensor[:,:,0]=torch.tanh(inter_output[:,:,0])*2 + 3.25
			#output_tensor[:,:,1]=softplus(inter_output[:,:,1])
			real_output=output_tensor
			hidden=output
		#end of if
		return(real_output,hidden)	#make sure the output shape is same as gru_d
	#end of forward
#end of grud_model

class Multimodal_GRUD(torch.nn.Module):
	def __init__(self,
			num_modal,
			input_size_tuple,
			hidden_size,
			output_size,
			x_mean_tuple=None,
			num_layers=1,
			dropoutratio=0,
			sigmoidscaler_grud=1,
			sigmoidscaler_gru=1,
			dtypearg=torch.float64,
			usedevice='cuda:0'
			):
		torch.set_default_dtype(dtypearg)
		super(Multimodal_GRUD,self).__init__()
		assert(len(input_size_tuple) == num_modal), "expect num_modal to match length of input_size_tuple"
		cuda_available = torch.cuda.is_available()
		device=torch.device(usedevice if cuda_available else 'cpu')
		self.device=device
		self.num_modal=num_modal
		self.hidden_size=hidden_size
		self.grud_model_list=[]
		self.output_size=output_size
		self.input_size_tuple=input_size_tuple
		self.w_multihidden_hidden=torch.nn.Linear(hidden_size*num_modal,hidden_size,bias=True,dtype=dtypearg,device=device)
		self.w_hidden_y=torch.nn.Linear(hidden_size,output_size,bias=True,dtype=dtypearg,device=device)
		#this is a special list that ensures parameters in list of models are recognizable by optimizer 
		self.paramlist=torch.nn.ParameterList()
		for i in range(num_modal):
			#by default use 0 for all input 
			x_mean=np.zeros(input_size_tuple[i])
			if(not x_mean_tuple is None):
				x_mean=x_mean_tuple[i]
				assert(len(x_mean)==input_size_tuple[i]),"Error: length of x_mea and input_size not match"
			#end of if
			self.grud_model_list.append(GRUD_model(input_size=input_size_tuple[i],
					hidden_size=hidden_size,
					output_size=output_size,
					num_layers=num_layers,
					x_mean=x_mean,
					dropoutratio=dropoutratio,
					sigmoidscaler_grud=sigmoidscaler_grud,
					sigmoidscaler_gru=sigmoidscaler_gru,
					dtypearg=dtypearg,
					usedevice=usedevice
				))
			#this is very important!
			self.paramlist+=list(self.grud_model_list[i].parameters())
		#end of for
	#end of def
	def _flat_weights(self):
		return list(self._parameters.values())
	def forward(self,input_tuple):
		#input_tuple is a tuple with each element [batch_size,3,feature_size,timestep_size]
		assert(len(input_tuple)==self.num_modal),"Error: input_tuple length not match self.num_modal:[%s]" % self.num_modal
		#batch_size and timestep_size much be equal for all input elements
		for i in range(len(input_tuple)-1):
			assert(input_tuple[i].shape[0] == input_tuple[i+1].shape[0] and input_tuple[i].shape[3] == input_tuple[i].shape[3]), "input_tuple batch size or timestep not match for %s th element" % i
		#end of for
		batchsize=input_tuple[0].shape[0]
		#print("check 1")
		#an empty list for holding the hidden output
		hidden_output_list=[]
		for i in range(self.num_modal):
			#hidden has shape [batch_size, timestep, hidden_size]
			#real_output is the real output from each individual model
			#	it has shape [batch_size,timestep, output_size]
			(real_output,hidden)=self.grud_model_list[i](input_tuple[i])	
			#print(hidden.shape)
			hidden_output_list.append(hidden)
		#end of for
		#print("check 2")
		#stack hidden output from each model together, result is [batch_size, timestep, stacked_output_size]
		hidden_output_cat=torch.cat(hidden_output_list,dim=2)
		#print(hidden_output_cat.shape)
		#convert stacked hidden output to final output
		output_layer1=self.w_multihidden_hidden(hidden_output_cat)
		output_layer2=torch.sigmoid(self.w_hidden_y(output_layer1))
		return(output_layer2,hidden_output_cat)
	#end of def
#end of class


