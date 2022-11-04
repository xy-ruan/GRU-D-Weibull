import numpy as np
import torch
import os
import sys
import sklearn
from sklearn import metrics
from sklearn import cluster
import random
from lifelines import KaplanMeierFitter

#expect output with shape [batch_size, output_feature]
#where output_feature should have 2 or 3 elements
#the first one corresponds to k
#the second one corresponds to lambda
#the third one, if any, is lagrange value
#about weights, either None or a tensor with length same as target
def weibull_LL(output,target,events,weights=None,compositeloss=False,tremain=None,epoch=None,epoch_mix_at=None,ts=None,calibrationhorizon=None,calibrationchunk=None,device=None):
	assert(output.shape[1] >= 2), "output must have >=2 features for each row"
	assert(output.shape[0] == target.shape[0]), "output and target must have the same record number"
	assert(output.shape[0] == len(events)), "output and events must have same length"
	if(not epoch_mix_at is None):
		assert(not epoch is None), "must provide epoch when epoch_mix_at is not none"
	if(not weights is None):
		assert(output.shape[0] == len(weights)), "output and weights, if any, must have same length"
	if(not calibrationhorizon is None):
		assert(not calibrationchunk is None), "calibrationchunk must be provided if calibrationhorizon is not none"
	#theoretically, all output should be > 0  for sigmoid softplus
	#however, for unknown reason, some small value may be truncated to 0
	output[output == 0]=1e-5
	assert(torch.sum(target<=0)==0), "target must be all > 0"
	k=output[:,0]
	lam=output[:,1]
	lagrange=1
	if(output.shape[1] == 3):
		lagrange=output[:,2]	#lagrange test, otherwise set to 1
	parta=partb=partc=partd=parte=None
	parta=torch.log(k/lam)	#logged for easy manipulation
	partb=(k-1)*torch.log(target/lam)
	partc= -1 * (target/lam)**(k)
	if(tremain is None):	#if use infinity target for censored 
		partx=partc
	else:			#otherwise use time remaining for censored
		partx=torch.log(torch.exp(-1 * (target/lam)**(k)) - torch.exp(-1 * (tremain/lam)**(k)))
	#end of if
	partx=torch.nan_to_num(partx,posinf=10,neginf=-10)	#replace potential nan with valid number
	calibrationloss=0
	if(not calibrationhorizon is None):  #timestep 99 is year 4 
		calibrationlosslist=[]
		kmf=KaplanMeierFitter()	#create KM object for calculating observed prob of survival
		for horizon in calibrationhorizon:	#iterate through e.g. 1,2,3,4,5 year horizon
			#calculate predicted prob of survival at horizon
			if((horizon == 1 and ts <= 99) or (horizon == 2 and ts <= 87) or 
				(horizon == 3 and ts <= 75) or (horizon == 4 and ts <= 63) or (horizon==5 and ts <= 44)):
				predsurv_h=torch.exp( -1 * (horizon/lam)**(k))
				orderidx=np.argsort(predsurv_h.cpu().detach())	#generate an array containing the order of predsurv_h (low to high)
				orderchunk=np.array_split(orderidx,calibrationchunk)
				horizonlist=[]
				for chunkidx in orderchunk:
					eventscount=torch.sum(events[chunkidx].cpu())
					chunkcount=len(chunkidx)
					if(chunkcount > 20 and eventscount/chunkcount > 0.1):	#require at least 20 samples in each chunk to calculate average prob of survival
						kmf.fit(durations=target[chunkidx].cpu(),event_observed=events[chunkidx].cpu())
						obssurv_h=kmf.predict(horizon)	#calculate the observed prob of survival on given horizon
						horizonlist.append(torch.abs(torch.mean(predsurv_h[chunkidx]) - obssurv_h))
					else:
						horizonlist.append(torch.tensor(0.0,device=device))
					#end of if
				#end of for
				horizonloss=torch.mean(torch.stack(horizonlist))	#should use mean rather than sum to make loss invariant to chunk number
				calibrationlosslist.append(horizonloss)
			#end of if
		#end of for
		if(len(calibrationlosslist) >= 1):	#only calculate calibration loss when there is >= 1 horizon loss
			calibrationloss=torch.mean(torch.stack(calibrationlosslist))	#should use mean rather than sum to make loss invariant to horizon number
		#end of if
	#end of if
	w=1
	if(not weights is None):
		w=weights
	if(compositeloss==True):	#also consider the difference bewteen predicted median survival time and observed time of events
		partd=torch.pow(torch.log(target + 1) - torch.log(lam*torch.pow(0.693,1/k) + 1),2)	#MSLE between predicted median and target
		uncensored = (-1 * (parta + partb + partc - partd*lagrange) * w) * events
		censored =  (-1 * partx * w) * (1 - events)
		return(torch.mean(uncensored + censored) + calibrationloss)
	else:
		uncensored = (-1 * (parta + partb + partc) * w) * events
		censored = (-1 * partc * w) * (1 - events)
		return(torch.mean(uncensored + censored))
#end of def


#it calculates a scalar loss between prediction and target
#loss_fn: a loss function
#pred: a tensor with (sample, timestep, outputfeature)
#tgt: a tensor with (timestep, sample)
#valid_ts: vector containing the valid timestep number for each sample
#events: vector containing the censoring status (1=uncensored, 0=censored)
#weights: optional, if presented, must have same length as pred.shape
def getlosshelper_exptube(loss_fn,pred,tgt,valid_ts,events,weights=None,compositeloss=False,
				tremain=None,epoch=None,epoch_mix_at=None,
				calibrationhorizon=None,calibrationchunk=None,device='cuda:0'):
	assert(pred.shape[0] == tgt.shape[1] and pred.shape[0] == len(valid_ts) and pred.shape[0] == len(events)),"pred,tgt,valid_ts length not match"
	if(not weights is None):
		assert(pred.shape[0] == len(weights)),"lenght of weights not match sample size"
	if(not tremain is None):
		assert(len(tremain) == pred.shape[1]), "length of tremain not match timestep size"
	loss=None
	losslist=[]
	for tsidx in range(pred.shape[1]):     #iterate each timestep
		validsmpidx=torch.where(valid_ts >= (tsidx + 1))[0]
		if(len(validsmpidx) > 0):
			w=None
			tremain_ts=None
			if(not weights is None):
				w=weights[validsmpidx]
			if(not tremain is None):
				tremain_ts=tremain[tsidx]
			losslist.append(loss_fn(pred[validsmpidx,tsidx,:],
					tgt[tsidx,validsmpidx],
					events[validsmpidx],
					w,
					compositeloss,
					tremain=tremain_ts,
					epoch=epoch,
					epoch_mix_at=epoch_mix_at,
					ts=tsidx+1,
					calibrationhorizon=calibrationhorizon,
					calibrationchunk=calibrationchunk,
					device=device
			))
		#end of if
	#end of for
	loss=torch.mean(torch.stack(losslist)).cpu().detach().numpy()
	#end of if
	return(loss)
#end of def

	
