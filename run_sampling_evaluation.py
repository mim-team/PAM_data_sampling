import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from pathlib import Path
import os
import sys
import random
import pickle
from tqdm import tqdm
import yaml

from utils.model import load_bird_list, load_birdnet_weights, import_dataset, sampling_strategy_evaluation


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

with open('config.yaml', 'r') as file: # load yaml config
    CONFIG = yaml.safe_load(file)



def calculate_draw_probability(target_proportion, wabad_train_sets_size, noise_sets_size ):

    ## Define the probablity of drawing each sample to virtually increase the number of non-event samples
    # target proportion of "samples with birds" in the total train set (compared to noise samples from esc50)

    #  probability of drawing noise samples to reach the target proportion 
    noise_proba = (wabad_train_sets_size / target_proportion - wabad_train_sets_size) / noise_sets_size

    proba_vect = np.concatenate([ np.ones(wabad_train_sets_size), np.ones(noise_sets_size)* noise_proba])
    proba_vect = proba_vect/np.sum(proba_vect)

    return proba_vect



def generate_iter_sampling_thresholds(sampling_parameter, sign, iter_per_condition, samples_df_train):
    
    sampling_parameter_min = CONFIG["sampling_parameter_min"] * 2
    sampling_parameter_max = CONFIG["sampling_parameter_max"] * 2

    param_thresholds_list = []
    for i, parameter in enumerate(sampling_parameter):
        samples_df_train[parameter]
        for j in range(iter_per_condition):

            range_l = sampling_parameter_min[i]
            range_h = sampling_parameter_max[i]
            range_order = range_h-range_l
        
            sampling_range =  np.linspace(range_l-0.1*range_order, range_h +0.1*range_order, 1000)
            sampling_range = np.clip(sampling_range, np.min(samples_df_train[parameter]),  np.max(samples_df_train[parameter]) )
            
            value = np.random.choice(sampling_range, size =1)[0]

            param_thresholds_list.append( [parameter, sign[i], float(value)])


    return param_thresholds_list



def data_sampling(x_train, y_train, samples_num, param_thresholds, samples_df_train, proba_vect):


    [sampling_param, sign, threshold] = param_thresholds

    selection_vect = sign*samples_df_train[sampling_param] >= sign*threshold

    idx_train_samples = samples_df_train[selection_vect].index
    
    ## Random subsampling
    if np.size(idx_train_samples) > samples_num:
        sampling_done = True
        proba_vect_samples = proba_vect[idx_train_samples]/np.sum(proba_vect[idx_train_samples])
        idx_train_selection = np.random.choice(idx_train_samples, size=samples_num, p=proba_vect_samples, replace=False)
        x_train_sampling = x_train[idx_train_selection,:]
        y_train_sampling = y_train[idx_train_selection,:]    
    else:
        #print(f"Error : less than {samples_num:.0f} samples in the selection" )
        sampling_done = False
        x_train_sampling = 0
        y_train_sampling = 0

    return x_train_sampling, y_train_sampling, sampling_done




### evaluate vairous random sampling strategies
if __name__ == "__main__":

    bird_list_index, num_class = load_bird_list() # load birdnet indices for species in WABAD
    
    birdnet_weights, birdnet_bias = load_birdnet_weights(bird_list_index) # load birdnet last layer weights


    x_train, y_train, x_val, y_val, x_test, y_test, samples_df_train, birdnet_scores_train,  wabad_train_set_size, esc50_set_size = import_dataset(bird_list_index)
    
    proba_vect = calculate_draw_probability(CONFIG["bird_sample_prop"], wabad_train_set_size, esc50_set_size )
    
    sampling_parameters = CONFIG["sampling_parameters"] * 2
    sampling_parameters_sign = [1]*int(len(sampling_parameters)/2) + [-1]*int(len(sampling_parameters)/2) 

    param_thresholds_plan = generate_iter_sampling_thresholds( sampling_parameters , sampling_parameters_sign, CONFIG["iter_per_condition"], samples_df_train)

    param_thresholds_list = []
    mAP_list = []
    cmAP_list = []
    class_AP_list = []

    iter_num = len(sampling_parameters)*CONFIG["iter_per_condition"]

    for i in tqdm(range(iter_num)):
        
        param_threshold = param_thresholds_plan[i]
        x_train_sampling, y_train_sampling, sampling_done = data_sampling(x_train, y_train, CONFIG["sample_num"] , param_threshold , samples_df_train, proba_vect)

        if sampling_done == True:

            mAP, cmAP, class_AP = sampling_strategy_evaluation( x_train_sampling, y_train_sampling, x_val, y_val, x_test, y_test, num_class, birdnet_weights, birdnet_bias, display=False )
            
            param_thresholds_list.append(param_threshold)
            mAP_list.append(mAP)
            cmAP_list.append(cmAP)
            class_AP_list.append(np.array(class_AP))

            print(f"params = {param_threshold}, mAP = {mAP:.3f}, cmAP = {cmAP:.3f}" )

        else :
            print("Sampling too small, skip training")


    param_thresholds_list = np.array(param_thresholds_list)
    mAP_list = np.array(mAP_list)
    cmAP_list = np.array(cmAP_list)
    class_AP_list = np.array(class_AP_list)

    print((param_thresholds_list))
    print(np.shape(mAP_list))
    #save results
    with open(os.path.join(CONFIG["results_path"],"results_sampling_evaluation.pkl"), "wb") as file:
        pickle.dump([param_thresholds_list, mAP_list, cmAP_list , class_AP_list, sampling_parameters,  sampling_parameters_sign] ,file)

