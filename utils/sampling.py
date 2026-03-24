import numpy as np
import pandas as pd


def calculate_draw_probability(target_proportion, wabad_train_sets_size, noise_sets_size ):

    ## Define the probablity of drawing each sample to virtually increase the number of non-event samples
    # target proportion of "samples with birds" in the total train set (compared to noise samples from esc50)

    #  probability of drawing noise samples to reach the target proportion 
    noise_proba = (wabad_train_sets_size / target_proportion - wabad_train_sets_size) / noise_sets_size

    proba_vect = np.concatenate([ np.ones(wabad_train_sets_size), np.ones(noise_sets_size)* noise_proba])
    proba_vect = proba_vect/np.sum(proba_vect)

    return proba_vect




def data_sampling(x_train, y_train, samples_num, param_thresholds, samples_df_train, proba_vect):


    if param_thresholds[0] == "Random":
        
        idx_train_samples = samples_df_train.index

    else :
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