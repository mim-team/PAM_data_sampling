import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import yaml
import torch
from utils.model import load_bird_list, load_birdnet_weights, import_dataset, data_random_sampling, sampling_strategy_evaluation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
print( torch.cuda.device_count())
print(torch.version.cuda)

with open('config.yaml', 'r') as file: # load yaml config
    CONFIG = yaml.safe_load(file)


if __name__ == "__main__":

    bird_list_index, num_class = load_bird_list() # load birdnet indices for species in WABAD

    birdnet_weights, birdnet_bias = load_birdnet_weights(bird_list_index) # load birdnet last layer weights

    x_train, y_train, x_val, y_val, x_test, y_test, _ , _,  _, _ = import_dataset(bird_list_index) # load train, val and test set
    
    proba_vect = np.ones(np.shape(x_train)[0]) # equal drawing probabilities for each sample

    sampling_vect_list = []
    mAP_list = []
    cmAP_list = []
    class_AP_list = []

    for i in tqdm(range(CONFIG["iter_num"])):
        # data sampling : random selection of "sample_num" samples in the train set
        x_train_sampling, y_train_sampling, sampling_vect = data_random_sampling(x_train, y_train, CONFIG["sample_num"], proba_vect) 
        
        # one reverse correlation iteration: model creation, model training, model evaluation
        mAP, cmAP, class_AP = sampling_strategy_evaluation( x_train_sampling, y_train_sampling, x_val, y_val, x_test, y_test, num_class, birdnet_weights, birdnet_bias, display=False )

        sampling_vect_list.append(sampling_vect)
        mAP_list.append(mAP)
        cmAP_list.append(cmAP)
        class_AP_list.append(np.array(class_AP))
        print(f"params = {i}, mAP = {mAP:.3f}, cmAP = {cmAP:.3f}" )


    sampling_vect_list = np.array(sampling_vect_list)
    mAP_list = np.array(mAP_list)
    cmAP_list = np.array(cmAP_list)
    class_AP_list = np.array(class_AP_list)
    
    #save results
    with open(os.path.join(CONFIG["data_path"],"results_samples_revcor.pkl"), "wb") as file:
        pickle.dump([sampling_vect_list, mAP_list, cmAP_list , class_AP_list] ,file)
