import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
#from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score 
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve, auc 
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from pathlib import Path
import os
import pickle
import yaml


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('config.yaml', 'r') as file: # load yaml config
    CONFIG = yaml.safe_load(file)


def load_bird_list():
     # load birdnet indices for species in WABAD 
    birdnet_all_labels = Path(os.path.join(CONFIG["data_path"], "BirdNET_GLOBAL_6K_V2.4_Labels.txt")).read_text(encoding="utf-8").splitlines()
    all_latin_labels = [item.split('_')[0] for item in birdnet_all_labels]
    bird_list_index = [all_latin_labels.index(specie_name) for specie_name in CONFIG["bird_list_wabad"]]
    num_class =  len(bird_list_index)
    return bird_list_index, num_class


def load_birdnet_weights(bird_list_index): 
    #Load weights of pretrained birdNET last layer
    with open(os.path.join(CONFIG["data_path"], "birdnet_last_layer.pkl"), "rb") as file:
        loaded_data = pickle.load(file)
    [birdnet_weights, birdnet_bias] = loaded_data
    birdnet_weights = birdnet_weights[bird_list_index,:]
    birdnet_bias = birdnet_bias[bird_list_index]
    return  birdnet_weights, birdnet_bias


def import_dataset(bird_list_index): 
    # load full train, val, test sets of WABAD (european species)

    dataset_path = CONFIG["data_path"]

    # load train labels and sampling data
    with open(os.path.join(dataset_path, "dataframe_train.pkl"), "rb") as file:
        loaded_data = pickle.load(file)

    [samples_df_train, y_true_train, y_scores_train] = loaded_data

    y_true_train = y_true_train.astype(int)
    y_true_train = y_true_train[:,bird_list_index]
    birdnet_scores_train = y_scores_train[:,bird_list_index]
    wabad_train_set_size = y_scores_train.shape[0]

    # load test embeddings
    with open(os.path.join(dataset_path, "embeddings_train.pkl"), "rb") as file:
        loaded_data = pickle.load(file)
    [embeddings_train, files_list] = loaded_data

    # load validation labels
    with open(os.path.join(dataset_path, "dataframe_validation.pkl"), "rb") as file:
        loaded_data = pickle.load(file)
    [samples_df_val, y_true_val, y_scores_val] = loaded_data

    y_true_val = y_true_val.astype(int)
    y_true_val = y_true_val[:,bird_list_index]

    # load validation embeddings
    with open(os.path.join(dataset_path, "embeddings_validation.pkl"), "rb") as file:
        loaded_data = pickle.load(file)
    [embeddings_val, files_list] = loaded_data
    
    #load test labels
    with open(os.path.join(dataset_path, "dataframe_test.pkl"), "rb") as file:
        loaded_data = pickle.load(file)
    [samples_df_test, y_true_test, y_scores_test] = loaded_data

    y_true_test = y_true_test.astype(int)
    y_true_test = y_true_test[:,bird_list_index]
 
    #Load test embeddings
    with open(os.path.join(dataset_path, "embeddings_test.pkl"), "rb") as file:
        loaded_data = pickle.load(file)

    [embeddings_test, files_list_test] = loaded_data

    # load environemental noise
    with open(os.path.join(dataset_path, "dataframe_noise_esc50.pkl"), "rb") as file:
        loaded_data = pickle.load(file)

    [samples_df_noise, y_true_noise, y_scores_noise] = loaded_data
    esc50_set_size = y_scores_noise.shape[0]

    y_true_noise = y_true_noise.astype(int)
    y_true_noise = y_true_noise[:,bird_list_index]
    birdnet_scores_noise = y_scores_noise[:,bird_list_index]

    # load environemental noise embeddings
    with open(os.path.join(dataset_path, "embeddings_noise_esc50.pkl"), "rb") as file:
        loaded_data = pickle.load(file)
    [embeddings_noise, files_list] = loaded_data

    x_train = np.vstack((embeddings_train, embeddings_noise))
    y_train = np.vstack((y_true_train, y_true_noise))
    samples_df_train = pd.concat([samples_df_train, samples_df_noise], ignore_index=True)
    birdnet_scores_train = np.vstack((birdnet_scores_train,birdnet_scores_noise))

    x_val = embeddings_val
    y_val = y_true_val
    x_test = embeddings_test
    y_test = y_true_test

    return x_train, y_train, x_val, y_val, x_test, y_test, samples_df_train, birdnet_scores_train, wabad_train_set_size, esc50_set_size



def data_random_sampling(x_train, y_train, samples_num, proba_vect):
    # random samplin of "samples_num" samples in the train set, with probabilities in proba_vect

    idx_train_samples = np.arange(np.shape(x_train)[0])
    proba_vect_samples = proba_vect[idx_train_samples]/np.sum(proba_vect[idx_train_samples])

    idx_train_selection = np.random.choice(idx_train_samples, size=samples_num, p=proba_vect_samples, replace=False)

    x_train_sampling = x_train[idx_train_selection,:]
    y_train_sampling = y_train[idx_train_selection,:]    

    sampling_vect = np.zeros(np.shape(x_train)[0])
    sampling_vect[idx_train_selection] = 1

    return x_train_sampling, y_train_sampling, sampling_vect



def prepare_data(x_train,y_train ,x_val, y_val, x_test, y_test, batch_size=32):
    # create train, val, test data loader 

    tensor_x_train = torch.Tensor(x_train) 
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_val = torch.Tensor(x_val) 
    tensor_y_val = torch.Tensor(y_val)
    tensor_x_test = torch.Tensor(x_test) 
    tensor_y_test = torch.Tensor(y_test)

    train_dataset = TensorDataset(tensor_x_train,tensor_y_train) 
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(tensor_x_val,tensor_y_val) 
    val_loader = DataLoader(val_dataset,batch_size=batch_size) 

    test_dataset = TensorDataset(tensor_x_test,tensor_y_test) 
    test_loader = DataLoader(test_dataset,batch_size=batch_size) 

    return train_loader, val_loader, test_loader



class ClassificationHead(nn.Module):
    # Create linear model and initialize with birdnet weights

    def __init__(self, input_size, output_size, weight_matrix=None, bias_vector=None):
        super(ClassificationHead, self).__init__()
        # Define the linear layer
        self.fc = nn.Linear(input_size, output_size)
        
        # If a weight matrix is provided, initialize the weights
        if weight_matrix is not None:
            with torch.no_grad():  
                # Convert numpy array to tensor and set it as the weight of the layer
                self.fc.weight = nn.Parameter(torch.tensor(weight_matrix, dtype=torch.float32))
        
        # If a bias vector is provided, initialize the biases
        if bias_vector is not None:
            with torch.no_grad():  
                # Convert numpy array to tensor and set it as the bias of the layer
                self.fc.bias = nn.Parameter(torch.tensor(bias_vector, dtype=torch.float32))

    def forward(self, x):
        x = self.fc(x)
        return x



def train_model(train_loader, val_loader, num_epochs, model, optimizer, criterion, birdnet_weights, birdnet_bias, lambda_reg):
    # train model with  L2-SP regularization (minimize the distance between the model weights and the original weights of birdnet's last layer)

    birdnet_weights = torch.tensor(birdnet_weights, dtype=torch.float32).to(DEVICE)
    birdnet_bias = torch.tensor(birdnet_bias, dtype=torch.float32).to(DEVICE)

    train_losses = [] 
    val_losses = [] 

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Forward pass
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss = loss + lambda_reg*(torch.norm(model.fc.weight - birdnet_weights,2)**2 + torch.norm(model.fc.bias - birdnet_bias,2)**2)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Calculate running loss and accuracy
            running_loss += total_loss.item()

        epoch_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        if epoch % 10 == 0 :
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss = loss + lambda_reg*(torch.norm(model.fc.weight - birdnet_weights,2)**2 + torch.norm(model.fc.bias - birdnet_bias,2)**2)

                    val_loss += total_loss.item()

            epoch_val_loss = val_loss / len(val_loader)

            # Store losses and accuracies for plotting later
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

    return model, train_losses, val_losses


def model_inference(test_loader, model, criterion):

    # Set model to evaluation mode
    model.eval()
    # Initialize lists to store true and predicted labels
    all_probs_test = []
    # Disable gradient computation for validation
    with torch.no_grad():
        for inputs, labels in test_loader:  # Assuming validation_loader is defined
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Get model predictions
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_probs_test.extend(probs.cpu().detach().numpy())

    y_scores_test = np.array(all_probs_test)

    return y_scores_test


def sampling_strategy_evaluation(x_train_sampling, y_train_sampling, x_val, y_val, x_test, y_test, num_class,  birdnet_weights, birdnet_bias, display=False ):
    # one reverse correlation iteration: model creation, model training, model evaluation

    train_loader, val_loader, test_loader = prepare_data(x_train_sampling, y_train_sampling, x_val, y_val, x_test, y_test, batch_size=32)

    input_dim = x_train_sampling.shape[1]

    model = ClassificationHead(input_size=input_dim, output_size=num_class, weight_matrix=birdnet_weights, bias_vector=birdnet_bias).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    #best params 500 samples
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    num_epochs = 150

    model, train_losses, val_losses = train_model(train_loader, val_loader, num_epochs, model, optimizer, criterion, birdnet_weights, birdnet_bias, lambda_reg=0.00032)

    y_scores_test = model_inference(test_loader, model, criterion)

    class_AP = average_precision_score(y_test, y_scores_test, average=None, sample_weight=None)
    mAP = average_precision_score(y_test, y_scores_test, average="weighted", sample_weight=None)
    cmAP = average_precision_score(y_test, y_scores_test , average="macro", sample_weight=None)

    if display==True:
        epochs_range = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

    return mAP, cmAP, class_AP