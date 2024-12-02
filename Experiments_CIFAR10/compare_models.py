import os
import wandb
import argparse, yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

import torch

from airbench94 import CifarLoader, make_net, evaluate

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from features_alignment import calculate_adjacency_matrix, adjacency_matrices_sum

hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
    'num_models' : 100           # number of "new" models 
}

# Simple function that creates a dataframe given a matrix, and plots its heatmap.
# Only creates the figure if it is going to be used
def df_plot_heatmap(matrix, classes, title, cmap, fmt, center=None, display= True):
    df = pd.DataFrame(matrix, index = [i for i in classes], columns = [i for i in classes])
    if display:
        fig=plt.figure(figsize = (12,7))
        sns.heatmap(df, cmap=cmap, annot=True, fmt=fmt, linewidth=.5, center=center, square= True, robust= True)
        plt.title(title)
        plt.xlabel("New prediction")
        plt.ylabel("Old prediction")
    else: 
        fig = None
    return df, fig

# Calculate a negative flips heatmap showing how many time the old model's predictions (y axes) 
# got wrongly predicted as another class by the new model (x axes)
def flips_heatmap(flips, num_flips, classes, display= True):
    flip_matrix = np.zeros((len(classes), len(classes)), dtype = int)
    percentage_flip_matrix = np.zeros((len(classes), len(classes)), dtype = float)

    # create flip matrix
    for key in flips:
        flip_matrix[key[0],key[1]] = flips[key]
        percentage_flip_matrix[key[0],key[1]] = flips[key]*100/num_flips
    
    df_flips, fig = df_plot_heatmap(flip_matrix, classes,'Negative flips', 'Purples', 'd', display=display )
    df_flips_perc, fig_perc = df_plot_heatmap(percentage_flip_matrix, classes,'Percentage of negative flips','Purples', '.1f',  display=display )
    
    return fig, df_flips, fig_perc, df_flips_perc

# Create a new heatmap where the negative flips that correspond to two adjacent classes are the same
# while the others become negative
def link_NFR_adjacency(nfr_heatmap, adj_heatmap, display= True):
    nfr_matrix, adj_matrix = nfr_heatmap.to_numpy(), adj_heatmap.to_numpy()

    # Replace all the zeros with -1
    tmp_adj_matrix = np.copy(adj_matrix)
    tmp_adj_matrix[ tmp_adj_matrix == 0] = -1

    nfr_adj = np.multiply(nfr_matrix, tmp_adj_matrix)
    cmap = LinearSegmentedColormap.from_list('', ['red', 'white', 'blue'])
    df_nfr_adj, fig = df_plot_heatmap(nfr_adj,nfr_heatmap.index, 'Negative flips - Adjacency matrix', cmap, '.1f', center=0, display=display )

    return fig, df_nfr_adj

# Creates, prints and plots a summary of all the NFR - adjacency matrices 
def summary_nfr_adj(nfr_adj_matrices, classes, impr = ""):
    # Count all the negative flips 
    sum_nfr_adj_mat =  np.sum(np.absolute(nfr_adj_matrices), axis = 0)
    df_sums, fig_sum = df_plot_heatmap(sum_nfr_adj_mat, classes,"All "+ impr+ " negative flips'", 'Purples', 'd')
    print_and_log(df_sums, fig_sum, "All "+ impr+ " negative flips'")
    total_flips = np.sum(sum_nfr_adj_mat)

    # Replace all the positive numbers with 0, count all the negative flips between non-adjacent classes
    tmp_negative = np.copy(nfr_adj_matrices)
    tmp_negative[ tmp_negative >0] = 0
    negative_nfr_adj_mat = np.sum(np.absolute(tmp_negative), axis =0)
    neg_perc_nfr_adj_mat = negative_nfr_adj_mat* 100/total_flips
    df_neg, fig_neg = df_plot_heatmap(negative_nfr_adj_mat, classes,impr+' Negative flips of non-adjacent classes', 'Reds', 'd')
    print_and_log(df_neg, fig_neg, impr+' Negative flips of non-adjacent classes')
    df_neg_per, fig_neg_per = df_plot_heatmap(neg_perc_nfr_adj_mat, classes,impr+ " Negative flips of non-adjacent classes' percentage ", 'Reds', '.1f')
    print_and_log(df_neg_per, fig_neg_per, impr+" Negative flips of non-adjacent classes' percentage ")
    neg_per = np.sum(neg_perc_nfr_adj_mat)

    # Replace all the negative numbers with 0, count all the negative flips between adjacent classes
    tmp_positive = np.copy(nfr_adj_matrices)
    tmp_positive[ tmp_positive <0] = 0
    positive_nfr_adj_mat = np.sum(tmp_positive, axis =0)
    potitive_perc_nfr_adj_mat = positive_nfr_adj_mat* 100/total_flips
    df_pos, fig_pos = df_plot_heatmap(positive_nfr_adj_mat, classes,impr+' Negative flips of adjacent classes', 'Blues', 'd')
    print_and_log(df_pos, fig_pos,impr+' Negative flips of adjacent classes')
    df_pos_per, fig_pos_per = df_plot_heatmap(potitive_perc_nfr_adj_mat, classes, impr+" Negative flips of adjacent classes' percentage ", 'Blues', '.1f')
    print_and_log(df_pos_per, fig_pos_per,impr+" Negative flips of adjacent classes' percentage ")
    pos_per = np.sum(potitive_perc_nfr_adj_mat)
    return neg_per, pos_per

# Simple function that prints and log a dataframe and a figure given their title
def print_and_log(df, fig, title):
    print(title+ f":\n {df}")
    wandb.log({title: wandb.Image(fig)})


def main():

    # Get env variables
    load_dotenv()

    # Select config 
    parser = argparse.ArgumentParser(description='Calculating Negative Flip Rate')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    old_model_name = loaded_params['old_model_name']
    new_model_name = loaded_params['new_model_name']

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "NFR 7 classes CIFAR10_" + str(hyp['num_models'])+"models",
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Get model with worse performances
    model_v1 = make_net()
    artifact = wandb_run.use_artifact(WANDB_PROJECT+old_model_name, type='model')
    artifact_dir = artifact.download()
    model_v1.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    adj_matrices = []
    flip_matrices, flip_per_mat, nfr_adj_matrices  = [], [], []
    impr_flip_matrices, impr_flip_per_mat, impr_nfr_adj_matrices  = [], [], []

    for i in range(hyp['num_models']):
        # Get model with better performances
        model_v2 = make_net()
        if hyp['num_models'] > 1:
            current_model_name = new_model_name.split(":v")[0]+":v"+str(i+int(new_model_name.split(":v")[1]))
            display_all = False
        else:
            current_model_name = new_model_name
            display_all = True
        print(current_model_name)
        artifact = wandb_run.use_artifact(WANDB_PROJECT+current_model_name, type='model')
        artifact_dir = artifact.download()
        model_v2.load_state_dict(torch.load(artifact_dir+'/model.pth'))

        # Get the better model's last layer's weigths and calculate its adjacency matrix
        W_v2 = model_v2[8].weight.detach().cpu().numpy().astype(np.float32)
        adjacency_matrix_v2=calculate_adjacency_matrix(W_v2)
        df_adjacency_v2, fig= df_plot_heatmap(adjacency_matrix_v2, classes,'Adjacency matrix', 'Purples', 'd', display=display_all )

        if hyp['num_models'] == 1:
            print_and_log(df_adjacency_v2, fig,"Adjacency matrix" )
        else: 
            adj_matrices.append(adjacency_matrix_v2)

        # Evaluate both models
        tta_val_acc_v1 = evaluate(model_v1, test_loader, tta_level=hyp['net']['tta_level'])
        print("tta_val_acc model v1: "+ str(tta_val_acc_v1))
        tta_val_acc_v2 = evaluate(model_v2, test_loader, tta_level=hyp['net']['tta_level'])
        print("tta_val_acc model v2: "+ str(tta_val_acc_v2))

        # Calculate the NFR, print the flip heatmaps and connect it to the adjacency matrix 
        nfr, flips, num_flips = negative_flip_rate(model_v1, model_v2, test_loader)
        print("Negative flip rate: "+ str(nfr))
        relative_nfr = relative_negative_flip_rate(nfr, tta_val_acc_v1, tta_val_acc_v2)
        print("Relative negative flip rate: "+ str(relative_nfr))
        flips_fig, flips_df, flips_fig_per, flips_df_per = flips_heatmap(flips, num_flips, classes, display=display_all)
        fig_nfr_adj, df_nfr_adj = link_NFR_adjacency(flips_df, df_adjacency_v2,  display=display_all)
        if hyp['num_models'] == 1:
            print_and_log(flips_df, flips_fig,"Heatmap of the negative flips between classes" )
            print_and_log(flips_df_per, flips_fig_per,"Heatmap of the negative flips' percentage between classes"  )
            print_and_log(df_nfr_adj,fig_nfr_adj, 'Heatmap of the negative flips - adjacency matrix' )
        else: 
            flip_matrices.append(flips_df.to_numpy())
            flip_per_mat.append(flips_df_per.to_numpy())
            nfr_adj_matrices.append(df_nfr_adj.to_numpy())

        # Same as before but with the improved NFR
        improved_nfr, impr_flips, impr_num_flips = improved_negative_flip_rate(model_v1, model_v2, test_loader)
        print("Improved negative flip rate: "+ str(improved_nfr))
        improved_relative_nfr = relative_negative_flip_rate(improved_nfr, tta_val_acc_v1, tta_val_acc_v2)
        print("Improved relative negative flip rate: "+ str(improved_relative_nfr))
        impr_flips_fig, impr_flips_df, impr_flips_fig_per, impr_flips_df_per = flips_heatmap(impr_flips, impr_num_flips, classes,  display=display_all)
        impr_fig_nfr_adj, impr_df_nfr_adj = link_NFR_adjacency(impr_flips_df, df_adjacency_v2,  display=display_all)
        if hyp['num_models'] == 1:
            print_and_log(impr_flips_df,impr_flips_fig, "Heatmap of the improved negative flips between classes" )
            print_and_log(impr_flips_df_per, impr_flips_fig_per, "Heatmap of the improved negative flips' percentage between classes")
            print_and_log(impr_df_nfr_adj, impr_fig_nfr_adj , 'Heatmap of the improved negative flips - adjacency matrix')
        else: 
            impr_flip_matrices.append(impr_flips_df.to_numpy())
            impr_flip_per_mat.append(impr_flips_df_per.to_numpy())
            impr_nfr_adj_matrices.append(impr_df_nfr_adj.to_numpy())
        
        eval_metrics = {'tta_val_acc_v1': tta_val_acc_v1, 'tta_val_acc_v2': tta_val_acc_v2,
                        'NFR': nfr, 'improved_NFR': improved_nfr,
                        'Relative NFR':relative_nfr , 'Improved relative NFR':improved_relative_nfr}
        wandb.log({**eval_metrics})
    if hyp['num_models']>1:
        #Calculate and show the sum of all the adjacency matrices
        df_sum_adj, df_sum_per_adj, fig_sum_adj, fig_per_adj= adjacency_matrices_sum(np.array(adj_matrices), classes)
        print_and_log(df_sum_adj,fig_sum_adj,'Heatmap of the sum of all adjacency matrices' )
        print_and_log(df_sum_per_adj, fig_per_adj,'Heatmap of the percentage of all adjacency matrices' )
        
        # Calculate and print/log the mean, min, max and std NFR heatmap and its connection to the adjacency matrices
        mean_flip_mat = np.mean(np.array(flip_matrices), axis = 0)
        df_mean_flip, fig_mean_flip= df_plot_heatmap(mean_flip_mat, classes, 'Mean negative flips', 'Purples', '.1f')
        print_and_log(df_mean_flip, fig_mean_flip,"Mean heatmap of the negative flips between classes")
        mean_flip_per_mat =  np.mean(np.array(flip_per_mat), axis = 0)
        df_mean_flip_per, fig_mean_flip_per= df_plot_heatmap(mean_flip_per_mat, classes, 'Mean percentage of negative flips', 'Purples', '.1f')
        print_and_log(df_mean_flip_per, fig_mean_flip_per,"Mean heatmap of the negative flips' percentage between classes")
        neg_per, pos_per = summary_nfr_adj(nfr_adj_matrices, classes)
        
        # Same code but with the improved NFR
        mean_impr_flip_mat = np.mean(np.array(impr_flip_matrices), axis = 0)
        df_impr_mean_flip, fig_impr_mean_flip= df_plot_heatmap(mean_impr_flip_mat, classes, 'Mean improved negative flips', 'Purples', '.1f')
        print_and_log(df_impr_mean_flip, fig_impr_mean_flip,"Mean heatmap of the improved negative flips between classes")
        mean_impr_per_flip_mat = np.mean(np.array(impr_flip_per_mat), axis = 0)
        df_mean_impr_flip_per, fig_mean_impr_flip_per= df_plot_heatmap(mean_impr_per_flip_mat, classes, 'Mean percentage of improved negative flips', 'Purples', '.1f')
        print_and_log(df_mean_impr_flip_per, fig_mean_impr_flip_per,"Mean heatmap of the improved negative flips' percentage between classes")
        impr_neg_per, impr_pos_per= summary_nfr_adj(impr_nfr_adj_matrices, classes, impr = "Improved")
        
        per_metrics = {'Non-adjacent negative flips percentage': neg_per,
                        'Adjacent negative flips percentage' : pos_per,
                        'Non-adjacent improved negative flips percentage': impr_neg_per,
                        'Adjacent improved negative flips percentage' : impr_pos_per}
        wandb.log({**per_metrics})
        
    wandb_run.finish()


if __name__ == '__main__':
    main()