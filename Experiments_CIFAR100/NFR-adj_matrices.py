import os
import wandb
import argparse, yaml
import numpy as np
from dotenv import load_dotenv
import torch

from dataset import *
from models import *
from testing import test_model
from utils import CIFAR100_CLASSES

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from features_alignment import calculate_adjacency_matrix, adjacency_matrices_sum
from print_and_logging import *
from link_NFR_adj_matrices import *

hyp = {
    'opt': {
        'batch_size': 128, 
    },
    'net': {
        'backbone': 'resnet18',         # resnet 18/34/50/101/152
        'feat_dim' : 3,                 # features' dimension
    },
    'data': {
        'num_classes': 100,
        'subset_list': None,        #es: list [0,50,1] corresponds to list(range(0,50,1))
        'old_subset_list': None,    # subset list of the old model, to create the correct dataloader to calculate NFR
    },
    'dSimplex': False,              # if the classifier is a dSimplex
    'num_models': 5,                
    'old_model_name' : None,        # the name of the worse older model that is going to be used in NFR calculation
}

# Transfer each loaded parameter to the correct hyp parameter
def define_hyp(loaded_params):
    hyp['net']['backbone'] = loaded_params['backbone']
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['num_models'] = loaded_params['num_models']
    hyp['data']['num_classes'] = loaded_params['num_classes']
    subset_list = loaded_params['subset_list']
    if subset_list is not None:
        hyp['data']['subset_list']  = list(range(subset_list[0], subset_list[1], subset_list[2]))
    hyp['opt']['batch_size'] = loaded_params['batch_size']
   
    hyp['old_model_name'] = loaded_params['old_model_name']
    old_subset_list = loaded_params['old_subset_list']
    hyp['data']['old_subset_list'] = list(range(old_subset_list[0], old_subset_list[1], old_subset_list[2]))

    hyp['dSimplex'] = loaded_params['dSimplex']

def main():

    # Get env variables
    load_dotenv()

    # Select config 
    parser = argparse.ArgumentParser(description='Calculating connection between Negative Flip Rate and adjacency matrices')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    new_model_name = loaded_params['model_name']
    define_hyp(loaded_params)

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')
    DATASET_PATH = os.getenv('DATASET_PATH')

    wandb.login(key= WANDB_API_KEY, verify=True)

    device = torch.device("cuda:0")

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "NFR-adj matrices "+ new_model_name.split(":v")[0]+"&" + hyp['old_model_name'].split(":v")[0],
        config=hyp)
    
    # Get test images
    _, cifar100_test_loader = create_dataloaders('cifar100', DATASET_PATH, hyp['opt']['batch_size'], 
                                                                subset_list= hyp['data']['subset_list'])
    classes = CIFAR100_CLASSES
    # Get the correct dataset to test the NFR
    _, cifar100_nfr_test_loader = create_dataloaders('cifar100', DATASET_PATH,hyp['opt']['batch_size'],subset_list= hyp['data']['old_subset_list'])

    # Get model with worse performances
    model_v1 = create_model(hyp['net']['backbone'], False, hyp['net']['feat_dim'], hyp['data']['num_classes'], 
                                    device, WANDB_PROJECT+hyp['old_model_name'], wandb_run )

    adj_matrices = []
    flip_matrices, flip_per_mat, nfr_adj_matrices  = [], [], []
    impr_flip_matrices, impr_flip_per_mat, impr_nfr_adj_matrices  = [], [], []

    for i in range(hyp['num_models']):
        if hyp['num_models'] > 1:
            current_model_name = new_model_name.split(":v")[0]+":v"+str(i+int(new_model_name.split(":v")[1]))
            display_all = False
        else:
            current_model_name = new_model_name
            display_all = True

        # Get model with better performances
        model_v2 =  create_model( hyp['net']['backbone'],False, hyp['net']['feat_dim'],hyp['data']['num_classes'], device,
                         WANDB_PROJECT+current_model_name, wandb_run )
        
        print(current_model_name)

        # Get the better model's last layer's weigths and calculate its adjacency matrix
        W_v2 = model_v2.fc2.weight.detach().cpu().numpy().astype(np.float32)
        adjacency_matrix_v2=calculate_adjacency_matrix(W_v2)
        df_adjacency_v2, fig= df_plot_heatmap(adjacency_matrix_v2, classes,'Adjacency matrix', 'Purples', 'd', "", "", display=display_all,
                                              figsize=( 20, 14), annot = False )

        if hyp['num_models'] == 1:
            print_and_log(df_adjacency_v2, fig,"Adjacency matrix" )
        else: 
            adj_matrices.append(adjacency_matrix_v2)

        # Evaluate both models
        test_acc_v1, _ = test_model(model_v1, cifar100_test_loader, wandb_run, step = i, wandb_log_name= " Old model")
        test_acc_v2, _ = test_model(model_v2, cifar100_test_loader, wandb_run, step = i, wandb_log_name= " New model")

        # Calculate the NFR, print the flip heatmaps and connect it to the adjacency matrix 
        nfr, flips, num_flips = negative_flip_rate(model_v1, model_v2, cifar100_nfr_test_loader, dict_output= True)
        print("Negative flip rate: "+ str(nfr))
        relative_nfr = relative_negative_flip_rate(nfr, test_acc_v1.detach().cpu().numpy(), test_acc_v2.detach().cpu().numpy())
        print("Relative negative flip rate: "+ str(relative_nfr))
        flips_fig, flips_df, _, _ = flips_heatmap(flips, num_flips, classes, display=display_all, figsize=( 20, 14), annot = False)
        fig_nfr_adj, df_nfr_adj = link_NFR_adjacency(flips_df, df_adjacency_v2,  display=display_all, figsize=( 20, 14), annot = False)
        if hyp['num_models'] == 1:
            print_and_log(flips_df, flips_fig,"Heatmap of the negative flips between classes" )
            print_and_log(df_nfr_adj,fig_nfr_adj, 'Heatmap of the negative flips - adjacency matrix' )
        else: 
            flip_matrices.append(flips_df.to_numpy())
            nfr_adj_matrices.append(df_nfr_adj.to_numpy())

        # Same as before but with the improved NFR
        improved_nfr, impr_flips, impr_num_flips = improved_negative_flip_rate(model_v1, model_v2, cifar100_nfr_test_loader, dict_output= True)
        print("Improved negative flip rate: "+ str(improved_nfr))
        improved_relative_nfr = relative_negative_flip_rate(improved_nfr, test_acc_v1.detach().cpu().numpy(), test_acc_v2.detach().cpu().numpy())
        print("Improved relative negative flip rate: "+ str(improved_relative_nfr))
        impr_flips_fig, impr_flips_df,_ , _ = flips_heatmap(impr_flips, impr_num_flips, classes,  display=display_all,
                                                                                              figsize=( 20, 14), annot = False)
        impr_fig_nfr_adj, impr_df_nfr_adj = link_NFR_adjacency(impr_flips_df, df_adjacency_v2,  display=display_all, figsize=( 20, 14), annot = False)
        if hyp['num_models'] == 1:
            print_and_log(impr_flips_df,impr_flips_fig, "Heatmap of the improved negative flips between classes" )
            print_and_log(impr_df_nfr_adj, impr_fig_nfr_adj , 'Heatmap of the improved negative flips - adjacency matrix')
        else: 
            impr_flip_matrices.append(impr_flips_df.to_numpy())
            impr_nfr_adj_matrices.append(impr_df_nfr_adj.to_numpy())
        
        nfr_metrics = {'NFR': nfr, 'improved_NFR': improved_nfr,
                        'Relative NFR':relative_nfr , 'Improved relative NFR':improved_relative_nfr}
        wandb.log({**nfr_metrics})
    if hyp['num_models']>1:
        #Calculate and show the sum of all the adjacency matrices
        df_sum_adj, _, fig_sum_adj, _= adjacency_matrices_sum(np.array(adj_matrices), classes,  figsize=( 20, 14), annot = False)
        print_and_log(df_sum_adj,fig_sum_adj,'Heatmap of the sum of all adjacency matrices' )
        
        # Calculate and print/log the mean, min, max and std NFR heatmap and its connection to the adjacency matrices
        mean_flip_mat = np.mean(np.array(flip_matrices), axis = 0)
        df_mean_flip, fig_mean_flip= df_plot_heatmap(mean_flip_mat, classes, 'Mean negative flips', 'Purples', '.1f', "New prediction", "Old predictions",
                                                      figsize=( 20, 14), annot = False)
        print_and_log(df_mean_flip, fig_mean_flip,"Mean heatmap of the negative flips between classes")
        neg_per, pos_per = summary_nfr_adj(nfr_adj_matrices, classes, figsize=( 20, 14), annot = False)
        
        # Same code but with the improved NFR
        mean_impr_flip_mat = np.mean(np.array(impr_flip_matrices), axis = 0)
        df_impr_mean_flip, fig_impr_mean_flip= df_plot_heatmap(mean_impr_flip_mat, classes, 'Mean improved negative flips', 'Purples', '.1f',
                                                                "New prediction", "Old predictions",  figsize=( 20, 14), annot = False)
        print_and_log(df_impr_mean_flip, fig_impr_mean_flip,"Mean heatmap of the improved negative flips between classes")
        impr_neg_per, impr_pos_per= summary_nfr_adj(impr_nfr_adj_matrices, classes, impr = "Improved", figsize=( 20, 14), annot = False)
        
        per_metrics = {'Non-adjacent negative flips percentage': neg_per,
                        'Adjacent negative flips percentage' : pos_per,
                        'Non-adjacent improved negative flips percentage': impr_neg_per,
                        'Adjacent improved negative flips percentage' : impr_pos_per}
        wandb.log({**per_metrics})
        
    wandb_run.finish()


if __name__ == '__main__':
    main()