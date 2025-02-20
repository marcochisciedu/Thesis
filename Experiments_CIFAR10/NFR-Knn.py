import os
import wandb
import argparse, yaml
import numpy as np
from dotenv import load_dotenv
import torch

from airbench94 import CifarLoader, make_net, evaluate

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from features_alignment import calculate_knn_matrix, knn_matrices_sum
from print_and_logging import *
from link_NFR_adj_matrices import *

hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
        'feat_dim' : 3,         # features' dimension
        'num_classes': 10,              # num classes of the new model
        'old_num_classes': 5,           # num classes of the old model
    },
    'num_models' : 100,          # number of "new" models 
    'k': 1,                      # number of neighbors    
}

"""
Code to search for the possible correlation between negative flip and the prototypes' KNN (k-Nearest Neighbors) matrix 
"""

def main():

    # Get env variables
    load_dotenv()

    # Select config 
    parser = argparse.ArgumentParser(description='Calculating relationship between NFR and Knn')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['net']['num_classes'] = loaded_params['num_classes']
    old_model_name = loaded_params['old_model_name']
    new_model_name = loaded_params['new_model_name']
    hyp['num_models'] = loaded_params['num_models']
    hyp['k'] = loaded_params['k']
    hyp['low_class_list'] = loaded_params['low_class_list']

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "NFR-Knn, k: "+ str(hyp['k'])+ " "+ new_model_name.split(":v")[0]+"&" + old_model_name.split(":v")[0],
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000,
                                   list_low_classes= hyp['low_class_list'], low_percentage= [0]*len(hyp['low_class_list']))
    classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Get model with worse performances
    model_v1 = make_net( hyp['net']['feat_dim'], hyp['net']['old_num_classes'])
    artifact = wandb_run.use_artifact(WANDB_PROJECT+old_model_name, type='model')
    artifact_dir = artifact.download()
    model_v1.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    knn_matrices = []
    flip_matrices, flip_per_mat, nfr_knn_matrices  = [], [], []
    impr_flip_matrices, impr_flip_per_mat, impr_nfr_knn_matrices  = [], [], []

    for i in range(hyp['num_models']):
        # Get model with better performances
        model_v2 = make_net( hyp['net']['feat_dim'], hyp['net']['num_classes'])
        if hyp['num_models'] > 1:
            current_model_name = new_model_name.split(":v")[0]+":v"+str(i+int(new_model_name.split(":v")[1]))
            display_all = False  # do not display every knn matrix
        else:
            current_model_name = new_model_name
            display_all = True
        print(current_model_name)
        artifact = wandb_run.use_artifact(WANDB_PROJECT+current_model_name, type='model')
        artifact_dir = artifact.download()
        model_v2.load_state_dict(torch.load(artifact_dir+'/model.pth'))

        # Get the better model's last layer's weigths and calculate its Knn matrix
        W_v2 = model_v2[-2].weight.detach().cpu().numpy().astype(np.float32)
        knn_matrix_v2=calculate_knn_matrix(W_v2, hyp['k'])
        df_knn_v2, fig= df_plot_heatmap(knn_matrix_v2, classes,'Knn matrix', 'Purples', 'd', "", "", display=display_all )

        if hyp['num_models'] == 1:
            print_and_log(df_knn_v2, fig, "Knn matrix" )
        else: 
            knn_matrices.append(knn_matrix_v2)

        # Evaluate both models
        tta_val_acc_v1 = evaluate(model_v1, test_loader, tta_level=hyp['net']['tta_level'])
        print("tta_val_acc model v1: "+ str(tta_val_acc_v1))
        tta_val_acc_v2 = evaluate(model_v2, test_loader, tta_level=hyp['net']['tta_level'])
        print("tta_val_acc model v2: "+ str(tta_val_acc_v2))

        # Calculate the NFR, print the flip heatmaps and connect it to the knn matrix 
        nfr, flips, num_flips = negative_flip_rate(model_v1, model_v2, test_loader)
        print("Negative flip rate: "+ str(nfr))
        relative_nfr = relative_negative_flip_rate(nfr, tta_val_acc_v1, tta_val_acc_v2)
        print("Relative negative flip rate: "+ str(relative_nfr))
        flips_fig, flips_df, flips_fig_per, flips_df_per = flips_heatmap(flips, num_flips, classes, display=display_all)
        fig_nfr_knn, df_nfr_knn = link_NFR_adjacency(flips_df, df_knn_v2,  display=display_all)
        if hyp['num_models'] == 1:
            print_and_log(flips_df, flips_fig,"Heatmap of the negative flips between classes" )
            print_and_log(flips_df_per, flips_fig_per,"Heatmap of the negative flips' percentage between classes"  )
            print_and_log(df_nfr_knn,fig_nfr_knn, 'Heatmap of the negative flips - Knn matrix' )
        else: 
            flip_matrices.append(flips_df.to_numpy())
            flip_per_mat.append(flips_df_per.to_numpy())
            nfr_knn_matrices.append(df_nfr_knn.to_numpy())

        # Same as before but with the improved NFR
        improved_nfr, impr_flips, impr_num_flips = improved_negative_flip_rate(model_v1, model_v2, test_loader)
        print("Improved negative flip rate: "+ str(improved_nfr))
        improved_relative_nfr = relative_negative_flip_rate(improved_nfr, tta_val_acc_v1, tta_val_acc_v2)
        print("Improved relative negative flip rate: "+ str(improved_relative_nfr))
        impr_flips_fig, impr_flips_df, impr_flips_fig_per, impr_flips_df_per = flips_heatmap(impr_flips, impr_num_flips, classes,  display=display_all)
        impr_fig_nfr_knn, impr_df_nfr_knn = link_NFR_adjacency(impr_flips_df, df_knn_v2,  display=display_all)
        if hyp['num_models'] == 1:
            print_and_log(impr_flips_df,impr_flips_fig, "Heatmap of the improved negative flips between classes" )
            print_and_log(impr_flips_df_per, impr_flips_fig_per, "Heatmap of the improved negative flips' percentage between classes")
            print_and_log(impr_df_nfr_knn, impr_fig_nfr_knn , 'Heatmap of the improved negative flips - knn matrix')
        else: 
            impr_flip_matrices.append(impr_flips_df.to_numpy())
            impr_flip_per_mat.append(impr_flips_df_per.to_numpy())
            impr_nfr_knn_matrices.append(impr_df_nfr_knn.to_numpy())
        
        eval_metrics = {'tta_val_acc_v1': tta_val_acc_v1, 'tta_val_acc_v2': tta_val_acc_v2,
                        'NFR': nfr, 'improved_NFR': improved_nfr,
                        'Relative NFR':relative_nfr , 'Improved relative NFR':improved_relative_nfr}
        wandb.log({**eval_metrics})
    if hyp['num_models']>1:
        #Calculate and show the sum of all the knn matrices
        df_sum_knn, df_sum_per_knn, fig_sum_knn, fig_per_knn= knn_matrices_sum(np.array(knn_matrices), classes)
        print_and_log(df_sum_knn,fig_sum_knn,'Heatmap of the sum of all knn matrices' )
        print_and_log(df_sum_per_knn, fig_per_knn,'Heatmap of the percentage of all knn matrices' )
        
        # Calculate and print/log the mean, min, max and std NFR heatmap and its connection to the adjacency matrices
        mean_flip_mat = np.mean(np.array(flip_matrices), axis = 0)
        df_mean_flip, fig_mean_flip= df_plot_heatmap(mean_flip_mat, classes, 'Mean negative flips', 'Purples', '.1f', "New prediction", "Old predictions")
        print_and_log(df_mean_flip, fig_mean_flip,"Mean heatmap of the negative flips between classes")
        mean_flip_per_mat =  np.mean(np.array(flip_per_mat), axis = 0)
        df_mean_flip_per, fig_mean_flip_per= df_plot_heatmap(mean_flip_per_mat, classes, 'Mean percentage of negative flips', 'Purples',
                                                              '.1f', "New prediction", "Old predictions")
        print_and_log(df_mean_flip_per, fig_mean_flip_per,"Mean heatmap of the negative flips' percentage between classes")
        neg_per, pos_per = summary_nfr_adj(nfr_knn_matrices, classes)
        
        # Same code but with the improved NFR
        mean_impr_flip_mat = np.mean(np.array(impr_flip_matrices), axis = 0)
        df_impr_mean_flip, fig_impr_mean_flip= df_plot_heatmap(mean_impr_flip_mat, classes, 'Mean improved negative flips', 'Purples', '.1f', "New prediction", "Old predictions")
        print_and_log(df_impr_mean_flip, fig_impr_mean_flip,"Mean heatmap of the improved negative flips between classes")
        mean_impr_per_flip_mat = np.mean(np.array(impr_flip_per_mat), axis = 0)
        df_mean_impr_flip_per, fig_mean_impr_flip_per= df_plot_heatmap(mean_impr_per_flip_mat, classes, 'Mean percentage of improved negative flips', 'Purples', '.1f',
                                                                       "New prediction", "Old predictions")
        print_and_log(df_mean_impr_flip_per, fig_mean_impr_flip_per,"Mean heatmap of the improved negative flips' percentage between classes")
        impr_neg_per, impr_pos_per= summary_nfr_adj(impr_nfr_knn_matrices, classes, impr = "Improved")
        
        per_metrics = {'Far classes negative flips percentage': neg_per,
                        'Near negative flips percentage' : pos_per,
                        'Far classes improved negative flips percentage': impr_neg_per,
                        'Near improved negative flips percentage' : impr_pos_per}
        wandb.log({**per_metrics})
        
    wandb_run.finish()


if __name__ == '__main__':
    main()