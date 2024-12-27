import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
import torch
from airbench94 import CifarLoader, make_net, evaluate
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from features_alignment import *


hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
        'feat_dim' : 3,         # features' dimension
    },
    'num_models': 100,
}
"""
Code that implements mutual knn-alignment (Appendix A of https://arxiv.org/abs/2405.07987) using the prototype of multiple models trained under the
same hyperpameters (different seeds)
"""

def main():

    # Select config 
    parser = argparse.ArgumentParser(description='Feature alignment')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    model_name = loaded_params['model_name']
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['num_models'] = loaded_params['num_models']

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "Comparing models prototypes "+ model_name[1:-4],
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    adj_matrices = []
    W_array = []

    for i in range(hyp['num_models']):
        current_model_name =  model_name.split(":v")[0]+":v"+str(i+int(model_name.split(":v")[1]))
        print(current_model_name)
        # Get model
        model = make_net(hyp['net']['feat_dim'])
        artifact = wandb_run.use_artifact(WANDB_PROJECT+current_model_name, type='model')
        artifact_dir = artifact.download()
        model.load_state_dict(torch.load(artifact_dir+'/model.pth'))

        # Evaluate model
        tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
        print("tta_val_acc for "+ current_model_name +": "+ str(tta_val_acc))

        eval_metrics = {'tta_val_acc': tta_val_acc}
        wandb.log({**eval_metrics})

        # Get model last layer's weights and plot its vectors
        W = model[8].weight.detach().cpu().numpy().astype(np.float32)
        W_array.append(W)

        # Calculate the adjacency matrix of the prototypes
        adjacency_matrix=calculate_adjacency_matrix(W)
        adj_matrices.append(adjacency_matrix)
        
    
    # Calculate mutual knn alignment between prototypes
    W_array = np.array(W_array)
    for k in [2,3,4,5]:
        knn_alignment= mutual_knn_alignment_prototypes(W_array, k)
        df_knn_alignment = pd.DataFrame(knn_alignment)
        print("k:" + str(k))
        print(f"KNN alignment, k: {k} \n {df_knn_alignment}")
        wandb.log({"KNN alignment, k: "+ str(k) + " " +  model_name[1:-4]: wandb.Table(dataframe=df_knn_alignment)})
        # Print the alignment between every model and the most/least/"medium" aligned models
        mean_alignment_vec= calculate_mean_alignment_vector(knn_alignment)
        indices = find_models_indices(mean_alignment_vec)
        fig = plot_alignment(knn_alignment, indices, k)
        wandb.log({'Mutual knn alignment, k:' + str(k): wandb.Image(fig)})
        
        #Calculate and show the sum of all the adjacency matrices
        df_summed_matrix, df_percent_matrix, fig_sum, fig_per= adjacency_matrices_sum(np.array(adj_matrices), classes)

        print(f"Sum of all adjacency matrices:\n {df_summed_matrix}")
        wandb.log({"Sum of all adjacency matrices "+  model_name[1:-4]: wandb.Table(dataframe=df_summed_matrix)})
        wandb.log({'Heatmap of the sum of all adjacency matrices': wandb.Image(fig_sum)})

        print(f"Percentage of all adjacency matrices:\n {df_percent_matrix}")
        wandb.log({"Percentage of all adjacency matrices "+  model_name[1:-4]: wandb.Table(dataframe=df_percent_matrix)})
        wandb.log({'Heatmap of the percentage of all adjacency matrices': wandb.Image(fig_per)})

    wandb_run.finish()
    


if __name__ == '__main__':
    main()