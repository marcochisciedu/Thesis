import os
import wandb
import argparse, yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import torch

from airbench94 import CifarLoader, make_net, evaluate

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip_rate import *

hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

def flips_heatmap(flips, num_flips, classes):
    flip_matrix = np.zeros((len(classes), len(classes)), dtype = int)
    percentage_flip_matrix = np.zeros((len(classes), len(classes)), dtype = float)

    # create flip matrix
    for key in flips:
        flip_matrix[key[0],key[1]] = flips[key]
        percentage_flip_matrix[key[0],key[1]] = flips[key]*100/num_flips
    
    df_flips = pd.DataFrame(flip_matrix, index = [i for i in classes], columns = [i for i in classes])
    fig=plt.figure(figsize = (12,7))
    sns.heatmap(df_flips, cmap="Purples", annot=True, fmt='d', linewidth=.5, square= True, robust= True)
    plt.title('Negative flips')
    plt.xlabel("New prediction")
    plt.ylabel("Old prediction")
    plt.show()

    df_flips_perc = pd.DataFrame(percentage_flip_matrix, index = [i for i in classes], columns = [i for i in classes])
    fig_perc=plt.figure(figsize = (12,7))
    sns.heatmap(df_flips_perc, cmap="Purples", annot=True, fmt=".1f", linewidth=.5, square= True, robust= True)
    plt.title('Percentage of negative flips')
    plt.xlabel("New prediction")
    plt.ylabel("Old prediction")
    plt.show()
    
    return fig, df_flips, fig_perc, df_flips_perc

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
        name = "NFR Half Classes CIFAR10",
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Get model with worse performances
    model_v1 = make_net()
    artifact = wandb_run.use_artifact(WANDB_PROJECT+old_model_name, type='model')
    artifact_dir = artifact.download()
    model_v1.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    # Get model with better performances
    model_v2 = make_net()
    artifact = wandb_run.use_artifact(WANDB_PROJECT+new_model_name, type='model')
    artifact_dir = artifact.download()
    model_v2.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    # Evaluate both models
    tta_val_acc_v1 = evaluate(model_v1, test_loader, tta_level=hyp['net']['tta_level'])
    print("tta_val_acc model v1: "+ str(tta_val_acc_v1))
    tta_val_acc_v2 = evaluate(model_v2, test_loader, tta_level=hyp['net']['tta_level'])
    print("tta_val_acc model v1: "+ str(tta_val_acc_v2))

    # Calculate the NFR
    nfr, flips, num_flips = negative_flip_rate(model_v1, model_v2, test_loader)
    print("Negative flip rate: "+ str(nfr))
    flips_fig, flips_df, flips_fig_per, flips_df_per = flips_heatmap(flips, num_flips, classes)
    print(f"Heatmap of the negative flips between classes:\n {flips_df}")
    wandb.log({'Heatmap of the negative flips between classes': wandb.Image(flips_fig)})
    print(f"Heatmap of the negative flips' percentage between classes:\n {flips_df_per}")
    wandb.log({"Heatmap of the negative flips' percentge between classes": wandb.Image(flips_fig_per)})
    relative_nfr = relative_negative_flip_rate(nfr, tta_val_acc_v1, tta_val_acc_v2)
    print("Relative negative flip rate: "+ str(relative_nfr))

    improved_nfr, impr_flips, impr_num_flips = improved_negative_flip_rate(model_v1, model_v2, test_loader)
    print("Improved negative flip rate: "+ str(improved_nfr))
    impr_flips_fig, impr_flips_df, impr_flips_fig_per, impr_flips_df_per = flips_heatmap(impr_flips, impr_num_flips, classes)
    print(f"Heatmap of the improved negative flips between classes:\n {impr_flips_df}")
    wandb.log({'Heatmap of the improved negative flips between classes': wandb.Image(impr_flips_fig)})
    print(f"Heatmap of the improved negative flips' percentage between classes:\n {impr_flips_df_per}")
    wandb.log({"Heatmap of the improved negative flips' percentge between classes": wandb.Image(impr_flips_fig_per)})
    improved_relative_nfr = relative_negative_flip_rate(improved_nfr, tta_val_acc_v1, tta_val_acc_v2)
    print("Improved relative negative flip rate: "+ str(improved_relative_nfr))

    eval_metrics = {'tta_val_acc_v1': tta_val_acc_v1, 'tta_val_acc_v2': tta_val_acc_v2,
                    'NFR': nfr, 'improved_NFR': improved_nfr,
                    'Relative NFR':relative_nfr , 'Improved relative NFR':improved_relative_nfr}
    wandb.log({**eval_metrics})

    wandb_run.finish()


if __name__ == '__main__':
    main()