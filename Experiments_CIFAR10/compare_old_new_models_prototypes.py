import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import torch
from airbench94 import CifarLoader, make_net, evaluate
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from features_alignment import *


hyp = {
    'net': {
        'tta_level': 0,                 # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
    'num_models': 100,
    'old_model_name' : None,            # the name of the worse older model that is going to be compare to the new models
    'old_model_classes_indices': None,  # indices of the classes shared by all the models
    'k' : [1],                          # List of k that is going to be used
}

def main():

    # Select config 
    parser = argparse.ArgumentParser(description='Feature alignment')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    model_names = loaded_params['model_names']
    labels = loaded_params['labels']
    colors = loaded_params['colors']
    hyp['num_models'] = loaded_params['num_models']
    hyp['old_model_name'] = loaded_params['old_model']
    hyp['old_model_classes_indices'] = loaded_params['old_model_classes_indices']
    hyp['k'] = loaded_params['k']

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "Comparing old model to "+ str(len(model_names)) + " new models' classes prototypes",
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)

    W_array = []
    for name_index in range(len(model_names)):
        for i in range(hyp['num_models']):
            current_model_name = model_names[name_index].split(":v")[0]+":v"+str(i+int(model_names[name_index].split(":v")[1]))
            print(current_model_name)
            # Get model
            model = make_net()
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
    
    # Calculate mutual knn alignment between the old and new model's prototypes 
    W_array = np.array(W_array)
    # Get old model's prototypes
    old_model = make_net()
    artifact = wandb_run.use_artifact(WANDB_PROJECT+hyp['old_model_name'], type='model')
    artifact_dir = artifact.download()
    old_model.load_state_dict(torch.load(artifact_dir+'/model.pth'))
    W_old = old_model[8].weight.detach().cpu().numpy().astype(np.float32)
    for k in hyp['k']:
        knn_alignments = []
        mean_alignments = []
        # Calculate and store knn_alignement bewteen the old model and each group of new models
        for name_index in range(len(model_names)):
            old_knn_alignment = mutual_knn_alignment_one_model_prototype(W_old, 
                                W_array[name_index*hyp['num_models']: name_index*hyp['num_models']+hyp['num_models'],:], k,
                                hyp['old_model_classes_indices'])
            mean_alignments.append(np.mean(old_knn_alignment))
            knn_alignments.append(old_knn_alignment)

        # Plot the alignment between every group of new models and the old one
        fig_old = plot_alignment_old_other_models(knn_alignments, k, labels, colors)
        wandb.log({'Old mutual knn alignment, k:' + str(k): wandb.Image(fig_old)})
        #Plot the mean alignment of each group of new models
        fig_mean= plot_mean_alignments(mean_alignments, k, labels, colors)
        wandb.log({'Mean old mutual knn alignment, k:' + str(k): wandb.Image(fig_mean)})

    old_distances = vectors_distances(W_old)
    df_old_distances = pd.DataFrame(old_distances)   
    print(f'old distances: \n {df_old_distances}')
    mean_distances = []
    for name_index in range(len(model_names)):
        distances = []
        for i in range(hyp['num_models']):
            distances.append(vectors_distances(W_array[name_index*hyp['num_models']: name_index*hyp['num_models']+hyp['num_models'],:][i]))
        mean_distance = np.mean(np.array(distances),0)
        mean_distances.append(mean_distance)
        df_mean_distance = pd.DataFrame(mean_distance)   
        print(f'distance {name_index+1}° group of models: \n {df_mean_distance}')
    wandb_run.finish()
    


if __name__ == '__main__':
    main()