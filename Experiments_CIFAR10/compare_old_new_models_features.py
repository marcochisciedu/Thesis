import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
import torch
from airbench94 import CifarLoader, make_net

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from features_alignment import *
from print_and_logging import *


hyp = {
    'net': {
        'feat_dim' : 3,                 # features' dimension
    },
    'batch_size': 2000,                 # Batch size, how many features are compared together
    'num_models': 100,
    'old_model_name' : None,            # the name of the worse older model that is going to be compare to the new models
    'old_model_classes_indices': None,  # indices of the classes shared by all the models
    'k' : 20                            # number of nearest neighbors
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
    run_name = loaded_params['run_name']
    model_names = loaded_params['model_names']
    labels = loaded_params['labels']
    colors = loaded_params['colors']
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['batch_size'] = loaded_params['batch_size']
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
        name = run_name,
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=hyp['batch_size'])

    # Get old model
    old_model = make_net( hyp['net']['feat_dim'])
    artifact = wandb_run.use_artifact(WANDB_PROJECT+hyp['old_model_name'], type='model')
    artifact_dir = artifact.download()
    old_model.load_state_dict(torch.load(artifact_dir+'/model.pth'))
    # Remove the last old model's layers to extrate the images' features
    old_model[-1] = torch.nn.Identity()
    old_model[-2] = torch.nn.Identity()

    mean_mutual_knn_alignments, all_mutual_knn_alignments = [], []
    # convert old classes indices to tensor for later
    tensor_old_indices = torch.tensor(np.array(hyp['old_model_classes_indices'])).cuda()
    for name_index in range(len(model_names)):      # iterate through each type of new model
        models_mutual_knn_alignments = []
        for i in range(hyp['num_models']):          # iterate through each new model
            current_model_name = model_names[name_index].split(":v")[0]+":v"+str(i+int(model_names[name_index].split(":v")[1]))
            print(current_model_name)
            # Get model
            model = make_net( hyp['net']['feat_dim'])
            artifact = wandb_run.use_artifact(WANDB_PROJECT+current_model_name, type='model')
            artifact_dir = artifact.download()
            model.load_state_dict(torch.load(artifact_dir+'/model.pth'))
            # Remove the last model's layers to extrate the images' features
            model[-1] = torch.nn.Identity()
            model[-2] = torch.nn.Identity()

            with torch.no_grad():
                all_intersections = 0
                num_images = 0
                for data in test_loader:        # iterate through the test loader
                    x,y = data
                    # Select the images' indices that correspond to classes that the old model has
                    class_indices=torch.isin(y,tensor_old_indices ).nonzero(as_tuple=True)[0]

                    # Extract both model's features
                    features = model(x[class_indices])
                    old_features = old_model(x[class_indices])

                    # Count how many images' features are being used
                    num_images += features.size()[0]

                    # Update the intersection count
                    all_intersections += mutual_knn_alignment_features(features.cpu().numpy(), old_features.cpu().numpy(), hyp['k']) 
            # Store each model's knn feature alignment to the old model
            models_mutual_knn_alignments.append(all_intersections/num_images)
        # Store the mean and all the knn alignments to the old model of each new model of this group
        all_mutual_knn_alignments.append(np.array(models_mutual_knn_alignments))
        mean_mutual_knn_alignments.append(np.mean(models_mutual_knn_alignments))

    # Plot the alignment of each new model
    fig_feat = plot_alignment_old_other_models(all_mutual_knn_alignments, hyp['k'], "Features mutual knn alignment, k: "+ str(hyp['k']), labels, colors)      
    wandb.log({'Features mutual knn alignment, k:' + str(hyp['k']): wandb.Image(fig_feat)})
    # Plot the mean alignment of each group of new models
    fig_feat= bar_plot(mean_mutual_knn_alignments, labels, colors, " Mean features mutual knn alignment, k: "+ str(hyp['k']), 
                        "New models", "Mean mutual knn alignment", (0, 1))
    wandb.log({'Mean features mutual knn alignment, k: '+ str(hyp['k']): wandb.Image(fig_feat)})


if __name__ == '__main__':
    main()