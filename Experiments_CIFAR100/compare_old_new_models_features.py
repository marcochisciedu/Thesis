import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
import torch

from dataset import *
from models import *
from testing import test_model
from utils import CIFAR100_CLASSES

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from features_alignment import *
from print_and_logging import *


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
    'k' : 20,                      # number of nearest neighbors
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

    hyp['k'] = loaded_params['k']

    hyp['dSimplex'] = loaded_params['dSimplex']
    
"""
Code that implements mutual knn-alignment (Appendix A of https://arxiv.org/abs/2405.07987) between the features of an old model and multiple new ones
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
    run_name = loaded_params['run_name']
    model_names = loaded_params['model_names']
    labels = loaded_params['labels']
    colors = loaded_params['colors']
    define_hyp(loaded_params)

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')
    DATASET_PATH = os.getenv('DATASET_PATH')

    wandb.login(key= WANDB_API_KEY, verify=True)

    device = torch.device("cuda:0")

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = run_name,
        config=hyp)
    
     # Get test images, only the ones with classes that the old model was trained on
    _, cifar100_test_loader = create_dataloaders('cifar100', DATASET_PATH,hyp['opt']['batch_size'],subset_list= hyp['data']['old_subset_list'])

    # Get old model
    old_model = create_model(hyp['net']['backbone'], False, hyp['net']['feat_dim'], len( hyp['data']['old_subset_list']), 
                                    device, WANDB_PROJECT+hyp['old_model_name'], wandb_run )

    mean_mutual_knn_alignments, all_mutual_knn_alignments = [], []
    for name_index in range(len(model_names)):      # iterate through each type of new model
        models_mutual_knn_alignments = []
        for i in range(hyp['num_models']):          # iterate through each new model
            current_model_name = model_names[name_index].split(":v")[0]+":v"+str(i+int(model_names[name_index].split(":v")[1]))
            print(current_model_name)
            # Get model
            model = create_model( hyp['net']['backbone'],False, hyp['net']['feat_dim'],hyp['data']['num_classes'], device,
                         WANDB_PROJECT+current_model_name, wandb_run )

            with torch.no_grad():
                all_intersections = 0
                num_images = 0
                for data in cifar100_test_loader:        # iterate through the test loader
                    x,_ = data
                    x = x.cuda()

                    # Extract both model's features
                    features = model(x)['features']
                    old_features = old_model(x)['features']

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

    wandb_run.finish()
if __name__ == '__main__':
    main()