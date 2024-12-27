import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
import torch

from models import *
from dataset import *
from testing import test_model

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from features_alignment import *
from plot import *
from utils import CIFAR100_CLASSES

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
    'add_convex_hull': True,        # if a convex hull is added to the 3D plot, only in the regular 3D plot
    'nfr' : False,                  # if, at the end of each epoch, the NFR will be calculated
    'class_index':  0,              # if nfr is True, which class is the focus of the NFR calculation
    'old_model_name' : None,        # the name of the worse older model that is going to be used in NFR calculation
}
# Transfer each loaded parameter to the correct hyp parameter
def define_hyp(loaded_params):
    hyp['net']['backbone'] = loaded_params['backbone']
    hyp['data']['num_classes'] = loaded_params['num_classes']
    subset_list = loaded_params['subset_list']
    if subset_list is not None:
        hyp['data']['subset_list']  = list(range(subset_list[0], subset_list[1], subset_list[2]))
    hyp['opt']['batch_size'] = loaded_params['batch_size']
    
    hyp['dSimplex'] = loaded_params['dSimplex']

    hyp['nfr'] = loaded_params['nfr']
    if hyp['nfr']: 
        hyp['old_model_name'] = loaded_params['old_model_name']
        old_subset_list = loaded_params['old_subset_list']
        hyp['data']['old_subset_list'] = list(range(old_subset_list[0], old_subset_list[1], old_subset_list[2]))
        hyp['class_index'] = loaded_params['class_index']
    else:
        hyp['add_convex_hull'] = loaded_params['add_convex_hull']

"""
Code that prints the given model's prototypes in 3D. Can print them showing their convex hull or with the features
of all the images of a chosen class highlighting the negative flips.
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
    define_hyp(loaded_params)
    if hyp['nfr']:
        wandb_name= "3D plot with NFR "+ model_name[1:-3]
    else:
        wandb_name = "3D plot "+ model_name[1:-3]

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
        name = wandb_name,
        config=hyp)
    
    # Get test images
    _, cifar100_test_loader = create_dataloaders('cifar100', DATASET_PATH, hyp['opt']['batch_size'], 
                                                                subset_list= hyp['data']['subset_list'])
    classes = CIFAR100_CLASSES

    print(model_name)
    # Load model
    model = create_model( hyp['net']['backbone'],False, hyp['net']['feat_dim'],hyp['data']['num_classes'], device,
                         WANDB_PROJECT+model_name, wandb_run )

    # Test the loaded model
    test_model(model, cifar100_test_loader , wandb_run)

    # Get model last layer's weights and plot its vectors
    W = model.fc2.weight.detach().cpu().numpy().astype(np.float32)
    # 3d plot
    if hyp['nfr'] == True:
        # Load the old model used to calculate NFR
        old_model = create_model(hyp['net']['backbone'], False, hyp['net']['feat_dim'], hyp['data']['num_classes'], 
                                    device, WANDB_PROJECT+hyp['old_model_name'], wandb_run )
        # Get the correct dataset to test the NFR
        _, cifar100_nfr_test_loader = create_dataloaders('cifar100', DATASET_PATH,hyp['opt']['batch_size'],subset_list= hyp['data']['old_subset_list'])
        # Calculate the features for each image in the selected class
        correct_features, adj_nf_features, non_adj_nf_features= class_negative_flip_rate_features_ResNet(hyp['class_index'],old_model, model, cifar100_nfr_test_loader)
        impr_correct_feat, impr_adj_nf_feat, impr_non_adj_feat = class_negative_flip_rate_features_ResNet(hyp['class_index'],old_model, model,
                                                                                                    cifar100_nfr_test_loader, impr = True)
        # Plot the new model's prototypes and images' features
        vect_feat_fig= vector_features_plot(W,correct_features, adj_nf_features, non_adj_nf_features, classes)
        impr_vect_feat_fig = vector_features_plot(W,  impr_correct_feat, impr_adj_nf_feat, impr_non_adj_feat, classes )
        wandb.log({"3d plot with "+ str(classes[hyp['class_index']])+ " features and nfr" : wandb.Plotly(vect_feat_fig)})
        wandb.log({"3d plot with "+ str(classes[hyp['class_index']])+ " features and improved nfr" : wandb.Plotly(impr_vect_feat_fig)})
    else:
        fig = vector_plot(W, classes, add_convex_hull=hyp['add_convex_hull'])
        wandb.log({"3d plot classes' prototypes and convex hull" : wandb.Plotly(fig)})

    wandb_run.finish()
    
if __name__ == '__main__':
    main()