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
        'old_backbone': 'resnet18',     # backbone of the old model
        'old_feat_dim': 512,            # feat dim of the old model
    },
    'data': {
        'num_classes': 100,
        'subset_list': None,        #es: list [0,50,1] corresponds to list(range(0,50,1))
        'old_subset_list': None,    # subset list of the old model, to create the correct dataloader to calculate NFR
    },
    'dSimplex': False,              # if the classifier is a dSimplex
    'num_models': 5,                
    'old_model_name' : None,        # the name of the worse older model that is going to be used in NFR calculation
    'k' : [1],                      # List of k that is going to be used
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
    hyp['net']['old_backbone'] = loaded_params['old_backbone']
    hyp['net']['old_feat_dim'] = loaded_params['old_feat_dim']
    
    hyp['k'] = loaded_params['k']

    hyp['dSimplex'] = loaded_params['dSimplex']
"""
Code that implements mutual knn-alignment (Appendix A of https://arxiv.org/abs/2405.07987) between the prototypes, instead of features, of an old model and multiple new ones
and calculate the cosine distance between them.
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
    
    # Get test images
    _, cifar100_test_loader = create_dataloaders('cifar100', DATASET_PATH, hyp['opt']['batch_size'], 
                                                                subset_list= hyp['data']['subset_list'])
    classes = CIFAR100_CLASSES

    # Collects all the new models' prototypes
    W_array = []
    for name_index in range(len(model_names)):
        for i in range(hyp['num_models']):
            current_model_name = model_names[name_index].split(":v")[0]+":v"+str(i+int(model_names[name_index].split(":v")[1]))
            print(current_model_name)
            # Get model
            model = create_model( hyp['net']['backbone'],False, hyp['net']['feat_dim'],hyp['data']['num_classes'], device,
                         WANDB_PROJECT+current_model_name, wandb_run )

            # Evaluate model
            test_model(model, cifar100_test_loader, wandb_run, step = i + name_index * hyp['num_models'])

            # Get model last layer's weights and plot its vectors
            W = model.fc2.weight.detach().cpu().numpy().astype(np.float32)
            W_array.append(W)
    
    # Get old model's prototypes
    old_model = create_model(hyp['net']['old_backbone'], False, hyp['net']['old_feat_dim'], len( hyp['data']['old_subset_list']), 
                                     device, WANDB_PROJECT+hyp['old_model_name'], wandb_run )
    W_old = old_model.fc2.weight.detach().cpu().numpy().astype(np.float32)

    # Calculate mutual knn alignment between the old and new model's prototypes for each k value
    W_array = np.array(W_array)
    for k in hyp['k']:
        knn_alignments = []
        mean_alignments = []
        # Calculate and store knn_alignement bewteen the old model and each group of new models
        for name_index in range(len(model_names)):
            old_knn_alignment = mutual_knn_alignment_one_model_prototype(W_old, 
                                W_array[name_index*hyp['num_models']: name_index*hyp['num_models']+hyp['num_models'],:], k,
                                hyp['data']['old_subset_list'])
            # Store mean alignment between a group of new models and the old one
            mean_alignments.append(np.mean(old_knn_alignment))
            knn_alignments.append(old_knn_alignment)
        
        # Plot the alignment between every group of new models and the old one
        fig_old = plot_alignment_old_other_models(knn_alignments, k, 'Old mutual knn alignment, k:' + str(k) ,labels, colors)
        wandb.log({'Old mutual knn alignment, k:' + str(k): wandb.Image(fig_old)})
        # Plot the mean alignment of each group of new models
        fig_mean= bar_plot(mean_alignments, labels, colors, " Mean old mutual knn alignment, k: "+ str(k),
                            "New models", "Mean mutual knn alignment", (0, 1))
        wandb.log({'Mean old mutual knn alignment, k:' + str(k): wandb.Image(fig_mean)})

    # Calculate and plot the cosine distances between the old model's prototypes
    old_distances = vectors_distances(W_old)
    df_old_distances , fig_old_distances = df_plot_heatmap(old_distances, classes[hyp['data']['old_subset_list'][0]: hyp['data']['old_subset_list'][-1]+1],
                                                            "Distances between old model class prototypes", 
                                                           'Purples', '.1f', "", "", center =1, vmin= 0, vmax = 2, figsize=( 20, 14), annot = False)
    print_and_log(df_old_distances, fig_old_distances,"Distances between old model class prototypes" )

    # Collect, for each model, the distance between the prototypes and the origin point
    mean_origin_distances = []
    mean_origin_distances.append( origin_distances(W_old))
    # Collect the mean distance between the distances of the old model's prototypes and the new models' prototypes, considering all classes
    # or only the ones that correspond to old model's classes, if they are a subset of the new model's classes
    mean_compare_distances, std_compare_distances = [], []
    for name_index in range(len(model_names)):                      # iterate through groups of new models
        distances = []
        origin_distance =[]
        for i in range(hyp['num_models']):                          # iterate through each model of a group
            distances.append(vectors_distances(W_array[name_index*hyp['num_models']: name_index*hyp['num_models']+hyp['num_models'],:][i]))
            origin_distance.append(origin_distances(W_array[name_index*hyp['num_models']: name_index*hyp['num_models']+hyp['num_models'],:][i]))

        mean_origin_distances.append(np.mean(np.array(origin_distance),0))

        mean_distance = np.mean(np.array(distances),0)
        df_mean_distance, fig_mean_distance =  df_plot_heatmap(mean_distance, classes, 
                                                               "Mean distances between class prototypes in " +  model_names[name_index].split(":v")[0],
                                                               'Purples', '.1f',  "", "", center=1, vmin= 0, vmax = 2, figsize=( 20, 14), annot = False)
        print_and_log(df_mean_distance, fig_mean_distance,"Mean distances between class prototypes in " +  model_names[name_index].split(":v")[0] )

        # Compare old-new models's prototype distances
        compare_old_new_distances = np.absolute(old_distances-mean_distance[hyp['data']['old_subset_list'][0]: hyp['data']['old_subset_list'][-1]+1,
                                                                             hyp['data']['old_subset_list'][0]: hyp['data']['old_subset_list'][-1]+1])
        df_compare, fig_compare =  df_plot_heatmap(compare_old_new_distances, classes[hyp['data']['old_subset_list'][0]: hyp['data']['old_subset_list'][-1]+1], 
                                                               "Absolute distance between old model and " +  model_names[name_index].split(":v")[0]+ " prototype distances",
                                                               'Reds', '.1f',  "", "", center=1, vmin= 0, vmax = 2, figsize=( 20, 14), annot = False)
        print_and_log(df_compare, fig_compare,"Absolute distance between old model and " +  model_names[name_index].split(":v")[0]+ " prototype distances" )
        # Save the mean and std of the compared distances for each group of models
        mean_compare_distances.append( np.mean(compare_old_new_distances))
        std_compare_distances.append(np.std(compare_old_new_distances))
    # Create and log the bar plot for the mean compared distances        
    fig_mean_comp_distances = bar_plot(mean_compare_distances, labels, colors, "Mean of the distances between old and new models prototypes", "New models", 
                                       "Mean distance", (0,2), yerr= std_compare_distances)
    wandb.log({'Mean of the distances between old and new models prototypes': wandb.Image(fig_mean_comp_distances)})

    
    # Heatmap containg the mean prototype-origin point of each group of models
    df_origin, fig_origin = df_plot_heatmap(mean_origin_distances, classes, "Origin distances", "Blues", '.1f', "Classes", "Models", 
                                            index = ['Old model']+ labels, columns = [i for i in classes], figsize=( 20, 14), annot = False)
    print_and_log(df_origin, fig_origin, "Origin distances")

    wandb_run.finish()


if __name__ == '__main__':
    main()