import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
import torch
import plotly
from airbench94 import CifarLoader, make_net, evaluate

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from features_alignment import *
from plot import *


hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
    'nfr' : False,                  # if True the NFR points are added to the 3D plot
    'add_convex_hull': True,        # if a convex hull is added to the 3D plot, only in the regular 3D plot
    'old_model_name' : None,        # the name of the worse older model that is going to be used in NFR calculation
}

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
    hyp['nfr'] = loaded_params['nfr']
    if hyp['nfr'] == True:
        hyp['old_model_name'] = loaded_params['old_model']
        hyp['class_index'] = loaded_params['class_index']
        wandb_name= "3D plot with NFR "+ model_name[1:-4]
    else:
        wandb_name = "3D plot "+ model_name[1:-4]
        hyp['add_convex_hull'] = loaded_params['add_convex_hull']

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)
    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = wandb_name,
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print(model_name)
    # Get model
    model = make_net(feat_dim= 3)
    artifact = wandb_run.use_artifact(WANDB_PROJECT+model_name, type='model')
    artifact_dir = artifact.download()
    model.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    # Evaluate model
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    print("tta_val_acc for "+ model_name +": "+ str(tta_val_acc))

    eval_metrics = {'tta_val_acc': tta_val_acc}
    wandb.log({**eval_metrics})

    # Get model last layer's weights and plot its vectors
    W = model[8].weight.detach().cpu().numpy().astype(np.float32)
    # 3d plot with negative flips features or prototypes' convex hull
    if hyp['nfr'] == True:
        old_model = make_net(feat_dim= 3)
        artifact = wandb_run.use_artifact(WANDB_PROJECT+hyp['old_model_name'], type='model')
        artifact_dir = artifact.download()
        old_model.load_state_dict(torch.load(artifact_dir+'/model.pth'))
        correct_features, adj_nf_features, non_adj_nf_features= class_negative_flip_rate_features(hyp['class_index'],old_model, model, test_loader)
        impr_correct_feat, impr_adj_nf_feat, impr_non_adj_feat = class_negative_flip_rate_features(hyp['class_index'],old_model, model, test_loader, impr = True)
        
        vect_feat_fig= vector_features_plot(W,correct_features, adj_nf_features, non_adj_nf_features, classes)
        vect_feat_fig.write_image("CIFAR-10 3D plot with neg flip features.pdf")
        impr_vect_feat_fig = vector_features_plot(W,  impr_correct_feat, impr_adj_nf_feat, impr_non_adj_feat, classes )
        impr_vect_feat_fig.write_image("CIFAR-10 3D plot with improved neg flip features.pdf")
        wandb.log({"3d plot with "+ classes[hyp['class_index']]+ " features and nfr" : wandb.Plotly(vect_feat_fig)})
        wandb.log({"3d plot with "+ classes[hyp['class_index']]+ " features and improved nfr" : wandb.Plotly(impr_vect_feat_fig)})
    else:
        fig = vector_plot(W, classes, add_convex_hull=hyp['add_convex_hull'])
        if hyp['add_convex_hull']:
            fig.write_image("CIFAR-10 3D plot with convex hull.pdf")
        else:
            fig.write_image("CIFAR-10 3D plot.pdf")
        wandb.log({"3d plot classes' prototypes and convex hull" : wandb.Plotly(fig)})


    wandb_run.finish()
    


if __name__ == '__main__':
    main()