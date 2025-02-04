import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
import torch
from airbench94 import CifarLoader, make_net, evaluate

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from features_alignment import *
from print_and_logging import *


hyp = {
    'net': {
        'feat_dim' : 3,                 # features' dimension
        'num_classes': 10,              # num classes of the new model
        'old_num_classes': 5,           # num classes of the old model
    },
    'batch_size': 2000,                 # Batch size, how many features are compared together
    'num_models': 100,
    'old_model_name' : None,            # the name of the worse older model that is going to be compare to the new models
    'low_class_list': None,             # indices of the classes that the old model was not trained on
    'nfr' : False,                      # if NFR will be calculated
}
"""
Code used to test models trained on CIFAR-10.
The new models can be compared to an older version.
Test time accuracy and Negative Flip Rate
"""

def main():

    # Select config 
    parser = argparse.ArgumentParser(description='Testing CIFAR-10')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    model_name = loaded_params['model_name']
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['batch_size'] = loaded_params['batch_size']
    hyp['num_models'] = loaded_params['num_models']
    hyp['net']['num_classes'] = loaded_params['num_classes']
    hyp['old_model_name'] = loaded_params['old_model']
    hyp['net']['old_num_classes'] = loaded_params['old_num_classes']
    hyp['low_class_list'] = loaded_params['low_class_list']
    hyp['nfr'] = loaded_params['nfr']
    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    print("Testing "+ model_name.split(":v")[0])
    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "Testing "+ model_name.split(":v")[0],
        config=hyp)
    
     # Get old model if needed to calculate NFR 
    if hyp['nfr']:
        old_model = make_net( hyp['net']['feat_dim'], hyp['net']['old_num_classes'])
        artifact = wandb_run.use_artifact(WANDB_PROJECT+hyp['old_model_name'], type='model')
        artifact_dir = artifact.download()
        old_model.load_state_dict(torch.load(artifact_dir+'/model.pth'))
        # Dataset used to train the old model
        old_test_loader = CifarLoader('cifar10', train=False, batch_size=hyp['batch_size'],
                                   list_low_classes= hyp['low_class_list'], low_percentage= [0]*len(hyp['low_class_list']))
    else:
        old_model = None

    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=hyp['batch_size'])
    
     # If more than one model collect all the accuracies and NFR to calculate the mean value
    if hyp['num_models'] > 1:
        new_model_full_top1 = []
        if hyp['nfr']: 
            new_model_old_top1, old_model_top1  = [], []
            nfrs, impr_nfrs, rel_nfrs, rel_impr_nfrs = [], [], [], []

    for i in range(hyp['num_models']):
        if hyp['num_models'] > 1:
            current_model_name = model_name.split(":v")[0]+":v"+str(i+int(model_name.split(":v")[1]))
        else:
            current_model_name = model_name

        # Load model
        model = make_net( hyp['net']['feat_dim'], hyp['net']['num_classes'])
        artifact = wandb_run.use_artifact(WANDB_PROJECT+current_model_name, type='model')
        artifact_dir = artifact.download()
        model.load_state_dict(torch.load(artifact_dir+'/model.pth'))

        # Test the loaded model
        new_top1 =evaluate(model, test_loader)

        # Calculate NFR       
        if hyp['nfr']:  
            nfr, _, _ = negative_flip_rate(old_model, model, old_test_loader)
            impr_nfr, _ , _ = improved_negative_flip_rate(old_model, model, old_test_loader)

            # Test both models using only the classes used to train the old model and calculate relative NFR
            new_old_top1 = evaluate(model, old_test_loader)
            old_top1 = evaluate(old_model, old_test_loader)
            rel_nfr = relative_negative_flip_rate(nfr, old_top1, new_old_top1)
            rel_impr_nfr = relative_negative_flip_rate(impr_nfr, old_top1, new_old_top1)


            print(f"Negative flip rate : {nfr} and Relative NFR: {rel_nfr}")
            print(f"Improved negative flip rate: {impr_nfr} and Relative Improved NFR: {rel_impr_nfr}")
            wandb.log({'New model, old test accuracy' : new_old_top1, 'Old model test accuracy': old_top1,
                        'NFR':nfr, 'Improved NFR': impr_nfr, 'Relative_NFR': rel_nfr,
                       'Relative Improved NFR': rel_impr_nfr}, step= i)

        if hyp['num_models'] > 1:
            new_model_full_top1.append(new_top1)
            if hyp['nfr']: 
                new_model_old_top1.append(new_old_top1)
                old_model_top1.append(old_top1)
                nfrs.append(nfr)
                impr_nfrs.append(impr_nfr)
                rel_nfrs.append(rel_nfr)
                rel_impr_nfrs.append(rel_impr_nfr)

    if hyp['num_models'] > 1:
        wandb.log({'Mean new model top1 accuracy':np.mean(new_model_full_top1)})
        if hyp['nfr']: 
            wandb.log({'Mean new model old test set top1 accuracy':np.mean(new_model_old_top1),
                       'Mean old model top1 accuracy':np.mean(old_model_top1)})
            wandb.log({'Mean NFR':np.mean(nfrs), 'Mean Improved NFR': np.mean(impr_nfrs),
                       'Mean Relative NFR': np.mean(rel_nfrs), 'Mean Relative Improved NFR': np.mean(rel_impr_nfrs)})

    wandb.finish()


if __name__ == '__main__':
    main()