import os
import wandb
import argparse, yaml
from dotenv import load_dotenv

import torch

from airbench94 import CifarLoader, make_net, evaluate

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip_rate import negative_flip_rate, improved_negative_flip_rate

hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}


def main():

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "Evaluate CIFAR10",
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)

    # Get model with worse performances
    model_v1 = make_net()
    artifact = wandb_run.use_artifact(WANDB_PROJECT+'/CIFAR10_50percent_run0:v10', type='model')
    artifact_dir = artifact.download()
    model_v1.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    # Get model with better performances
    model_v2 = make_net()
    artifact = wandb_run.use_artifact(WANDB_PROJECT+'/CIFAR10_100percent_run3:v0', type='model')
    artifact_dir = artifact.download()
    model_v2.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    # Evaluate both models
    tta_val_acc_v1 = evaluate(model_v1, test_loader, tta_level=hyp['net']['tta_level'])
    print("tta_val_acc model v1: "+ str(tta_val_acc_v1))
    tta_val_acc_v2 = evaluate(model_v2, test_loader, tta_level=hyp['net']['tta_level'])
    print("tta_val_acc model v1: "+ str(tta_val_acc_v2))

    # Calculate the NFR
    nfr = negative_flip_rate(model_v1, model_v2, test_loader)
    print("Negative flip rate: "+ str(nfr))

    improved_nfr = improved_negative_flip_rate(model_v1, model_v2, test_loader)
    print("Improved negative flip rate: "+ str(improved_nfr))

    eval_metrics = {'tta_val_acc_v1': tta_val_acc_v1, 'tta_val_acc_v2': tta_val_acc_v2,
                    'NFR': nfr, 'improved_NFR': improved_nfr}
    wandb.log({**eval_metrics})

    wandb_run.finish()


if __name__ == '__main__':
    main()