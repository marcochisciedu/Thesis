import os
import sys
import wandb
import numpy as np
import argparse, yaml
from dotenv import load_dotenv

from dataset import *
from models import *
from utils import *

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from fixed_classifiers import *
torch.backends.cudnn.benchmark = True

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
    'num_models' : 1,
    'dSimplex': False,              # if the classifier is a dSimplex
    'nfr' : False,                  # if, at the end of each epoch, the NFR will be calculated
    'old_model_name' : None,        # the name of the worse older model that is going to be used in NFR calculation
}
# Transfer each loaded parameter to the correct hyp parameter
def define_hyp(loaded_params):
    hyp['net']['backbone'] = loaded_params['backbone']
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['data']['num_classes'] = loaded_params['num_classes']
    subset_list = loaded_params['subset_list']
    if subset_list is not None:
        hyp['data']['subset_list']  = list(range(subset_list[0], subset_list[1], subset_list[2]))

    hyp['opt']['batch_size'] = loaded_params['batch_size']
    hyp['num_models'] = loaded_params['num_models']
   
    hyp['nfr'] = loaded_params['nfr']
    if hyp['nfr']: 
        hyp['old_model_name'] = loaded_params['old_model_name']
        old_subset_list = loaded_params['old_subset_list']
        hyp['data']['old_subset_list'] = list(range(old_subset_list[0], old_subset_list[1], old_subset_list[2]))
    hyp['dSimplex'] = loaded_params['dSimplex']


# Function that tests the final model, returns top1 and top5 accuracy
@torch.no_grad()
def test_model(model, dataloader, wandb_run, step = 0, wandb_log_name = ""):

    model.eval()

    correct_1 = 0.0
    correct_5 = 0.0

    for (images, labels) in dataloader:

        images = images.cuda()
        labels = labels.cuda()

        logits = model(images)['logits']  # the model's output is a dictionary
        _, pred = logits.topk(5, 1, largest=True, sorted=True)

        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()

        #compute top1
        correct_1 += correct[:, :1].sum()

    print('Evaluating Network.....')
    print('Top1 accuracy: {:.4f} \tTop5 accuracy: {:.4f}'.format(correct_1.float() / len(dataloader.dataset), 
                                                           correct_5.float() / len(dataloader.dataset)))

    # Wandb log
    wandb_run.log({'Top 1 accuracy'+wandb_log_name: correct_1.float() / len(dataloader.dataset),
                   'Top 5 accuracy'+wandb_log_name: correct_5.float() / len(dataloader.dataset)}, step= step)

    return correct_1.float() / len(dataloader.dataset), correct_5.float() / len(dataloader.dataset)


def main():
     # Select config 
    parser = argparse.ArgumentParser(description='Training ResNet on CIFAR100')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    model_name = loaded_params['model_name']
    define_hyp(loaded_params)

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')
    DATASET_PATH = os.getenv('DATASET_PATH')

    wandb.login(key= WANDB_API_KEY, verify=True)

    device = torch.device("cuda:0")

    print("Testing "+ model_name.split(":v")[0])
    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "Testing "+ model_name.split(":v")[0],
        config=hyp)
    
    # Get old model if needed to calculate NFR 
    if hyp['nfr']:
        old_model = create_model(hyp['net']['backbone'], False, hyp['net']['feat_dim'], len( hyp['data']['old_subset_list']), 
                                    device, WANDB_PROJECT+hyp['old_model_name'], wandb_run )
        # Dataset used to train the old model
        _, cifar100_old_test_loader = create_dataloaders('cifar100', DATASET_PATH,hyp['opt']['batch_size'],subset_list= hyp['data']['old_subset_list'])
    else:
        old_model = None

    # Create the dataloaders
    _, cifar100_test_loader = create_dataloaders('cifar100', DATASET_PATH, hyp['opt']['batch_size'], 
                                                                subset_list= hyp['data']['subset_list'])
    # If more than one model collect all the accuracies and NFR to calculate the mean value
    if hyp['num_models'] > 1:
        new_model_full_top1,new_model_full_top5 = [], []
        if hyp['nfr']: 
            new_model_old_top1, new_model_old_top5,  old_model_top1, old_model_top5 = [], [], [], []
            nfrs, impr_nfrs = [], []

    for i in range(hyp['num_models']):
        if hyp['num_models'] > 1:
            current_model_name = model_name.split(":v")[0]+":v"+str(i+int(model_name.split(":v")[1]))
        else:
            current_model_name = model_name

        # Load model
        model = create_model( hyp['net']['backbone'],False, hyp['net']['feat_dim'],hyp['data']['num_classes'], device,
                            WANDB_PROJECT+current_model_name, wandb_run )

        # Test the loaded model
        new_top1, new_top5 =test_model(model, cifar100_test_loader , wandb_run, step= i)

        # Calculate NFR       
        if hyp['nfr']:  
            nfr, _, _ = negative_flip_rate(old_model, model, cifar100_old_test_loader, dict_output= True)
            impr_nfr, _ , _ = improved_negative_flip_rate(old_model, model, cifar100_old_test_loader, dict_output=True)
            print(f"Negative flip rate : {nfr}")
            print(f"Improved negative flip rate: {impr_nfr}")
            wandb.log({'NFR':nfr, 'Improved NFR': impr_nfr}, step= i)

            # Test both models using only the classes used to train the old model
            new_old_top1, new_old_top5 = test_model(model, cifar100_old_test_loader, wandb_run,  step= i, wandb_log_name= " New model old test set")
            old_top1, old_top5 = test_model(old_model, cifar100_old_test_loader, wandb_run,  step= i, wandb_log_name= " Old model old test set")

        if hyp['num_models'] > 1:
            new_model_full_top1.append(new_top1.detach().cpu().numpy())
            new_model_full_top5.append(new_top5.detach().cpu().numpy())
            if hyp['nfr']: 
                new_model_old_top1.append(new_old_top1.detach().cpu().numpy())
                new_model_old_top5.append(new_old_top5.detach().cpu().numpy())
                old_model_top1.append(old_top1.detach().cpu().numpy())
                old_model_top5.append(old_top5.detach().cpu().numpy()) 
                nfrs.append(nfr)
                impr_nfrs.append(impr_nfr)

    if hyp['num_models'] > 1:
        wandb.log({'Mean new model top1 accuracy':np.mean(new_model_full_top1), 'Mean new model top5 accuracy': np.mean(new_model_full_top5)})
        if hyp['nfr']: 
            wandb.log({'Mean new model old test set top1 accuracy':np.mean(new_model_old_top1), 'Mean new model old test set top5 accuracy': np.mean(new_model_old_top5),
                       'Mean old model top1 accuracy':np.mean(old_model_top1), 'Mean old model top5 accuracy': np.mean(old_model_top5)})
            wandb.log({'Mean NFR':np.mean(nfrs), 'Mean Improved NFR': np.mean(impr_nfrs)})

    wandb.finish()

if __name__ == '__main__':
    main()