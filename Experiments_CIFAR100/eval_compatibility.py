import os
import sys
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np

from dataset import *
from models import *
from utils import *

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from compatibility_eval import cmc_evaluate
torch.backends.cudnn.benchmark = True

hyp = {
    'opt': {
        'batch_size': 256, 
    },
    'net': {
        'backbone': 'resnet18',         # resnet 18/34/50/101/152
        'feat_dim' : 512,                 # features' dimension
    },
    'data': {
        'num_classes_gallery_model': 100,
        'num_classes_query_model': 50,
        'subset_list': None,        #es: list [0,50,1] corresponds to list(range(0,50,1))
    },
    'gallery_model_name' : None,       # the name of the model that extracts feature from the gallery set
    'query_model_name' : None,        # the name of the model that extracts feature from the query set
    'num_models' : 1,                 # number of query models to evaluate
    'output_type': 'features',        # type of output to compare, features logits or softmax
}
# Transfer each loaded parameter to the correct hyp parameter
def define_hyp(loaded_params):
    hyp['net']['backbone'] = loaded_params['backbone']
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['data']['num_classes_gallery_model'] = loaded_params['num_classes_gallery_model']
    hyp['data']['num_classes_query_model'] = loaded_params['num_classes_query_model']
    subset_list = loaded_params['subset_list']
    if subset_list is not None:
        hyp['data']['subset_list']  = list(range(subset_list[0], subset_list[1], subset_list[2]))

    hyp['opt']['batch_size'] = loaded_params['batch_size']
    hyp['num_models'] = loaded_params['num_models']
    hyp['output_type'] = loaded_params['output_type']
   
    hyp['gallery_model_name'] = loaded_params['gallery_model_name']
    hyp['query_model_name'] = loaded_params['query_model_name']


def main():
     # Select config 
    parser = argparse.ArgumentParser(description='Training ResNet on CIFAR100')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    define_hyp(loaded_params)

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')
    DATASET_PATH = os.getenv('DATASET_PATH')

    wandb.login(key= WANDB_API_KEY, verify=True)

    device = torch.device("cuda:0")

    print("Evaluate compatibility "+ hyp['gallery_model_name'].split(":v")[0]+ "&"+hyp['query_model_name'].split(":v")[0])
    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "Softmax Evaluate compatibility "+ hyp['gallery_model_name'].split(":v")[0]+ "&"+hyp['query_model_name'].split(":v")[0],
        config=hyp)

    # Get the whole CIFAR100 dataset
    _, val_loader = create_dataloaders('cifar100', DATASET_PATH, hyp['opt']['batch_size'], subset_list= hyp['data']['subset_list'])

    # Load gallery model
    gallery_model = create_model( hyp['net']['backbone'],False, hyp['net']['feat_dim'],hyp['data']['num_classes_gallery_model'], device,
                            WANDB_PROJECT+hyp['gallery_model_name'], wandb_run )
    
    # collect all the performance metrics
    cmc_top1,cmc_top5, mAPs = [], [],[]
    for i in range(hyp['num_models']):
        if hyp['num_models'] > 1:
            current_query_model_name = hyp['query_model_name'].split(":v")[0]+":v"+str(i+int(hyp['query_model_name'].split(":v")[1]))
        else:
            current_query_model_name = hyp['query_model_name']
         # Load query model
        query_model = create_model( hyp['net']['backbone'],False, hyp['net']['feat_dim'],hyp['data']['num_classes_query_model'], device,
                                WANDB_PROJECT+current_query_model_name, wandb_run )

        cmc_out, mean_ap_out = cmc_evaluate(gallery_model, query_model, val_loader, device, distance_metric='cosine',
                                             output_type=hyp['output_type'] ,compute_map= True)

        print('CMC Top-1 = {}, CMC Top-5 = {}'.format(*cmc_out))
        print('mAP = {}'.format(mean_ap_out))
        wandb.log({'CMC Top-1':cmc_out[0], 'CMC Top-5': cmc_out[1], 'mAP': mean_ap_out})
        cmc_top1.append(cmc_out[0])
        cmc_top5.append(cmc_out[1])
        mAPs.append(mean_ap_out)
    
    wandb.log({'Mean CMC Top-1':np.mean(cmc_top1), 'Mean CMC Top-5': np.mean(cmc_top5), 'Mean mAP': np.mean(mAPs)})

    wandb.finish()

if __name__ == '__main__':
    main()