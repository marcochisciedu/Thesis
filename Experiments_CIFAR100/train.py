import os
import sys
import wandb
import argparse, yaml
import numpy as np
from dotenv import load_dotenv
import torch.optim as optim

from dataset import *
from models import *
from utils import *

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from NFR_losses import *
from fixed_classifiers import *
from features_alignment import calculate_knn_matrix
torch.backends.cudnn.benchmark = True

"""
Code used to train Incremental ResNets on CIFAR100.
The new models can be trained with different losses: cross entropy and focal distillation.
"""
hyp = {
    'opt': {
        'train_epochs': 200,
        'batch_size': 256,
        'lr': 0.1,     
        'milestones': [60, 120, 160],   # learning rate scheduler's milestones                 
        'momentum': 0.9,
        'weight_decay': 5e-4,           
    },
    'net': {
        'backbone': 'resnet18',         # resnet 18/34/50/101/152
        'feat_dim' : 512,               # features' dimension
        'pretrained' : False,           # if the trained model is fine tuned from a pretrained model
        'old_backbone': 'resnet18',     # backbone of the old model
        'old_feat_dim': 512,            # feat dim of the old model
    },
    'data': {
        'num_classes': 100,
        'subset_list': None,        #es: list [0,50,1] corresponds to list(range(0,50,1))
        'old_subset_list': None,    # subset list of the old model, to create the correct dataloader to calculate NFR
    },
    'dSimplex': False,              # if the classifier is a dSimplex
    'seed' : 111,                   # for reproducibility
    'num_models' : 1,               # number of models to train
    'nfr' : False,                  # if, at the end of each epoch, the NFR will be calculated
    'nfr_eval' : 20,                # after how many epochs to evaluate the NFR
    'old_model_name' : None,        # the name of the worse older model that is going to be used in NFR calculation
    'loss' : 'default' ,            # training loss used
    'fd' :{                         # focal distillation parameters
        'fd_alpha' : 1,
        'fd_beta' : 5,
        'focus_type' : 'old_correct',
        'distillation_type' : 'kl',
        'kl_temperature' : 100,
        'lambda' : 1,
    },
}
# Transfer each loaded parameter to the correct hyp parameter
def define_hyp(loaded_params):
    hyp['net']['backbone'] = loaded_params['backbone']
    hyp['net']['pretrained'] = loaded_params['pretrained']
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['data']['num_classes'] = loaded_params['num_classes']
    subset_list = loaded_params['subset_list']
    if subset_list is not None:
        hyp['data']['subset_list']  = list(range(subset_list[0], subset_list[1], subset_list[2]))
    hyp['num_models'] = loaded_params['num_models']

    hyp['opt']['train_epochs'] = loaded_params['train_epochs']
    hyp['opt']['batch_size'] = loaded_params['batch_size']
    hyp['opt']['lr'] = loaded_params['lr']
    hyp['opt']['milestones'] = loaded_params['milestones']
    hyp['opt']['momentum'] = loaded_params['momentum']
    hyp['opt']['weight_decay'] = loaded_params['weight_decay']
   
    hyp['loss']= loaded_params['loss']
    hyp['nfr'] = loaded_params['nfr']
    if hyp['nfr']:
        hyp['nfr_eval'] = loaded_params['nfr_eval']
    if hyp['nfr'] or hyp['loss'] != 'default': 
        hyp['old_model_name'] = loaded_params['old_model_name']
        old_subset_list = loaded_params['old_subset_list']
        hyp['data']['old_subset_list'] = list(range(old_subset_list[0], old_subset_list[1], old_subset_list[2]))
        hyp['net']['old_backbone'] = loaded_params['old_backbone']
        hyp['net']['old_feat_dim'] = loaded_params['old_feat_dim']
    FD = False
    if hyp['loss'] == 'New stuff':
        hyp['FD'] = loaded_params['FD']
        if hyp['FD']:
            FD = True
        hyp['CF'] = loaded_params['CF']
        if hyp['CF']:
            hyp['tau_f']= loaded_params['tau_f']
            hyp['lambda_f']= loaded_params['lambda_f']
            hyp['only_old']= loaded_params['only_old']
        hyp['CP'] = loaded_params['CP']
        if hyp['CP']:
            hyp['tau_p']= loaded_params['tau_p']
            hyp['lambda_p']= loaded_params['lambda_p']
        hyp['CPL'] = loaded_params['CPL']
        if hyp['CPL']:
            hyp['lambda_cpl']= loaded_params['lambda_cpl']
    if (hyp['loss'] == 'Focal Distillation') or FD:
        hyp['fd']['fd_alpha'] = loaded_params['fd_alpha']
        hyp['fd']['fd_beta'] = loaded_params['fd_beta']
        hyp['fd']['focus_type'] = loaded_params['focus_type']
        hyp['fd']['distillation_type'] = loaded_params['distillation_type']
        hyp['fd']['kl_temperature'] = loaded_params['kl_temperature']
        hyp['fd']['lambda'] = loaded_params['lambda']
    hyp['seed'] = loaded_params['seed']
    hyp['dSimplex'] = loaded_params['dSimplex']


# Function that executes one epoch of training the model
def train(model, epoch, optimizer, dataloader, loss_function, wandb_run, old_model = None):

    model.train()
    losses = []

    if hyp['loss'] == 'New stuff':
        if hyp['CF']: 
            contr_feat_loss = ContrastiveFeaturesLoss(hyp['tau_f'])
        if hyp['CP']: 
            contr_proto_loss = ContrastivePrototypeLoss(hyp['tau_p'])
        if hyp['CPL']:
            cosine_loss = CosinePrototypeLoss()
        if hyp['FD']: 
            fd_loss= FocalDistillationLoss(hyp['fd']['fd_alpha'], hyp['fd']['fd_beta'], hyp['fd']['focus_type'],
                                        hyp['fd']['distillation_type'], hyp['fd']['kl_temperature'] )
    elif hyp['loss'] == 'Focal Distillation':
        fd_loss= FocalDistillationLoss(hyp['fd']['fd_alpha'], hyp['fd']['fd_beta'], hyp['fd']['focus_type'],
                                               hyp['fd']['distillation_type'], hyp['fd']['kl_temperature'] )

    for (images, labels) in dataloader:

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)  # the model's output is a dictionary

        if hyp['loss'] == 'New stuff':
            loss_CE = loss_function(outputs['logits'], labels).sum()
            loss = loss_CE 
            if hyp['CF']: 
                # Extract and normalize the old and new features
                new_features = l2_norm(outputs['features'])
                old_outputs= old_model(images)
                old_features = l2_norm(old_outputs['features'])
                
                loss_CF = contr_feat_loss(old_features, new_features, labels,hyp['data']['num_classes'],
                                            len( hyp['data']['old_subset_list']), hyp['only_old'])
                loss += hyp['lambda_f']*loss_CF  # add contrastive features loss
            if hyp['CP']: 
                # Extract and normalize the old and new model's class prototypes
                new_prototypes = model.fc2.weight
                new_prototypes = l2_norm(new_prototypes)
                old_prototypes = old_model.fc2.weight
                old_prototypes = l2_norm(old_prototypes)

                loss_CP = contr_proto_loss(old_prototypes, new_prototypes)
                loss += hyp['lambda_p']*loss_CP # add contrastive prototypes loss
            if hyp['CPL']: 
                # Extract and normalize the old and new model's class prototypes
                new_prototypes = model.fc2.weight
                new_prototypes = l2_norm(new_prototypes)
                old_prototypes = old_model.fc2.weight
                old_prototypes = l2_norm(old_prototypes)

                loss_CPL = cosine_loss(old_prototypes, new_prototypes)
                loss += hyp['lambda_cpl']*loss_CPL # add cosine prototypes loss
            if hyp['FD']: # add focal distillation
                old_logits = old_model(images)['logits'] 
                loss_focal_distillation = fd_loss(outputs['logits'], old_logits, labels)
                loss += hyp['fd']['lambda']*loss_focal_distillation
        elif hyp['loss'] == 'Focal Distillation':
            # Get old model's prediction
            old_logits = old_model(images)['logits'] 
            loss_focal_distillation = fd_loss(outputs['logits'], old_logits, labels)
            default_loss = loss_function(outputs['logits'], labels).sum()
            loss = default_loss + hyp['fd']['lambda']*loss_focal_distillation
        else:
            loss = loss_function(outputs['logits'], labels).sum() 

        loss.backward()
        losses.append( loss.item())
        optimizer.step()

    print('Training Epoch: {epoch} \tLoss: {:0.4f}'.format(np.mean(losses) ,epoch=epoch,))
    wandb_run.log({'train_loss': np.mean(losses)}, step= epoch)

# Function that tests the model after an epoch of training
@torch.no_grad()
def eval_training(model, epoch, dataloader, loss_function, wandb_run):

    model.eval()

    test_losses = []
    correct = 0.0

    for (images, labels) in dataloader:

        images = images.cuda()
        labels = labels.cuda()

        logits = model(images)['logits']  # the model's output is a dictionary
        loss = loss_function(logits, labels)

        test_losses.append( loss.item())
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum()

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch,
        np.mean(test_losses),
        correct.float() / len(dataloader.dataset)))

    # Wandb log
    wandb_run.log({'val_loss':np.mean(test_losses),
                   'val_accuracy': correct.float() / len(dataloader.dataset)}, step= epoch)

    return correct.float() / len(dataloader.dataset)

def main():
    # Select config 
    parser = argparse.ArgumentParser(description='Training ResNet on CIFAR100')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    wandb_name = loaded_params['wandb_name']
    define_hyp(loaded_params)

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')
    DATASET_PATH = os.getenv('DATASET_PATH')

    wandb.login(key= WANDB_API_KEY, verify=True)

    device = torch.device("cuda:0")

    for i in range(hyp['num_models']): # iterate for each model that needs to be trained

        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(hyp['seed']+i)
        np.random.seed(hyp['seed']+i)

        print("Running "+ str(i+1) +"° training")
        wandb_run=wandb.init(
            project=WANDB_PROJECT,
            name = wandb_name,
            config=hyp)
        # Get old model if needed to calculate NFR or a loss
        if hyp['nfr'] or hyp['loss'] != 'default':
            old_model = create_model(hyp['net']['old_backbone'], False, hyp['net']['old_feat_dim'], len( hyp['data']['old_subset_list']), 
                                     device, WANDB_PROJECT+hyp['old_model_name'], wandb_run )
            # Get the correct dataset to test the NFR
            print(f'Creating and loading Negative Flip Rate dataset')
            _, cifar100_nfr_test_loader = create_dataloaders('cifar100', DATASET_PATH,hyp['opt']['batch_size'],subset_list= hyp['data']['old_subset_list'])
        else:
            old_model = None

        # Create new model
        model = create_model( hyp['net']['backbone'], hyp['net']['pretrained'], hyp['net']['feat_dim'],hyp['data']['num_classes'], device )

        # Create the dataloaders
        print(f'Creating and loading training and testing dataloaders')
        cifar100_train_loader, cifar100_test_loader = create_dataloaders('cifar100', DATASET_PATH, hyp['opt']['batch_size'], 
                                                                    subset_list= hyp['data']['subset_list'])

        # Set up training
        optimizer = optim.SGD(model.parameters(), lr=hyp['opt']['lr'], momentum=hyp['opt']['momentum'], weight_decay=hyp['opt']['weight_decay'])
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=hyp['opt']['milestones'], gamma=0.2) #learning rate decay
        loss = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(1, hyp['opt']['train_epochs'] + 1):
            # Train and evaluate the model
            train(model, epoch, optimizer, cifar100_train_loader, loss, wandb_run, old_model)
            train_scheduler.step()
            val_acc = eval_training(model, epoch, cifar100_test_loader, loss, wandb_run)

            # Calculate NFR after nfr_eval epochs or if it is the last epoch
            if hyp['nfr'] and (epoch % hyp['nfr_eval'] == 0 or epoch == hyp['opt']['train_epochs']):  
                nfr, _, _ = negative_flip_rate(old_model, model, cifar100_nfr_test_loader, dict_output= True)
                impr_nfr, _ , _ = improved_negative_flip_rate(old_model, model, cifar100_nfr_test_loader, dict_output=True)
                print(f"Negative flip rate at epoch {epoch}: {nfr}")
                print(f"Improved negative flip rate at epoch {epoch}: {impr_nfr}")
                wandb.log({'NFR':nfr, 'Improved NFR': impr_nfr}, step= epoch)

            # Start to save best performance model after second milestone
            if epoch > hyp['opt']['milestones'][1] and best_acc < val_acc:
                best_acc=val_acc
                # Save all the parameters of the model
                model_state_dict = model.state_dict()

        # Load saved weights of the best model
        model.load_state_dict(model_state_dict)

        # Save the best model on weights and biases as an artifact
        best_model_artifact = wandb.Artifact(
                    wandb_name.replace(" ", "_"), type="model",
                    description="best model for "+ wandb_name,
                    metadata=hyp)

        torch.save(model.state_dict(), "best_model.pth")
        best_model_artifact.add_file("best_model.pth")
        wandb.save("best_model.pth")
        wandb_run.log_artifact(best_model_artifact)

        wandb.finish()


if __name__ == '__main__':
    main()