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
torch.backends.cudnn.benchmark = True


hyp = {
    'opt': {
        'train_epochs': 200,
        'batch_size': 128,
        'lr': 0.1,                      
        'milestones': [60, 120, 160],   # learning rate scheduler's milestones
        'momentum': 0.9,
        'weight_decay': 5e-4,           
    },
    'net': {
        'backbone': 'resnet18',         # resnet 18/34/50/101/152
        'feat_dim' : 3,                 # features' dimension
        'pretrained' : False,           # if the trained model is fine tuned from a pretrained model

    },
    'data': {
        'num_classes': 100,
        'subset_list': None,        #es: list [0,50,1] corresponds to list(range(0,50,1))
    },
    'dSimplex': False,              # if the classifier is a dSimplex
    'seed' : 111,                   # for reproducibility
    'num_models' : 5,               # number of models to train
    'nfr' : False,                  # if, at the end of each epoch, the NFR will be calculated
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
    'li':{                         # Logit Inhibition (ELODI) parameters
        'ensemble_model' : None,
        'num_models': 5,
        'li_p': 2,
        'li_compute_topk': -1,
        'li_use_p_norm': False,
        'lambda' :1,
        'reduction' : 'mean',
    }
}

def define_hyp(loaded_params):
    hyp['net']['backbone'] = loaded_params['backbone']
    hyp['net']['pretrained'] = loaded_params['pretrained']
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['data']['num_classes'] = loaded_params['num_classes']
    subset_list = loaded_params['subset_list']
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
    if hyp['nfr'] or hyp['loss'] != 'default':
        hyp['old_model_name'] = loaded_params['old_model_name']
    
    if hyp['loss'] == 'Focal Distillation':
        hyp['fd']['fd_alpha'] = loaded_params['fd_alpha']
        hyp['fd']['fd_beta'] = loaded_params['fd_beta']
        hyp['fd']['focus_type'] = loaded_params['focus_type']
        hyp['fd']['distillation_type'] = loaded_params['distillation_type']
        hyp['fd']['kl_temperature'] = loaded_params['kl_temperature']
        hyp['fd']['lambda'] = loaded_params['lambda']
    hyp['seed'] = loaded_params['seed']

# Training the ResNet
def train(model, epoch, optimizer, dataloader, loss_function, wandb_run, old_model = None):

    model.train()
    training_loss = 0
    for batch_index, (images, labels) in enumerate(dataloader):

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        logits = model(images)

        if hyp['loss'] == 'Focal Distillation' and old_model is not None:
                # Get old model's prediction
                old_logits = old_model(images)
                fd_loss= FocalDistillationLoss(hyp['fd']['fd_alpha'], hyp['fd']['fd_beta'], hyp['fd']['focus_type'],
                                               hyp['fd']['distillation_type'], hyp['fd']['kl_temperature'] )
                loss_focal_distillation = fd_loss(logits, old_logits, labels)
                default_loss = loss_function(logits, labels).sum()
                loss = default_loss + hyp['fd']['lambda']*loss_focal_distillation
        else:
            loss = loss_function(logits, labels).sum()

        loss.backward()
        training_loss += loss.item()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}'.format(
            loss.item(),
            epoch=epoch,
            trained_samples=batch_index * hyp['opt']['batch_size'] + len(images),
            total_samples=len(dataloader.dataset)
        ))

    wandb_run.log({'train_loss': training_loss / len(dataloader.dataset)}, step= epoch)

@torch.no_grad()
def eval_training(model, epoch, dataloader, loss_function, wandb_run):

    model.eval()

    test_loss = 0.0 
    correct = 0.0

    for (images, labels) in dataloader:

        images = images.cuda()
        labels = labels.cuda()

        logits = model(images)
        loss = loss_function(logits, labels)

        test_loss += loss.item()
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum()

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch,
        test_loss / len(dataloader.dataset),
        correct.float() / len(dataloader.dataset),
    ))

    # Wandb log
    wandb_run.log({'val_loss':test_loss / len(dataloader.dataset),
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

    for i in range(hyp['num_models']):

        # reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(hyp['seed']+i)
        np.random.seed(hyp['seed']+i)

        print("Running "+ str(i+1) +"° training")
        wandb_run=wandb.init(
            project=WANDB_PROJECT,
            name = wandb_name,
            config=hyp)
        
        if hyp['nfr'] or hyp['loss'] != 'default':
            old_model = create_model(hyp['net']['backbone'], False, hyp['net']['feat_dim'], loaded_params['old_num_classes'], 
                                     device, WANDB_PROJECT+hyp['old_model_name'], wandb_run )
        else:
            old_model = None

        model = create_model( hyp['net']['backbone'], hyp['net']['pretrained'], hyp['net']['feat_dim'],hyp['data']['num_classes'], device )

        cifar100_train_loader, cifar100_test_loader = load_datasets('cifar100', DATASET_PATH, hyp['opt']['batch_size'], hyp['data']['subset_list'])

        optimizer = optim.SGD(model.parameters(), lr=hyp['opt']['lr'], momentum=hyp['opt']['momentum'], weight_decay=hyp['opt']['weight_decay'])
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=hyp['opt']['milestones'], gamma=0.2) #learning rate decay
        cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')

        best_acc = 0
        for epoch in range(1, hyp['opt']['train_epochs'] + 1):
        
            train(model, epoch, optimizer, cifar100_train_loader, cross_entropy_loss, wandb_run, old_model)
            train_scheduler.step(epoch)
            val_acc = eval_training(model, epoch, cifar100_test_loader, cross_entropy_loss, wandb_run)

            # TODO: fix: when new model, select a subset of CIFAR100, the same as the old model
            if hyp['nfr']:
                nfr, _, _ = negative_flip_rate(old_model, model, cifar100_test_loader)
                impr_nfr, _ , _ = improved_negative_flip_rate(old_model, model, cifar100_test_loader)
                print(f"Negative flip rate at epoch {epoch}: {nfr}")
                print(f"Improved negative flip rate at epoch {epoch}: {impr_nfr}")
                wandb.log({'NFR':nfr, 'Improved NFR': impr_nfr}, step= epoch)


            # Start to save best performance model after first milestone
            if epoch > hyp['opt']['milestones'] and best_acc < val_acc:
                best_acc=val_acc
                # Save all the parameters of the model
                model_state_dict = model.state_dict()

        # Load saved weights of the best model
        model.load_state_dict(model_state_dict)

        # Save the best model on weights and biases as an artifact
        best_model_artifact = wandb.Artifact(
                    wandb_name, type="model",
                    description="best model for "+ wandb_name,
                    metadata=hyp)

        torch.save(model.state_dict(), "best_model.pth")
        best_model_artifact.add_file("best_model.pth")
        wandb.save("best_model.pth")
        wandb_run.log_artifact(best_model_artifact)



if __name__ == '__main__':
    main()