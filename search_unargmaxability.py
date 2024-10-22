import os
import wandb
import argparse, yaml
import numpy as np
from dotenv import load_dotenv

from unargmaxability import StolenProbabilitySearch, ApproxAlgorithms, ExactAlgorithms

import torch

from airbench94 import CifarLoader, make_net, evaluate

hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
        'bias': False,
    },
    'algorithm':{
        'patiance': 100, 
        'ub': 100,
        'lb': -100,
    }
}


def main():

    # Select config
    parser = argparse.ArgumentParser(description='Searching for unargmaxable class')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        default=os.path.join(os.getcwd(), "configs/half_CIFAR10.yaml"), 
                        type=str)
    parser.add_argument('--approx-algorithm', default=ApproxAlgorithms.default(),
                        choices=ApproxAlgorithms.choices(),
                        help='Choice of approximate algorithm. Default: %s' %
                        ApproxAlgorithms.default())
    parser.add_argument('--exact-algorithm', default=ExactAlgorithms.default(),
                        choices=ExactAlgorithms.choices(),
                        help='Choice of exact algorithm. Default: %s' %
                        ExactAlgorithms.default())
    params = parser.parse_args()

    if params.exact_algorithm == 'none':
        params.exact_algorithm = None
        params.approximate = True
    else:
        params.approximate = False

    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    # Modify the default hyperparameters
    hyp['algorithm']['patience'] = loaded_params['patience']
    hyp['algorithm']['ub'] = loaded_params['logit_upper_bound']
    hyp['algorithm']['lb'] = loaded_params['logit_lower_bound']

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "Unargmaxability CIFAR10",
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    # Cifar10 classes
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get model trained with lower percentages of data in some classes
    low_model = make_net()
    artifact = wandb_run.use_artifact(WANDB_PROJECT+'/CIFAR10_100_5percent_9.9epochs:v18', type='model')
    artifact_dir = artifact.download()
    low_model.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    # Get model trained on all the images
    model = make_net()
    artifact = wandb_run.use_artifact(WANDB_PROJECT+'/CIFAR10_100_100percent_9.9epochs:v14', type='model')
    artifact_dir = artifact.download()
    model.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    # Evaluate both models
    low_tta_val_acc = evaluate(low_model, test_loader, tta_level=hyp['net']['tta_level'])
    print("tta_val_acc low model: "+ str(low_tta_val_acc))
    full_tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    print("tta_val_acc full model: "+ str(full_tta_val_acc))

    eval_metrics = {'low_tta_val_acc': low_tta_val_acc, 'full_tta_val_acc': full_tta_val_acc}
    wandb.log({**eval_metrics})
    
    # Get the last layer's weights and bias
    print('Loading weights from low model...')
    W_low = low_model[8].weight.detach().cpu().numpy().astype(np.float32)
    num_classes, dim = W_low.shape
    if hyp['net']['bias']:
        b_low = low_model[8].bias.detach().cpu().numpy().T
    else:
        b_low = None
    print('\tWeight matrix found with dim %s' % repr(W_low.shape))
    if b_low is not None:
        assert b_low.shape[0] == W_low.shape[0]
        print('\tBias vector found with dim %s' % repr(b_low.shape))

    # Run the unargamxability algorithm
    print('Asserting whether some classes are bounded in probability ...')
    print('\tUsing approximate algorithm *%s* ...' % params.approx_algorithm)
    if params.exact_algorithm is not None:
        print('\tUsing exact algorithm *%s* ...' % params.exact_algorithm)
    sp_search = StolenProbabilitySearch(W_low, b=b_low)
    results = sp_search.find_bounded_classes(class_list=list(range(num_classes)),
                                             exact_algorithm=params.exact_algorithm,
                                             approx_algorithm=params.approx_algorithm,
                                             lb= hyp['algorithm']['lb'],
                                             ub= hyp['algorithm']['ub'],
                                             patience= hyp['algorithm']['patience'])

    # Add classes to the results
    for r in results:
        r['class'] = classes[r['index']]

    num_bounded = 0

    if params.approximate:
        print('Potentially bounded in probability:\n')
    else:
        print('Bounded in probability:\n')
    for each in results:
        if each['is_bounded']:
            num_bounded += 1
            print('\t%d\t%s' % (each['index'], each['class']))
    print('*%d/%d* in total were found to be bounded' % (num_bounded, len(results)))


    wandb_run.finish()


if __name__ == '__main__':
    main()