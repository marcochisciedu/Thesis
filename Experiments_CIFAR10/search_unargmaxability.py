import os
import wandb
import argparse, yaml
import numpy as np
from dotenv import load_dotenv
import pandas

from airbench94 import CifarLoader, make_net, evaluate

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from unargmaxability import StolenProbabilitySearch, ApproxAlgorithms, ExactAlgorithms

import torch

hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
        'bias': False,          # if the last lineal layer has bias
        'feat_dim' : 3,         # features' dimension
    },
    'algorithm':{               # algorithm hyperparameters
        'patiance': 100, 
        'ub': 100,
        'lb': -100,
    }
}

# Check each class accuracy score
def check_unargmaxability(model, dataloader, classes):
    model.eval()
    outputs = []
    accuracy = [0] * len(classes)
    num_examples = [0]* len(classes)

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data

            # Get both models outputs
            logits = model(inputs)
            output = logits.argmax(1)
            outputs.extend(output.detach().cpu().numpy())

            # Calculate accuracy for each class
            for c in range(len(classes)):
                accuracy[c] += ((output == labels) * (labels == c)).float().sum().item()
                num_examples[c] += (labels == c).sum().item()
    # Print and log each class accuracy and search if each label is assigned at least once
    for i in range(len(classes)):
        accuracy[i] = accuracy[i]/num_examples[i]
        if i in outputs:
            print(str(i)+": " + classes[i]+ " is argmaxable with an accuracy of "+ str(accuracy[i]))
        else:
            print(str(i)+": "+ classes[i]+ " is never assigned with an accuracy of "+ str(accuracy[i]))

    return accuracy

    
def main():

    # Select config and algorithm
    parser = argparse.ArgumentParser(description='Searching for unargmaxable class')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
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
    hyp['net']['feat_dim'] = loaded_params['feat_dim']
    hyp['algorithm']['patience'] = loaded_params['patience']
    hyp['algorithm']['ub'] = loaded_params['logit_upper_bound']
    hyp['algorithm']['lb'] = loaded_params['logit_lower_bound']
    model_name = loaded_params['model_name']
    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "Unargmaxability" + model_name[1:-4],
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    # Cifar10 classes
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get model 
    model = make_net( hyp['net']['feat_dim'])
    artifact = wandb_run.use_artifact(WANDB_PROJECT+model_name, type='model')
    artifact_dir = artifact.download()
    model.load_state_dict(torch.load(artifact_dir+'/model.pth'))

    # Evaluate the model
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    print("tta_val_acc model: "+ str(tta_val_acc))
    eval_metrics = {'tta_val_acc': tta_val_acc}
    wandb.log({**eval_metrics}, commit = False)
    
    # Get the last layer's weights and bias
    print('Loading weights from model...')
    W = model[8].weight.detach().cpu().numpy().astype(np.float32)
    num_classes, dim = W.shape
    if hyp['net']['bias']:
        b = model[8].bias.detach().cpu().numpy().astype(np.float32).T
    else:
        b = None
    print('\tWeight matrix found with dim %s' % repr(W.shape))
    if b is not None:
        assert b.shape[0] == W.shape[0]
        print('\tBias vector found with dim %s' % repr(b.shape))

    # Run the unargamxability algorithm
    print('Asserting whether some classes are bounded in probability ...')
    print('\tUsing approximate algorithm *%s* ...' % params.approx_algorithm)
    if params.exact_algorithm is not None:
        print('\tUsing exact algorithm *%s* ...' % params.exact_algorithm)
    sp_search = StolenProbabilitySearch(W, b=b)
    results = sp_search.find_bounded_classes(class_list=list(range(num_classes)),
                                             exact_algorithm=params.exact_algorithm,
                                             approx_algorithm=params.approx_algorithm,
                                             lb= hyp['algorithm']['lb'],
                                             ub= hyp['algorithm']['ub'],
                                             patience= hyp['algorithm']['patience'])

    # Add classes to the results
    for r in results:
        r['class'] = classes[r['index']]

    # Print the bounded classes
    num_bounded = 0
    is_bounded = [False] * num_classes
    if params.approximate:
        print('Potentially bounded in probability:\n')
    else:
        print('Bounded in probability:\n')
    for each in results:
        if each['is_bounded']:
            num_bounded += 1
            is_bounded[each['index']] = True
            print('\t%d\t%s' % (each['index'], each['class']))
    print('*%d/%d* in total were found to be bounded' % (num_bounded, len(results)))
    
    # Check each class accuracy score and argmaxability on the test images
    accuracies= check_unargmaxability(model, test_loader, classes)

    data = {'Class': classes, 'is bounded': is_bounded, 'Accuracy': accuracies}
    df = pandas.DataFrame(data=data)
    wandb.log({"Unargmaxability table "+  model_name[1:-4]: wandb.Table(dataframe=df)})

    wandb_run.finish()


if __name__ == '__main__':
    main()