import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import plotly.graph_objs as go
from airbench94 import CifarLoader, make_net, evaluate
from scipy.spatial import ConvexHull
import pandas as pd



hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

def vector_plot(tvects,is_vect=True,orig=[0,0,0]):
    """Plot vectors using plotly"""

    if is_vect:
        if not hasattr(orig[0],"__iter__"):
            coords = [[orig,np.sum([orig,v],axis=0)] for v in tvects]
        else:
            coords = [[o,np.sum([o,v],axis=0)] for o,v in zip(orig,tvects)]
    else:
        coords = tvects

    data = []
    for i,c in enumerate(coords):
        X1, Y1, Z1 = zip(c[0])
        X2, Y2, Z2 = zip(c[1])
        vector = go.Scatter3d(x = [X1[0],X2[0]],
                              y = [Y1[0],Y2[0]],
                              z = [Z1[0],Z2[0]],
                              marker = dict(size = [0,5],
                                            color = ['blue'],
                                            line=dict(width=5,
                                                      color='DarkSlateGrey')),
                              name = 'Vector'+str(i+1))
        data.append(vector)

    layout = go.Layout(
             margin = dict(l = 4,
                           r = 4,
                           b = 4,
                           t = 4)
                  )
    fig = go.Figure(data=data,layout=layout)
    
    # Add convex hull
    hull_points=tvects[ConvexHull(tvects).vertices]

    fig.add_trace(go.Mesh3d(x=hull_points[:, 0], 
                        y=hull_points[:, 1], 
                        z=hull_points[:, 2], 
                        color="blue", 
                        opacity=.2,
                        alphahull=0))
    fig.show()

    return fig

# Distance between all the vectors
def vectors_distances(W):
    distances = np.zeros((W.shape[0], W.shape[0]))

    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            squared_dist = np.sum((W[i]- W[j])**2, axis=0)
            distances[i][j] = np.sqrt(squared_dist)

    return distances

def mean_std_matrices(matrices):
    means = np.zeros((matrices.shape[1], matrices.shape[2]))
    std = np.zeros((matrices.shape[1], matrices.shape[2]))

    for i in range(matrices.shape[1]):
        for j in range(matrices.shape[2]):
            means[i][j]=np.mean(matrices[:, i, j])
            std[i][j] = np.std(matrices[:, i ,j])

    return means, std

def main():

    # Select config 
    parser = argparse.ArgumentParser(description='Searching for unargmaxable class')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    model_name = loaded_params['model_name'][:-1]

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "3D plot multiple "+ model_name[1:-4],
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)

    matrices = []

    for i in range(5):
        current_model_name = model_name+ str(i)
        # Get model
        model = make_net()
        artifact = wandb_run.use_artifact(WANDB_PROJECT+current_model_name, type='model')
        artifact_dir = artifact.download()
        model.load_state_dict(torch.load(artifact_dir+'/model.pth'))

        # Evaluate model
        tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
        print("tta_val_acc for "+ current_model_name +": "+ str(tta_val_acc))

        eval_metrics = {'tta_val_acc': tta_val_acc}
        wandb.log({**eval_metrics})

        # Get model last layer's weights and plot its vectors
        W = model[8].weight.detach().cpu().numpy().astype(np.float32)
        figure= vector_plot(W)
        wandb.log({"3d plot" +  current_model_name[1:]: wandb.Plotly(figure)})

        distances=vectors_distances(W)
        matrices.append(distances)
        classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        df_distances = pd.DataFrame(distances)
        df_distances.columns = classes
        df_distances.index = classes

        print(df_distances)
        wandb.log({"Vector distances "+  current_model_name[1:]: wandb.Table(dataframe=df_distances)})
    matrices = np.array(matrices)
    means, std = mean_std_matrices(matrices)
    df_means = pd.DataFrame(means)
    df_means.columns = classes
    df_means.index = classes

    print(df_means)
    wandb.log({"Matrices mean "+ model_name[1:-4]: wandb.Table(dataframe=df_means)})

    df_std = pd.DataFrame(std)
    df_std.columns = classes
    df_std.index = classes
    
    print(df_std)
    wandb.log({"Matrices std "+  model_name[1:-4]: wandb.Table(dataframe=df_std)})

    wandb_run.finish()
    


if __name__ == '__main__':
    main()