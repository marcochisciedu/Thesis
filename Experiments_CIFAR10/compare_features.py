import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import torch
import plotly.graph_objs as go
from airbench94 import CifarLoader, make_net, evaluate
from scipy.spatial import ConvexHull
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Thesis')))
from negative_flip import *
from features_alignment import *


hyp = {
    'net': {
        'tta_level': 0,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
    'num_models': 1,
    'nfr' : False,                  # if, at the end of each epoch, the NFR will be calculated
    'old_model_name' : None,        # the name of the worse older model that is going to be used in NFR calculation
}

def vector_plot(tvects, classes, is_vect=True,orig=[0,0,0]):
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
                              name = classes[i])
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
                        opacity=.1,
                        alphahull=0))
    fig.show()

    return fig

def vector_features_plot(tvects, correct_feat, adj_nf_feat, non_adj_nf_feat, classes, is_vect=True,orig=[0,0,0]):
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
                              name = classes[i])
        data.append(vector)
    layout = go.Layout(
             margin = dict(l = 4,
                           r = 4,
                           b = 4,
                           t = 4)
                  )
    fig = go.Figure(data=data,layout=layout)

    # Add correct prediction's features
    x, y , z = correct_feat[:,0], correct_feat[:,1], correct_feat[:,2]
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z,
    mode='markers',
    marker=dict(
        size=2,
        color='limegreen'),
    name = 'Correct predictions'
    ))

    x, y , z = adj_nf_feat[:,0], adj_nf_feat[:,1], adj_nf_feat[:,2]
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z,
    mode='markers',
    marker=dict(
        size=2,
        color='blue'),
    name = 'Adjacent Negative flips'
    ))

    x, y , z = non_adj_nf_feat[:,0], non_adj_nf_feat[:,1], non_adj_nf_feat[:,2]
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z,
    mode='markers',
    marker=dict(
        size=2,
        color='red'),
    name = 'Non adjacent Negative flips'
    ))

    fig.show()

    return fig



def main():

    # Select config 
    parser = argparse.ArgumentParser(description='Feature alignment')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        type=str)
    params = parser.parse_args()
    
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    model_name = loaded_params['model_name'][:-1]
    hyp['nfr'] = loaded_params['nfr']
    if hyp['nfr'] == True:
        hyp['old_model_name'] = loaded_params['old_model']

    # Get env variables
    load_dotenv()

    # Setup wandb run
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    wandb.login(key= WANDB_API_KEY, verify=True)

    wandb_run=wandb.init(
        project=WANDB_PROJECT,
        name = "3D plotting negative flips "+ model_name[1:-4],
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    adj_matrices = []
    W_array = []

    for i in range(hyp['num_models']):
        current_model_name = model_name+ str(i)
        print(current_model_name)
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
        W_array.append(W)
        # 3d plot
        if hyp['num_models'] <=5:
            if hyp['nfr'] == True:
                old_model = make_net()
                artifact = wandb_run.use_artifact(WANDB_PROJECT+hyp['old_model_name'], type='model')
                artifact_dir = artifact.download()
                old_model.load_state_dict(torch.load(artifact_dir+'/model.pth'))
                class_index = 3
                correct_features, adj_nf_features, non_adj_nf_features= class_negative_flip_rate_features(class_index,old_model, model, test_loader)
                impr_correct_feat, impr_adj_nf_feat, impr_non_adj_feat = class_improved_negative_flip_rate_features(class_index,old_model, model, test_loader)
                
            vect_feat_fig= vector_features_plot(W,correct_features, adj_nf_features, non_adj_nf_features, classes)
            impr_vect_feat_fig = vector_features_plot(W,  impr_correct_feat, impr_adj_nf_feat, impr_non_adj_feat, classes )
            wandb.log({"3d plot with "+ str(classes[class_index]+ " features and nfr") : wandb.Plotly(vect_feat_fig)})
            wandb.log({"3d plot with "+ str(classes[class_index]+ " features and improved nfr") : wandb.Plotly(impr_vect_feat_fig)})


        # Calculate the adjacency matrix of the features
        adjacency_matrix=calculate_adjacency_matrix(W)
        adj_matrices.append(adjacency_matrix)
        # print each adjacency matrix
        if hyp['num_models'] <=5:
            df_adjacency = pd.DataFrame(adjacency_matrix, index = [i for i in classes], columns = [i for i in classes] )
            print(df_adjacency)
            wandb.log({"Adjacency matrix "+  current_model_name[1:]: wandb.Table(dataframe=df_adjacency)})

            fig=plt.figure(figsize = (12,7))
            sns.heatmap(df_adjacency, cmap="Purples", vmin=0, vmax=1, annot=True, fmt='d', linewidth=.5, square= True)
            plt.title('Adjacency matrix')
            plt.show()
            wandb.log({'Adjacency matrix heatmap' +current_model_name[1:]: wandb.Image(fig)})
        
    if hyp['num_models'] > 1:
        # Calculate mutual knn alignment between features
        W_array = np.array(W_array)
        for k in [2,3,4,5]:
            knn_alignment= mutual_knn_alignment_features(W_array, k)
            df_knn_alignment = pd.DataFrame(knn_alignment)
            print("k:" + str(k))
            print(f"KNN alignment, k: {k} \n {df_knn_alignment}")
            wandb.log({"KNN alignment, k: "+ str(k) + " " +  model_name[1:-4]: wandb.Table(dataframe=df_knn_alignment)})
            # Print the alignment between every model and the most/least/"medium" aligned models
            mean_alignment_vec= calculate_mean_alignment_vector(knn_alignment)
            indices = find_models_indices(mean_alignment_vec)
            fig = plot_alignment(knn_alignment, indices, k)
            wandb.log({'Mutual knn alignment, k:' + str(k): wandb.Image(fig)})
        
        #Calculate and show the sum of all the adjacency matrices
        df_summed_matrix, df_percent_matrix, fig_sum, fig_per= adjacency_matrices_sum(adj_matrices, classes)

        print(f"Sum of all adjacency matrices:\n {df_summed_matrix}")
        wandb.log({"Sum of all adjacency matrices "+  model_name[1:-4]: wandb.Table(dataframe=df_summed_matrix)})
        wandb.log({'Heatmap of the sum of all adjacency matrices': wandb.Image(fig_sum)})

        print(f"Percentage of all adjacency matrices:\n {df_percent_matrix}")
        wandb.log({"Percentage of all adjacency matrices "+  model_name[1:-4]: wandb.Table(dataframe=df_percent_matrix)})
        wandb.log({'Heatmap of the percentage of all adjacency matrices': wandb.Image(fig_per)})

    wandb_run.finish()
    


if __name__ == '__main__':
    main()