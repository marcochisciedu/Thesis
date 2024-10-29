import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
from numpy import linalg
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

# Calculates the adjecency matrix of the features
def calculate_adjecency_matrix(W):
    # Calculate the convex hull (Delaunay triangulation on the sphere)
    hull = ConvexHull(W)

    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((W.shape[0], W.shape[0]), dtype=int)
    
    # Update the adjacency matrix based on the convex hull simplices
    # Each simplex contains the indices of the 3 vertices it connects
    for simplex in hull.simplices:
        adjacency_matrix[simplex[0], simplex[1]] = 1
        adjacency_matrix[simplex[1], simplex[0]] = 1
        adjacency_matrix[simplex[1], simplex[2]] = 1
        adjacency_matrix[simplex[2], simplex[1]] = 1
        adjacency_matrix[simplex[2], simplex[0]] = 1
        adjacency_matrix[simplex[0], simplex[2]] = 1

    # Show the adjacency matrix
    return adjacency_matrix

# Get the squared frobenius norm of all the combinations of the adjacency matrices
def frobenius_norm_matrices(matrices):
    fro_norm_matrix = np.zeros((matrices.shape[0], matrices.shape[0]), dtype = int)

    for i in range(matrices.shape[0]):
        for j in range(matrices.shape[0]):
            fro_norm_matrix[i][j] = linalg.norm(np.dot(matrices[i].T, matrices[j]),ord= 'fro')**2
    
    return fro_norm_matrix

# Distance between all the vectors
def vectors_distances(W):
    distances = np.zeros((W.shape[0], W.shape[0]))

    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            squared_dist = np.sum((W[i]- W[j])**2, axis=0)
            distances[i][j] = np.sqrt(squared_dist)

    return distances

# Find the k nearest neighbors of each vector given their distances
def find_knn_vectors(distances, k):
    knn_vectors= np.empty((distances.shape[0]), dtype= object)

    for i in range(distances.shape[0]):
        # Find indices of the k nearest vector, do not include the self-distance
        knn_vectors[i]= np.argpartition(distances[i], k+1)[1:k+1]
    return knn_vectors

# Calculate the intersection between k nearest neighbors of all the features
def mutual_knn_alignment_features(W_array, k):
    knn_alignment = np.zeros((W_array.shape[0], W_array.shape[0]) ,dtype= float)
    knn_vectors = []
    # Find distances and knn vectors for each set of features
    for i in range(W_array.shape[0]):
        tmp_distances = vectors_distances(W_array[i])
        knn_vectors.append(find_knn_vectors(tmp_distances, k))
    
    # Find intersection between each possible features couple
    for row in range(W_array.shape[0]):
        for col in range(W_array.shape[0]):
            intersection_size = 0
            total_size = 0
            for feature in range(k):
                for value in knn_vectors[row][feature]:
                    if value in knn_vectors[col][feature]:
                        intersection_size += 1
                        total_size+= 1
                    else:
                        total_size +=1
            knn_alignment[row][col] = intersection_size/ total_size
    
    return knn_alignment




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
        name = "3D plot multiple runs"+ model_name[1:-4],
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)

    adj_matrices = []
    W_array = []

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
        W_array.append(W)
        figure= vector_plot(W)
        wandb.log({"3d plot" +  current_model_name[1:]: wandb.Plotly(figure)})

        # Calculate the adjecency matrix of the features
        adjacency_matrix=calculate_adjecency_matrix(W)
        adj_matrices.append(adjacency_matrix)
        classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        df_adjacency = pd.DataFrame(adjacency_matrix)
        df_adjacency.columns = classes
        df_adjacency.index = classes

        print(df_adjacency)
        wandb.log({"Adjecency matrix "+  current_model_name[1:]: wandb.Table(dataframe=df_adjacency)})
    
    # Get the frobenius norm matrix of all the adjecency matrices    
    adj_matrices = np.array(adj_matrices)
    fro_norm_matrix=frobenius_norm_matrices(adj_matrices)
    df_fro_norm = pd.DataFrame(fro_norm_matrix)

    print( df_fro_norm)
    wandb.log({"Frobenius norm matrix "+  model_name[1:-4]: wandb.Table(dataframe=df_fro_norm)})

    # Calculate mutual knn aligment between features
    W_array = np.array(W_array)
    for k in [2,3,4,5]:
        knn_alignment= mutual_knn_alignment_features(W_array, k)
        df_knn_alignment = pd.DataFrame(knn_alignment)
        print("k:" + str(k))
        print( df_knn_alignment)
        wandb.log({"KNN alignment, k: "+ str(k) + " " +  model_name[1:-4]: wandb.Table(dataframe=df_knn_alignment)})

    wandb_run.finish()
    


if __name__ == '__main__':
    main()