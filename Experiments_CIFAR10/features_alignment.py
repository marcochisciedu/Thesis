import os
import wandb
import argparse, yaml
from dotenv import load_dotenv
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
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

# Calculates the adjacency matrix of the features
def calculate_adjacency_matrix(W):
    # Calculate the convex hull (Delaunay triangulation on the sphere)
    hull = ConvexHull(W)

    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((W.shape[0], W.shape[0]), dtype=int)
    
    # Update the adjacency matrix based on the convex hull simplices
    # Each simplex contains the indices of the vertices it connects
    for simplex in hull.simplices:
        for i in range(simplex.shape[0]-1):
            for j in range(i+1, simplex.shape[0]):
                adjacency_matrix[simplex[i],simplex[j]]=1
                adjacency_matrix[simplex[j],simplex[i]]=1

    # Show the adjacency matrix
    return adjacency_matrix

# Calculate and show the sum of all the adjacency matrices
def adjacency_matrices_sum(adj_matrices, classes):
    summed_matrix = np.zeros((adj_matrices.shape[1], adj_matrices.shape[2]), dtype = int)

    # Sum all the adjacency matrices
    for i in range(adj_matrices.shape[0]):
        summed_matrix += adj_matrices[i]
    
    df_sum = pd.DataFrame(summed_matrix, index = [i for i in classes], columns = [i for i in classes])
    figure_sum=plt.figure(figsize = (12,7))
    sns.heatmap(df_sum, cmap="Purples", vmin=0, vmax=adj_matrices.shape[0], annot=True, fmt='d', linewidth=.5, square= True)
    plt.title('Sum of all the adjacency matrices')
    plt.show()
    
    # Create the correspondent percentages matrix
    percent_matrix = summed_matrix.astype(float)
    percent_matrix = percent_matrix/(adj_matrices.shape[0])*100

    df_sum_per = pd.DataFrame(percent_matrix, index = [i for i in classes], columns = [i for i in classes])
    figure_per= plt.figure(figsize = (12,7))
    sns.heatmap(df_sum_per, cmap="Purples", vmin=0, vmax=100, annot=True, fmt=".1f" , linewidth=.5, square= True)
    plt.title('Percentage of connections between classes')
    plt.show()
    
    return df_sum, df_sum_per, figure_sum, figure_per

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

# Given the knn alignment matrix calculate the mean alignment for each model
def calculate_mean_alignment_vector(knn_alignment_matrix):
    mean_alignment_vec = np.zeros((knn_alignment_matrix.shape[0]) ,dtype= float)

    # Remove the diagonal
    temp_mat= knn_alignment_matrix[~np.eye(knn_alignment_matrix.shape[0],dtype=bool)].reshape(knn_alignment_matrix.shape[0],-1)

    # Calculate the mean alignment of each model
    for i in range(knn_alignment_matrix.shape[0]):
        mean_alignment_vec[i] = np.mean(temp_mat[i])
    
    return mean_alignment_vec

# Find indices of the most, least and "medium" similar to the others
def find_models_indices(mean_alignment_vec):
    # Find most and least aligned models
    max_index = np.argmax(mean_alignment_vec)
    min_index = np.argmin(mean_alignment_vec)

    # Calculate the mean alignment value and find its closest model
    mean = np.mean(mean_alignment_vec)
    dist_from_mean = np.square(mean_alignment_vec -mean)
    mean_index = np.argmin(dist_from_mean)

    return  [min_index] + [mean_index] + [max_index] 

# Given the knn alignment matrix and the model index plot its alignment to the other models
def plot_alignment(knn_alignment_matrix, indices, k):

    fig, ax = plt.subplots()
    x = [knn_alignment_matrix[indices[0]], knn_alignment_matrix[indices[1]], knn_alignment_matrix[indices[2]]]
    colors = ['firebrick', 'royalblue' , 'limegreen']
    labels = [ 'least aligned', 'mean', 'most aligned']
    ax.hist(x, bins = 20, range = (0,1), color= colors, label = labels, edgecolor='black', linewidth=1)
    plt.legend(loc='upper left')
    plt.ylim(0, knn_alignment_matrix.shape[0])  # Set y-axis range 
    plt.title("Mutual knn alignment, k: "+ str(k))
    plt.xlabel("Mutual knn alignement")
    plt.ylabel("Number of models")
    plt.tight_layout()
    plt.show()

    return fig




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
        name = "Features alignment "+ model_name[1:-4],
        config=hyp)
    
    # Get test images
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)

    adj_matrices = []
    W_array = []

    for i in range(100):
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
        """ 3d plot
        figure= vector_plot(W)
        wandb.log({"3d plot" +  current_model_name[1:]: wandb.Plotly(figure)})
        """

        # Calculate the adjacency matrix of the features
        adjacency_matrix=calculate_adjacency_matrix(W)
        adj_matrices.append(adjacency_matrix)
        """ print each adjacency matrix
        classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        df_adjacency = pd.DataFrame(adjacency_matrix)
        df_adjacency.columns = classes
        df_adjacency.index = classes
    
        print(df_adjacency)
        wandb.log({"Adjacency matrix "+  current_model_name[1:]: wandb.Table(dataframe=df_adjacency)})
        """
    
    # Get the frobenius norm matrix of all the adjacency matrices    
    adj_matrices = np.array(adj_matrices)
    fro_norm_matrix=frobenius_norm_matrices(adj_matrices)
    df_fro_norm = pd.DataFrame(fro_norm_matrix)

    print(f"Frobenius norm matrix: \n {df_fro_norm}")
    wandb.log({"Frobenius norm matrix "+  model_name[1:-4]: wandb.Table(dataframe=df_fro_norm)})

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
    classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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