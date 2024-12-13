import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cosine
import torch
import copy
from sklearn.preprocessing import normalize

from print_and_logging import df_plot_heatmap


############################################
#                Adjacency                 #
############################################


# Calculates the adjacency matrix of the prototypes
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
def adjacency_matrices_sum(adj_matrices, classes, figsize = (12,7), annot = True):
    summed_matrix = np.zeros((adj_matrices.shape[1], adj_matrices.shape[2]), dtype = int)

    # Sum all the adjacency matrices
    for i in range(adj_matrices.shape[0]):
        summed_matrix += adj_matrices[i]

    df_sum, figure_sum = df_plot_heatmap(summed_matrix, classes,'Sum of all the adjacency matrices', "Purples", 'd', 
                                        "", "", vmin = 0, vmax =adj_matrices.shape[0] , figsize= figsize, annot= annot )
    
    # Create the correspondent percentages matrix
    percent_matrix = summed_matrix.astype(float)
    percent_matrix = percent_matrix/(adj_matrices.shape[0])*100

    df_sum_per, figure_per = df_plot_heatmap(percent_matrix, classes, 'Percentage of connections between classes', "Purples", '.1f',
                                             "", "", vmin=0, vmax = 100, figsize= figsize, annot= annot)
    
    return df_sum, df_sum_per, figure_sum, figure_per

############################################
#            Vectors distances             #
############################################

# Distance between all the vectors (usually prototypes)
def vectors_distances(W):
    distances = np.zeros((W.shape[0], W.shape[0]))

    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            distances[i][j] = cosine(W[i], W[j])

    return distances

# Distance between each vector (prototype) and the origin point
def origin_distances(W):
    distances = np.zeros(W.shape[0])

    for i in range(W.shape[0]):
        distances[i] = np.linalg.norm(W[i])
    return distances

############################################
#           Mutual knn-alignment           #
############################################

# Find the k nearest neighbors of each vector given their distances
def find_knn_vectors(distances, k):
    knn_vectors= np.empty((distances.shape[0]), dtype= object)

    for i in range(distances.shape[0]):
        # Find indices of the k nearest vector, do not include the self-distance
        knn_vectors[i]= distances[i].argsort()[1:k+1]
    return knn_vectors
# Calculate the intersection between the k nearest neighbors of two model's features
def mutual_knn_alignment_features(features_v1, features_v2, k):
    # Normalize the features to avoid overflow while calculating cosine distances
    distances_v1 = vectors_distances(normalize(features_v1, axis=1, norm='l2'))
    knn_vectors_v1 = find_knn_vectors(distances_v1, k)

    distances_v2 = vectors_distances(normalize(features_v2, axis=1, norm='l2'))
    knn_vectors_v2 = find_knn_vectors(distances_v2, k)
    
    # Calculate the intersection between each features' knn
    intersection = 0
    for i in range(len(knn_vectors_v1)):  # iterate through the features
        for value in knn_vectors_v1[i]:   # iterate through each nearest neighbor
            if value in knn_vectors_v2[i]:
                intersection += 1
                
    return intersection/k

# Convert old class indices to the new model's class indices
def old_to_new_class_indices(knn_old, indices):
    for index in range(len(indices)):
        for j in range(len((knn_old[index]))):
            knn_old[index][j] = indices[knn_old[index][j]]
       
    return knn_old

# Calculate the intersection between k nearest neighbors of a single model to all the others
def mutual_knn_alignment_one_model_prototype(W_old, W_array, k, indices):
    knn_alignment = np.zeros((W_array.shape[0]) ,dtype= float)
    knn_vectors = []
    # Find distances and knn vectors for each set of prototypes
    for i in range(W_array.shape[0]):
        tmp_distances = vectors_distances(W_array[i])
        knn_vectors.append(find_knn_vectors(tmp_distances, k)[indices])

    old_distances = vectors_distances(W_old[indices])
    knn_old= find_knn_vectors(old_distances,k)
    knn_old = old_to_new_class_indices(knn_old, indices)

    # Find intersection between each old-new model class prototype
    for W_index in range(W_array.shape[0]):   # iterate through each model
        intersection_size = 0
        total_size = 0
        for prototype in range(len(indices)):     # iterate through each class prototype
            for value in knn_vectors[W_index][prototype]:   # iterate through each nearest neighbor
                if value in knn_old[prototype]:
                    intersection_size += 1
                    total_size+= 1
                else:
                    total_size +=1
        knn_alignment[W_index] = intersection_size/ total_size
    
    return knn_alignment

# Given the knn alignment vector, k, the labels and the colors, plot the alignment between the old model and 
# each group of new models
def plot_alignment_old_other_models(knn_alignment_vector,k, title, labels, colors):
    fig, ax = plt.subplots()
    x = [knn_alignment_vector[index] for index in range(len(knn_alignment_vector))]
    ax.hist(x, bins = 20, range = (0,1), color= colors, label = labels, edgecolor='black', linewidth=1)
    plt.legend(loc='upper left')
    plt.ylim(0, knn_alignment_vector[0].shape[0])  # Set y-axis range 
    plt.title(title)
    plt.xlabel("Mutual knn alignement")
    plt.ylabel("Number of models")
    plt.tight_layout()
    plt.show()

    return fig

# Calculate the intersection between k nearest neighbors of all the prototypes
def mutual_knn_alignment_prototypes(W_array, k):
    knn_alignment = np.zeros((W_array.shape[0], W_array.shape[0]) ,dtype= float)
    knn_vectors = []
    # Find distances and knn vectors for each set of prototypes
    for i in range(W_array.shape[0]):
        tmp_distances = vectors_distances(W_array[i])
        knn_vectors.append(find_knn_vectors(tmp_distances, k))
    
    # Find intersection between each possible prototypes couple
    for row in range(W_array.shape[0]):
        for col in range(W_array.shape[0]):
            intersection_size = 0
            total_size = 0
            for prototype in range(k):
                for value in knn_vectors[row][prototype]:
                    if value in knn_vectors[col][prototype]:
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

############################################
#          Negative flip features          #
############################################

# Get the features of a given class, separates them in correct predictions, negative flip between classes that are adjacent
# and negative flips between non adjacent classes
def class_negative_flip_rate_features(class_index, model_v1, model_v2, test_loader, impr= False):
    all_correct_feat = []
    all_adj_flip_feat= []
    all_non_adj_flip_feat = []

    # Remove the last model's layers to extrate the images' features
    model_features_v2 = copy.deepcopy(model_v2)
    model_features_v2[-1] = torch.nn.Identity()
    model_features_v2[-2] = torch.nn.Identity()

    # Calculate adjacency matrix
    W_v2 = model_v2[8].weight.detach().cpu().numpy().astype(np.float32)
    adj_mat_v2 = calculate_adjacency_matrix(W_v2)
    # Indices of classes that are adjacent to the given class index
    adj_indices = (adj_mat_v2[class_index] == 1).nonzero()
    non_adj_indices = (adj_mat_v2[class_index] == 0).nonzero()

    model_v1.eval()
    model_v2.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # Use only the inputs of the selected class
            class_indices=(labels == class_index).nonzero(as_tuple=True)[0]

            # Get both models outputs and the features of the new model
            logits_v1 = model_v1(inputs[class_indices])
            output_v1 = logits_v1.argmax(1)

            logits_v2 = model_v2(inputs[class_indices])
            output_v2 = logits_v2.argmax(1)
            features_v2 = model_features_v2(inputs[class_indices])
            # Indices of correct predictions
            correct_indices = (output_v2 == class_index).nonzero(as_tuple=True)[0]
            correct_features = features_v2[correct_indices]
            all_correct_feat.extend(correct_features.cpu().numpy())
            
            if impr:
                # Improved negative flip if model_v2 is incorrect and its prediction is not the same as model_v1
                flipping = torch.logical_and(output_v2 != labels[class_indices],output_v2 != output_v1) 
            else:
                # Negative flip if model_v2 is wrong while model_v1 is correct
                flipping = torch.logical_and(output_v2 != labels[class_indices], output_v1 == labels[class_indices])

            flipping_adj = torch.logical_and(flipping, torch.isin(output_v2, torch.tensor(np.array(adj_indices)).cuda()))
            flipping_non_adj = torch.logical_and(flipping, torch.isin(output_v2, torch.tensor(np.array(non_adj_indices)).cuda()))
            
            # Features that correspond to negative flips, adjacent and non
            flip_indices_adj = flipping_adj.nonzero(as_tuple=True)[0]
            adj_nf_features = features_v2[flip_indices_adj]
            all_adj_flip_feat.extend(adj_nf_features.cpu().numpy())
            flip_indices_non_adj = flipping_non_adj.nonzero(as_tuple=True)[0]
            non_adj_nf_features = features_v2[flip_indices_non_adj]
            all_non_adj_flip_feat.extend(non_adj_nf_features.cpu().numpy())

            
    # Normalize all the features
    all_correct_feat = normalize(np.array(all_correct_feat), axis=1, norm='l2')
    all_adj_flip_feat = normalize(np.array(all_adj_flip_feat), axis=1, norm='l2')
    all_non_adj_flip_feat = normalize(np.array(all_non_adj_flip_feat), axis=1, norm='l2')

    return all_correct_feat, all_adj_flip_feat, all_non_adj_flip_feat

def class_negative_flip_rate_features_ResNet(class_index, model_v1, model_v2, test_loader, impr= False):
    all_correct_feat = []
    all_adj_flip_feat= []
    all_non_adj_flip_feat = []

    # Calculate adjacency matrix
    W_v2 = model_v2.fc2.weight.detach().cpu().numpy().astype(np.float32)
    adj_mat_v2 = calculate_adjacency_matrix(W_v2)
    # Indices of classes that are adjacent to the given class index
    adj_indices = (adj_mat_v2[class_index] == 1).nonzero()
    non_adj_indices = (adj_mat_v2[class_index] == 0).nonzero()

    model_v1.eval()
    model_v2.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # Use only the inputs of the selected class
            class_indices=(labels == class_index).nonzero(as_tuple=True)[0]

            # Get both models outputs and the features of the new model
            logits_v1 = model_v1(inputs[class_indices])['logits']
            output_v1 = logits_v1.argmax(1)

            logits_v2 = model_v2(inputs[class_indices])['logits']
            output_v2 = logits_v2.argmax(1)
            features_v2 =  model_v2(inputs[class_indices])['features']
            # Indices of correct predictions
            correct_indices = (output_v2 == class_index).nonzero(as_tuple=True)[0]
            correct_features = features_v2[correct_indices]
            all_correct_feat.extend(correct_features.cpu().numpy())
            
            if impr:
                # Improved negative flip if model_v2 is incorrect and its prediction is not the same as model_v1
                flipping = torch.logical_and(output_v2 != labels[class_indices],output_v2 != output_v1) 
            else:
                # Negative flip if model_v2 is wrong while model_v1 is correct
                flipping = torch.logical_and(output_v2 != labels[class_indices], output_v1 == labels[class_indices])

            flipping_adj = torch.logical_and(flipping, torch.isin(output_v2, torch.tensor(np.array(adj_indices)).cuda()))
            flipping_non_adj = torch.logical_and(flipping, torch.isin(output_v2, torch.tensor(np.array(non_adj_indices)).cuda()))
            
            # Features that correspond to negative flips, adjacent and non
            flip_indices_adj = flipping_adj.nonzero(as_tuple=True)[0]
            adj_nf_features = features_v2[flip_indices_adj]
            all_adj_flip_feat.extend(adj_nf_features.cpu().numpy())
            flip_indices_non_adj = flipping_non_adj.nonzero(as_tuple=True)[0]
            non_adj_nf_features = features_v2[flip_indices_non_adj]
            all_non_adj_flip_feat.extend(non_adj_nf_features.cpu().numpy())

            
    # Normalize all the features
    all_correct_feat = normalize(np.array(all_correct_feat), axis=1, norm='l2')
    all_adj_flip_feat = normalize(np.array(all_adj_flip_feat), axis=1, norm='l2')
    all_non_adj_flip_feat = normalize(np.array(all_non_adj_flip_feat), axis=1, norm='l2')

    return all_correct_feat, all_adj_flip_feat, all_non_adj_flip_feat