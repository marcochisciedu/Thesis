from print_and_logging import *
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

"""
Code to link the negative flips with the adjacency matrices
"""

# Calculate a negative flips heatmap showing how many times the old model's predictions (y axes) 
# got wrongly predicted as another class by the new model (x axes)
def flips_heatmap(flips, num_flips, classes, display= True, figsize = (12,7), annot = True):
    flip_matrix = np.zeros((len(classes), len(classes)), dtype = int)
    percentage_flip_matrix = np.zeros((len(classes), len(classes)), dtype = float)

    # create flip matrix
    for key in flips:
        flip_matrix[key[0],key[1]] = flips[key]
        percentage_flip_matrix[key[0],key[1]] = flips[key]*100/num_flips
    
    df_flips, fig = df_plot_heatmap(flip_matrix, classes,'Negative flips', 'Purples', 'd', "New prediction", "Old predictions", display=display,
                                    figsize = figsize, annot= annot )
    df_flips_perc, fig_perc = df_plot_heatmap(percentage_flip_matrix, classes,'Percentage of negative flips','Purples', '.1f',
                                              "New prediction", "Old predictions",  display=display, figsize = figsize, annot= annot )
    
    return fig, df_flips, fig_perc, df_flips_perc

# Create a new heatmap where the negative flips that correspond to two adjacent classes are the same
# while the others become negative
def link_NFR_adjacency(nfr_heatmap, adj_heatmap, display= True, figsize = (12,7), annot = True):
    nfr_matrix, adj_matrix = nfr_heatmap.to_numpy(), adj_heatmap.to_numpy()

    # Replace all the zeros with -1
    tmp_adj_matrix = np.copy(adj_matrix)
    tmp_adj_matrix[ tmp_adj_matrix == 0] = -1

    nfr_adj = np.multiply(nfr_matrix, tmp_adj_matrix)
    cmap = LinearSegmentedColormap.from_list('', ['red', 'white', 'blue'])
    df_nfr_adj, fig = df_plot_heatmap(nfr_adj,nfr_heatmap.index, 'Negative flips - Adjacency matrix', cmap, '.1f',
                                      "New prediction", "Old predictions", center=0, display=display, figsize = figsize, annot= annot )

    return fig, df_nfr_adj

# Creates, prints and plots a summary of all the NFR - adjacency matrices 
def summary_nfr_adj(nfr_adj_matrices, classes, impr = "",  figsize = (12,7), annot = True):
    # Count all the negative flips 
    sum_nfr_adj_mat =  np.sum(np.absolute(nfr_adj_matrices), axis = 0)
    df_sums, fig_sum = df_plot_heatmap(sum_nfr_adj_mat, classes,"All "+ impr+ " negative flips'", 'Purples', 'd', "New prediction", "Old predictions",
                                        figsize = figsize, annot= annot)
    print_and_log(df_sums, fig_sum, "All "+ impr+ " negative flips'")
    total_flips = np.sum(sum_nfr_adj_mat)

    # Replace all the positive numbers with 0, count all the negative flips between non-adjacent classes
    tmp_negative = np.copy(nfr_adj_matrices)
    tmp_negative[ tmp_negative >0] = 0
    negative_nfr_adj_mat = np.sum(np.absolute(tmp_negative), axis =0)
    neg_perc_nfr_adj_mat = negative_nfr_adj_mat* 100/total_flips
    df_neg, fig_neg = df_plot_heatmap(negative_nfr_adj_mat, classes,impr+' Negative flips of non-adjacent classes', 'Reds', 'd', "New prediction", "Old predictions",
                                       figsize = figsize, annot= annot)
    print_and_log(df_neg, fig_neg, impr+' Negative flips of non-adjacent classes')
    df_neg_per, fig_neg_per = df_plot_heatmap(neg_perc_nfr_adj_mat, classes,impr+ " Negative flips of non-adjacent classes' percentage ", 'Reds', '.1f',
                                              "New prediction", "Old predictions",  figsize = figsize, annot= annot)
    print_and_log(df_neg_per, fig_neg_per, impr+" Negative flips of non-adjacent classes' percentage ")
    neg_per = np.sum(neg_perc_nfr_adj_mat)

    # Replace all the negative numbers with 0, count all the negative flips between adjacent classes
    tmp_positive = np.copy(nfr_adj_matrices)
    tmp_positive[ tmp_positive <0] = 0
    positive_nfr_adj_mat = np.sum(tmp_positive, axis =0)
    potitive_perc_nfr_adj_mat = positive_nfr_adj_mat* 100/total_flips
    df_pos, fig_pos = df_plot_heatmap(positive_nfr_adj_mat, classes,impr+' Negative flips of adjacent classes', 'Blues', 'd', "New prediction", "Old predictions",
                                       figsize = figsize, annot= annot)
    print_and_log(df_pos, fig_pos,impr+' Negative flips of adjacent classes')
    df_pos_per, fig_pos_per = df_plot_heatmap(potitive_perc_nfr_adj_mat, classes, impr+" Negative flips of adjacent classes' percentage ", 'Blues', '.1f',
                                              "New prediction", "Old predictions",  figsize = figsize, annot= annot)
    print_and_log(df_pos_per, fig_pos_per,impr+" Negative flips of adjacent classes' percentage ")
    pos_per = np.sum(potitive_perc_nfr_adj_mat)
    return neg_per, pos_per