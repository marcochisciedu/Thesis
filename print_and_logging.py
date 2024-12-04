import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Simple function that prints and log a dataframe and a figure given their title
def print_and_log(df, fig, title):
    print(title+ f":\n {df}")
    wandb.log({title: wandb.Image(fig)})

# Simple function that creates a dataframe given a matrix, and plots its heatmap.
# Only creates the figure if it is going to be used
def df_plot_heatmap(matrix, classes, title, cmap, fmt, xlabel, ylabel, center=None,vmin=None, vmax=None, display= True, index = None, columns =None):
    # The model's classes are index and columns default value
    if index == None:
        index = classes
    if columns == None:
        columns = classes

    # Create the dataframe
    df = pd.DataFrame(matrix, index = index, columns = columns)
    # Create the heatmap
    if display:
        fig=plt.figure(figsize = (12,7))
        sns.heatmap(df, cmap=cmap, annot=True, fmt=fmt, linewidth=.5,vmin= vmin, vmax=vmax, center=center, square= True, robust= True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else: 
        fig = None
    return df, fig

# Simple function that creates the a bar plot figure from a list of values
def bar_plot(list, tick_labels, colors, title, xlabel, ylabel, ylim, yerr = None):
    array = np.array(list)
    x = np.arange(len(tick_labels))
    fig, ax = plt.subplots()
    ax.set(ylim=ylim)
    bars=ax.bar(x, array,color= colors, edgecolor= 'black', tick_label= tick_labels, label = tick_labels)
    ax.bar_label(bars)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if yerr is not None:
        plt.errorbar(x, array, yerr, fmt='.', color='Black', elinewidth=2,capthick=5,errorevery=1, alpha=0.5, ms=4, capsize = 2)
    plt.show()

    return fig
