import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Simple function that creates a dataframe given a matrix, and plots its heatmap.
# Only creates the figure if it is going to be used
def df_plot_heatmap(matrix, classes, title, cmap, fmt, xlabel, ylabel, center=None, display= True):
    df = pd.DataFrame(matrix, index = [i for i in classes], columns = [i for i in classes])
    if display:
        fig=plt.figure(figsize = (12,7))
        sns.heatmap(df, cmap=cmap, annot=True, fmt=fmt, linewidth=.5, center=center, square= True, robust= True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else: 
        fig = None
    return df, fig

# Simple function that prints and log a dataframe and a figure given their title
def print_and_log(df, fig, title):
    print(title+ f":\n {df}")
    wandb.log({title: wandb.Image(fig)})