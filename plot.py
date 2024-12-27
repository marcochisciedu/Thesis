import numpy as np
import plotly.graph_objs as go
from scipy.spatial import ConvexHull

"""
Code to 3D plot a model's prototypes, either with a convex hull or adding images' features as color coded points
"""

# 3D plot with convex hull
def vector_plot(tvects, classes, is_vect=True,orig=[0,0,0], add_convex_hull = True):
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
    if add_convex_hull:
        hull_points=tvects[ConvexHull(tvects).vertices]

        fig.add_trace(go.Mesh3d(x=hull_points[:, 0], 
                            y=hull_points[:, 1], 
                            z=hull_points[:, 2], 
                            color="blue", 
                            opacity=.1,
                            alphahull=0))
    fig.show()

    return fig

# 3D plot with negative flips
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
    #Add negative flip in adjacent classes
    x, y , z = adj_nf_feat[:,0], adj_nf_feat[:,1], adj_nf_feat[:,2]
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z,
    mode='markers',
    marker=dict(
        size=2,
        color='blue'),
    name = 'Adjacent Negative flips'
    ))
    
    #Add negative flip in non-adjacent classes
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