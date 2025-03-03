import numpy as np
import plotly.graph_objs as go
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


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
                              line=dict(width = 10),
                              marker = dict(size = [7,7],
                                            color = ['blue'],
                                            line=dict(width=100,
                                                      color='black')),
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

    # remove background
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    return fig

def plot_geodesic(ax, start, end, color='k', num=100):
    # Compute a set of points along the geodesic (great circle) between start and end
    t = np.linspace(0, 1, num)
    omega = np.arccos(np.clip(np.dot(start, end), -1.0, 1.0))
    points = np.outer(np.sin((1-t)*omega), start) + np.outer(np.sin(t*omega), end)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=0.5)

# Vector plot with the geodesic
def vector_plot_geodesic(tvects, classes, is_vect=True,orig=[0,0,0]):
    tvects = normalize(tvects, axis=1, norm='l2')

    # Calculate the convex hull (Delaunay triangulation on the sphere)
    hull = ConvexHull(tvects)

    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the sphere with solid surface, making occluded areas invisible
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='black', alpha=0.05, linewidth=0, antialiased=False)

    # Plot the points
    ax.scatter(tvects[:, 0], tvects[:, 1], tvects[:, 2], color='black', s=50)

    # Draw each edge of the simplex as a geodesic on the sphere
    for simplex in hull.simplices:
        for i in range(3):
            plot_geodesic(ax, tvects[simplex[i]], tvects[simplex[(i+1)%3]], color='black')

    # Draw vectors from origin to points
    colors = ['gray','gray','gray', 'purple', 'gray', 'green', 'gray', 'gray', 'gray', 'gray']
    for point, color, c in zip(tvects, colors, classes):
        ax.plot([0, point[0]], [0, point[1]], [0, point[2]], color=color, linestyle='-', linewidth=5.5, label = c)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()
    ax.view_init(elev=130, azim=10, roll=20)
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio
    plt.title('3D plot ')
    ax.set_axis_off()
    plt.savefig('3D plot.svg', format='svg')
    plt.show()

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
    #colors = ['gray','gray','gray', 'purple', 'gray', 'green', 'gray', 'gray', 'gray', 'gray']
    data = []
    for i,c in enumerate(coords):
        X1, Y1, Z1 = zip(c[0])
        X2, Y2, Z2 = zip(c[1])
        vector = go.Scatter3d(x = [X1[0],X2[0]],
                              y = [Y1[0],Y2[0]],
                              z = [Z1[0],Z2[0]],
                              line=dict(width = 12),            #color = colors[i] if needed
                              marker = dict(size = [7,7],
                                            color = ['blue'],
                                            line=dict(width=100,
                                                      color='black')),
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

    # remove background
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    fig.show()

    return fig