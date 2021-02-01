import numpy as np
import matplotlib.pyplot as plt
import os
from GeomCA import GeomCA
from scipy.spatial import distance


# -------------------------------------------------------------------------- #
# Functions for drawing points in 2D
# -------------------------------------------------------------------------- #
def circle(n=5000, r=1, noise=0.05):
    """
    Samples points from a circle.
    :param n: number of points.
    :param r: circle radius.
    :param noise: amount of gaussian noise to misplace points.
    """
    phis = 2 * np.pi * np.random.rand(n)
    x = [[r * np.sin(phi), r * np.cos(phi)] for phi in phis]
    x = np.array(x)
    x = x + noise * np.random.randn(n, 2)
    return x

def half_circle(n=5000, r=1, noise=0.05):
    """
    Samples points from a half circle.
    :param n: number of points.
    :param r: circle radius.
    :param noise: amount of gaussian noise to misplace points.
    """
    phis = np.pi * np.random.rand(n)
    x = [[r * np.sin(phi), r * np.cos(phi)] for phi in phis]
    x = np.array(x)
    x = x + noise * np.random.randn(n, 2)
    return x

def ball(n=5000, center=[4, 5]):
    """
    Samples points from a unit ball.
    :param n: number of points.
    """
    ans = []
    while len(ans) < n:
        x = np.random.rand(2) * 2.0 - 1.0
        ans.append(np.add(x, center))
    return np.array(ans) + 0.05 * np.random.randn(n, 2)

def get_color(edge, nR):
    """
    Gets the color of the edge.
    :param edge: edge given as a list of two indices.
    :param nR: number of R points in the graph.
    """
    R_color, E_color = 'C0', 'C1'
    edge = sorted(edge)
    if edge[0] < nR:
        if edge[1] > nR:
            comp_color = 'gray'
            zorder = 10
        else:
            comp_color = R_color
            zorder = 5
    else:
        comp_color = E_color
        zorder = 5
    return comp_color, zorder

# -------------------------------------------------------------------------- #
# Runs and visualizes GeomCA
# -------------------------------------------------------------------------- #
def run_GeomCA_and_visualize(R, E, subfolder, GeomCA_parameters):
    """
    Runs GeomCA and visualizes the results
    :param R: array of R points.
    :param E: array of E points.
    :param subfolder: subfolder to save results to.
    :param GeomCA_parameters: dictionary containing GeomCA parameters.
    :return: GeomCA result dictionaries: components statistics, network statistics and network parameters
    """
    GeomCA_graph = GeomCA(R, E, GeomCA_parameters, load_existing=False, subfolder=subfolder)
    GeomCA_results = GeomCA_graph.get_connected_components()

    # Plot results
    GeomCA_graph.plot_connected_components()
    GeomCA_graph.plot_RE_components_consistency(min_comp_size=1, annotate_largest=True, display_smaller=False)
    GeomCA_graph.plot_RE_components_quality(min_comp_size=1, annotate_largest=True, display_smaller=False)
    GeomCA_graph.plot_RE_components_quality(min_comp_size=5, annotate_largest=True, display_smaller=True)

    save_path = os.path.join(GeomCA_parameters['experiment_path'], subfolder, GeomCA_parameters['experiment_filename_prefix'])
    vertices = np.concatenate([R, E])
    edges = list(GeomCA_graph.RE_network.edges)
    plt.figure(1, figsize=(5, 5))
    plt.clf()
    plt.scatter(R[:, 0], R[:, 1], s=10, color=GeomCA_graph.R_color, alpha=0.7)
    plt.scatter(E[:, 0], E[:, 1], s=10, color=GeomCA_graph.E_color, alpha=0.7)
    # draw edges of correct color
    for e in edges:
        color, zorder = get_color(e, len(R))
        start, end = vertices[e[0]], vertices[e[1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], '-', linewidth=1.0, color=color, zorder=zorder)
    plt.axis('off')
    plt.savefig(save_path + 'GeomCA_graph')
    return GeomCA_results

# -------------------------------------------------------------------------- #
# Functions for running GeomCA on various examples
# -------------------------------------------------------------------------- #
def example1_perfect_overlap(GeomCA_parameters):
    """
    Executes GeomCA on points from two circles with perfect overlap.
    :param GeomCA_parameters: dictionary containing GeomCA parameters.
    :return: GeomCA result dictionaries: components statistics, network statistics and network parameters
    """
    num_pts = 100
    GeomCA_parameters['experiment_filename_prefix'] = 'perfect_overlap_'
    subfolder = 'perfect_overlap'
    R = np.concatenate([circle(n=num_pts, r=1.5), circle(n=int(num_pts/5), r=0.3)])
    E = np.concatenate([circle(n=num_pts, r=1.45), circle(n=int(num_pts/5), r=0.32)])
    return run_GeomCA_and_visualize(R, E, subfolder, GeomCA_parameters)

def example2_concatenated_RE(GeomCA_parameters):
    """
    Executes GeomCA on points from two circles with concatenation.
    :param GeomCA_parameters: dictionary containing GeomCA parameters.
    :return: GeomCA result dictionaries: components statistics, network statistics and network parameters
    """
    num_pts = 100
    GeomCA_parameters['experiment_filename_prefix'] = 'problematic_overlap_'
    subfolder = 'problematic_overlap'
    GeomCA_parameters['comp_consistency_threshold'] = 0.0
    GeomCA_parameters['comp_quality_threshold'] = 0.3
    R = np.concatenate([circle(n=num_pts, r=1.5), circle(n=int(num_pts/5), r=0.3)])
    E = np.concatenate([circle(n=num_pts, r=1.2), circle(n=int(num_pts/5), r=0.32)])
    return run_GeomCA_and_visualize(R, E, subfolder, GeomCA_parameters)

def example3_half_overlap(GeomCA_parameters):
    """
    Executes GeomCA on points from two half circles.
    :param GeomCA_parameters: dictionary containing GeomCA parameters.
    :return: GeomCA result dictionaries: components statistics, network statistics and network parameters
    """
    num_pts = 100
    GeomCA_parameters['experiment_filename_prefix'] = 'problematic_half_overlap_'
    subfolder = 'problematic_half_overlap'
    GeomCA_parameters['pr_comp_quality_threshold'] = 0.3
    GeomCA_parameters['pr_comp_consistency_threshold'] = 0.7
    R = np.concatenate([circle(n=num_pts, r=1.5), circle(n=int(num_pts/5), r=0.3)])
    E = np.concatenate([half_circle(n=int((2*num_pts)/3), r=1.4, noise=0.1), circle(n=int(num_pts/8), r=0.2, noise=0.1)])
    return run_GeomCA_and_visualize(R, E, subfolder, GeomCA_parameters)

def example4_many_components(GeomCA_parameters):
    """
    Executes GeomCA on points from two circles with many components.
    :param GeomCA_parameters: dictionary containing GeomCA parameters.
    :return: GeomCA result dictionaries: components statistics, network statistics and network parameters
    """
    num_pts = 100
    GeomCA_parameters['experiment_filename_prefix'] = 'many_components_'
    subfolder = 'many_components'
    GeomCA_parameters['Tdist_percentile'] = 1
    R = np.concatenate([circle(n=num_pts, r=1.6, noise=0.7), circle(n=int(num_pts/5), r=0.3)])
    E = np.concatenate([half_circle(n=int((2*num_pts)/3), r=1.4, noise=0.1), circle(n=int(num_pts/5), r=0.2, noise=0.1), 
                        ball(n=int(num_pts/5), center=[2, 4])])
    return run_GeomCA_and_visualize(R, E, subfolder, GeomCA_parameters)

if __name__ == '__main__':
    GeomCA_parameters = {
        'experiment_path': 'examples/toy_2Dcircle_data',      # Path to the experiment folder
        'experiment_filename_prefix': '',                     # Potential filename prefix to use
        'Rdist_ratio': 1.0,						# Percentage of R points to use for epsilon estimation
        'Rdist_percentile': 5,					# Percentile of R distances D determining epsilon estimate
        'gamma': 1,								# Portion of epsilon to use for sparsification: delta = gamma * epsilon(p)
        'reduceR': False,              			# Whether to reduce number of points in R
        'reduceE': False,              			# Whether to reduce number of points in E
        'sparsify': False,                		# Reduction type: sampling or sparsification
        'n_Rsamples': None, 					# Number of R samples if reducing by sampling
        'n_Esamples': None,						# Number of E samples if reducing by sampling
        'log_reduced': False,               	# Whether to save the reduced representations 
        'comp_consistency_threshold': 0.0,      # Component consistency threshold eta_c
        'comp_quality_threshold': 0.0,          # Component quality score eta_q
        'random_seed': 1201}    
    
    # Select an example to run
    GeomCA_results = example2_concatenated_RE(GeomCA_parameters)