import argparse
import os
import pickle
from GeomCA import GeomCA
from importlib.machinery import SourceFileLoader

parser = argparse.ArgumentParser()
parser.add_argument('--config_name' , type=str, required=True, 
                    help='name of the config file with specified parameters')
parser.add_argument('--run_GeomCA' , type=int, default=1, 
                    help='runs geomCA algorithm')
parser.add_argument('--load_GeomCA' , type=int, default=0, 
                    help='loads existing GeomCA directory with saved logs in experiment folder')
args_opt = parser.parse_args()

# Load the config file
config_file = os.path.join('configs', args_opt.config_name)
config = SourceFileLoader(args_opt.config_name, config_file).load_module().config 
experiment_path = config['GeomCA']['experiment_path']
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
    
print('Loaded config: {0}'.format(config_file))
print('Results saved to: {0}'.format(experiment_path))

if args_opt.run_GeomCA:
    print('Starting to run GeomCa...')

    with open(config['data']['path_to_dataset'], 'rb') as f:
        data = pickle.load(f)
        R = data['T'] # Don't worry, just old naming :)
        E = data['V']
    print('Data representations loaded, R shape {0}, E shape {1}.'.format(R.shape, E.shape))
    
    for epsilon in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: 
        config['GeomCA']['epsilon'] = epsilon
        print('Chosen epsilon: {0}'.format(epsilon))
        
        # Run GeomCA
        GeomCA_graph = GeomCA(R, E, config['GeomCA'], load_existing=False, 
            subfolder='epsilon{0:.2f}'.format(epsilon))
        GeomCA_results = GeomCA_graph.get_connected_components()
        component_stats, network_stats, network_params = GeomCA_results
        print('Done running GeomCA.')
        
        # Plot results
        GeomCA_graph.plot_connected_components()
        GeomCA_graph.plot_RE_components_consistency(min_comp_size=2, annotate_largest=True)
        GeomCA_graph.plot_RE_components_quality(min_comp_size=2, annotate_largest=True)
        print('Done visualizing GeomCA results.')

if args_opt.load_GeomCA:
    subfolder = os.path.join('epsilon{0:.2f}'.format(0.4))
    GeomCA_graph = GeomCA([], [], config['GeomCA'], load_existing=True, subfolder=subfolder)
    GeomCA_graph.plot_RE_components_quality(min_comp_size=5, annotate_largest=True, figsize=(10, 5))
    