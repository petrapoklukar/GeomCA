import numpy as np 
import argparse
import os
import pickle
import multiview_box_utils as box_utils
from GeomCA import GeomCA
from importlib.machinery import SourceFileLoader

parser = argparse.ArgumentParser()
parser.add_argument('--config_name' , type=str, required=True, 
                    help='name of the config file with specified parameters')
parser.add_argument('--train' , type=int, default=1, 
                    help='Whether to run any of the algorithms')
parser.add_argument('--run_GeomCA' , type=int, default=1, 
                    help='runs geomCA algorithm')
parser.add_argument('--run_IPR' , type=int, default=0, 
                    help='run Improved Precision and Recall algorithm')
parser.add_argument('--run_GS' , type=int, default=0, 
                    help='run Gometry Score algorithm')
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

# Load representations
Rrepresentations, Rlabels = box_utils.load_formated_dataset(config['data']['path_to_train_dataset'])
Erepresentations, Elabels = box_utils.load_formated_dataset(config['data']['path_to_holdout_dataset'])
print('Data representations loaded.')

# Extract initial R and E splits
Eclasses = config['data']['Eclasses']
R = box_utils.get_representations_by_class(
    Rrepresentations, Rlabels, config['data']['Rclasses'])
E = box_utils.get_representations_by_class(
    Erepresentations, Elabels, Eclasses)

if args_opt.train:
    while len(Eclasses) <= config['data']['n_classes']:
        print('Current Eclasses: {0}'.format(Eclasses))
        subfolder='n_Eclasses{0}'.format(len(Eclasses))
        
        # Run GeomCA
        if args_opt.run_GeomCA:
            print('Starting to run GeomCa...')
            
            GeomCA_graph = GeomCA(R, E, config['GeomCA'], load_existing=False, subfolder=subfolder)
            GeomCA_results = GeomCA_graph.get_connected_components()
            component_stats, network_stats, network_params = GeomCA_results
            print('Done running GeomCA.')
                        
            # Plot results
            GeomCA_graph.plot_connected_components()
            GeomCA_graph.plot_RE_components_quality(min_comp_size=2, annotate_largest=True)
            GeomCA_graph.plot_RE_components_consistency(min_comp_size=2, annotate_largest=True)
            print('Done visualizing GeomCA results.')


        # Run Improved precision and recall (IPR)
        if args_opt.run_IPR:
            if not 'IPR' in config.keys():
                raise ValueError('IPR config not found.')
            from benchmark_utils import run_ipr 
            
            print('Starting to run IPR...')
            IPR_subfolder = os.path.join(experiment_path, subfolder, 'IPR_results')
            
            # IPR requires same amount of samples
            max_n_points = min(len(R), len(E))
            Rsubsampled = R[np.random.choice(np.arange(len(R)), size=max_n_points)]
            Esubsampled = E[np.random.choice(np.arange(len(E)), size=max_n_points)]
            assert(Rsubsampled.shape == Esubsampled.shape)
            print('Runing IPR on original points of shape: {0}'.format(Rsubsampled.shape))
            ip, ir = run_ipr(Rsubsampled, Esubsampled, IPR_subfolder, config['IPR'])
            
                    
        if args_opt.run_GS:
            if not 'GS' in config.keys():
                raise ValueError('GS config not found.')
            from benchmark_utils import run_GS
            
            print('Starting to run GS...')
            GS_subfolder = os.path.join(experiment_path, subfolder, 'GS_results') 
            
            # Run GS on both balanced and imbalanced sets
            print('Runing GS on original points, Rshape: {0}, Eshape: {1}'.format(
                R.shape, E.shape))
            geom_score = run_GS(R, E, os.path.join(GS_subfolder, 'original_imbalanced'), config['GS'])
            
            max_n_points = min(len(R), len(E))
            Rsubsampled = R[np.random.choice(np.arange(len(R)), size=max_n_points)]
            Esubsampled = E[np.random.choice(np.arange(len(E)), size=max_n_points)]
            assert(Rsubsampled.shape == Esubsampled.shape)
            print('Runing GS on original points of same shape: {0}'.format(Rsubsampled.shape))
            
            geom_score = run_GS(Rsubsampled, Esubsampled, 
                                os.path.join(GS_subfolder, 'original_balanced'), config['GS'])
            
        # Add a class to E and obtain the data
        new_class = Eclasses[-1] + 1
        new_class_representations = box_utils.get_representations_by_class(
            Erepresentations, Elabels, [new_class])
        Eclasses.append(Eclasses[-1] + 1)
        E = np.concatenate([E, new_class_representations])
        print('E set updated with class {0}'.format(new_class))

if args_opt.load_GeomCA:
    GeomCA_graph = GeomCA([], [], config['GeomCA'], load_existing=True, subfolder='n_Eclasses1')
    GeomCA_graph.plot_RE_components_quality(min_comp_size=5, annotate_largest=True)
    
    