import vgg16_experiment_utils as vgg16_utils
import numpy as np 
import argparse
import os
import pickle
from GeomCA import GeomCA
from importlib.machinery import SourceFileLoader

parser = argparse.ArgumentParser()
parser.add_argument('--config_name' , type=str, required=True, 
                    help='name of the config file with specified parameters')
parser.add_argument('--load_representations' , type=int, default=1, 
                    help='load preobtained representations from VGG16')
parser.add_argument('--train' , type=int, default=1, 
                    help='Whether to run any of the algorithms')
parser.add_argument('--run_GeomCA' , type=int, default=0, 
                    help='runs geomCA algorithm')
parser.add_argument('--run_IPR' , type=int, default=0, 
                    help='run IPR algorithm')
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

Rclasses, Eclasses = config['data']['Rclasses'], config['data']['Eclasses']
repr_levels = config['data']['repr_levels']
path_to_dataset = config['data']['path_to_dataset']

# Load if representation file already exists else see vgg16_data_extraction.py
if args_opt.load_representations:    
    print('Loading VGG16 representations...')
    path_to_Rfeatures = os.path.join(path_to_dataset, 'Rfeatures.pkl') 
    if os.path.isfile(path_to_Rfeatures):
        with open(path_to_Rfeatures, 'rb') as f:
            Rdata = pickle.load(f)
            
    path_to_Efeatures = os.path.join(path_to_dataset, 'Efeatures.pkl') 
    if os.path.isfile(path_to_Efeatures):
        with open(path_to_Efeatures, 'rb') as f:
            Edata = pickle.load(f)
    print('VGG16 representations loaded.')

if args_opt.train:      
    for repr_level in repr_levels:    
        print('Current repr_level: {0}'.format(repr_level))
        subfolder = repr_level
        R = Rdata[repr_level]
        E = Edata[repr_level]
        
        # Run GeomCA
        if args_opt.run_GeomCA:    
            print('Starting to run GeomCa...')
            
            GeomCA_graph = GeomCA(R, E, config['GeomCA'], load_existing=False, subfolder=subfolder)
            GeomCA_results = GeomCA_graph.get_connected_components()
            component_stats, network_stats, network_params = GeomCA_results
            print('Done running GeomCA.')
            
            # Plot results
            GeomCA_graph.plot_connected_components()
            GeomCA_graph.plot_RE_components_consistency(min_comp_size=2, annotate_largest=True)
            GeomCA_graph.plot_RE_components_quality(min_comp_size=2, annotate_largest=True)
            print('Done visualizing GeomCA results.')
            
            save_folder = os.path.join(experiment_path, repr_level)
            vgg16_utils.get_comp_points(
                component_stats, Rdata[repr_level], Rdata['paths'], GeomCA_graph.sparseR,
                Edata[repr_level], Edata['paths'], GeomCA_graph.sparseE, save_folder)
        
        # Run improved precision and recall (IPR)
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
    
        # Run Geometry Score
        if args_opt.run_GS:
            if not 'GS' in config.keys():
                raise ValueError('GS config not found.')
            from benchmark_utils import run_GS
            
            print('Starting to run GS...')
            GS_subfolder = os.path.join(experiment_path, subfolder, 'GS_results') 
            
            geom_score = run_GS(R, E, os.path.join(GS_subfolder, 'original_imbalanced'), config['GS'])
            max_n_points = min(len(R), len(E))
            Rsubsampled = R[np.random.choice(np.arange(len(R)), size=max_n_points)]
            Esubsampled = E[np.random.choice(np.arange(len(E)), size=max_n_points)]
            assert(Rsubsampled.shape == Esubsampled.shape)
            print('Runing GS on original points of same shape: {0}'.format(Rsubsampled.shape))
            
            geom_score = run_GS(Rsubsampled, Esubsampled, 
                                      os.path.join(GS_subfolder, 'original_balanced'), config['GS'])

if args_opt.load_GeomCA:
    GeomCA_graph = GeomCA([], [], config['GeomCA'], load_existing=True, subfolder='feat_lin1')
    GeomCA_graph.plot_RE_components_quality(min_comp_size=5, annotate_largest=True, display_smaller=True)
    
    