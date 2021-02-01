import numpy as np 
import argparse
import os
import pickle
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
                    help='run IPR algorithm')
parser.add_argument('--run_GS' , type=int, default=0, 
                    help='run Gometry Score algorithm')
parser.add_argument('--subsample' , type=int, default=0, 
                    help='Whether to subsample 50k array prior to training')
parser.add_argument('--num_samples' , type=int, default=10000, 
                    help='Number of sample if subsample is set to True')
parser.add_argument('--load_GeomCA' , type=int, default=0, 
                    help='loads existing GeomCA directory with saved logs in experiment folder')
args_opt = parser.parse_args()

# Load the config file
config_file = os.path.join('configs', args_opt.config_name)
config = SourceFileLoader(args_opt.config_name, config_file).load_module().config 
path_to_dataset = config['data']['path_to_dataset']
experiment_path = config['GeomCA']['experiment_path']
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
    
print('Loaded config: {0}'.format(config_file))
print('Results saved to: {0}'.format(experiment_path))

res_dict = {'GeomCAp': [], 'GeomCAr': [], 'GeomCAnc': [], 'GeomCAnq': [], 
            'IPRp': [], 'IPRr': [], 
            'GS': []}

if args_opt.train:
    for truncation in config['data']['truncations']:
        print('Truncation: {:.1f}'.format(truncation))
        subfolder = 'truncation{:.1f}'.format(truncation)
        
        # Load truncated representations
        path_to_representations = os.path.join(
            path_to_dataset, 
            'stylegan_truncation{0:.1f}_{1}representations.pkl'.format(
                truncation, config['data']['num_samples']))
        with open(path_to_representations, 'rb') as f:
            data = pickle.load(f)
            R = data['ref_features']
            E = data['eval_features']
            print('Data representations loaded from {0}'.format(path_to_representations))
        
        # Subsample T and V points prior to training if desired
        if args_opt.subsample:
            Ridxs = np.random.choice(np.arange(R.shape[0]), size=args_opt.num_samples, replace=False)
            R = R[Ridxs]
            Eidxs = np.random.choice(np.arange(E.shape[0]), size=args_opt.num_samples, replace=False)
            E = E[Eidxs]
            print('Subsampled R to: {0}, and E to: {1}'.format(R.shape, E.shape))
        
        # Run GeomCA
        if args_opt.run_GeomCA:
            print('Starting to run GeomCa...')
        
            GeomCA_graph = GeomCA(R, E, config['GeomCA'], load_existing=False, subfolder=subfolder)
            GeomCA_results = GeomCA_graph.get_connected_components()
            component_stats, network_stats, network_params = GeomCA_results
            print('Done running GeomCA.')
            
            # Plot results
            GeomCA_graph.plot_connected_components()
            GeomCA_graph.plot_RE_components_consistency(min_comp_size=5, annotate_largest=True)
            GeomCA_graph.plot_RE_components_quality(min_comp_size=5, annotate_largest=True)
            print('Done visualizing GeomCA results.')
            
            res_dict['GeomCAp'].append(GeomCA_graph.network_stats_logger['precision'])
            res_dict['GeomCAr'].append(GeomCA_graph.network_stats_logger['recall'])
            res_dict['GeomCAnc'].append(GeomCA_graph.network_stats_logger['network_consistency'])
            res_dict['GeomCAnq'].append(GeomCA_graph.network_stats_logger['network_quality'])
    
        # Run improved precision and recall (IPR)
        if args_opt.run_IPR:
            if not 'IPR' in config.keys():
                raise ValueError('IPR config not found.')
            from benchmark_utils import run_ipr 
            
            print('Starting to run IPR...')
            IPR_subfolder = os.path.join(experiment_path, subfolder, 'IPR_results')
        
            # Run IPR on all points
            assert(R.shape == E.shape)
            print('Runing IPR on original points of shape: {0}'.format(R.shape))
            ip, ir = run_ipr(R, E, IPR_subfolder, config['IPR'])
            res_dict['IPRp'].append(ip)
            res_dict['IPRr'].append(ir)

        # Run Geometry Scpre    
        if args_opt.run_GS:
            if not 'GS' in config.keys():
                raise ValueError('GS config not found.')
            from benchmark_utils import run_GS
            
            print('Starting to run GS...')
            GS_subfolder = os.path.join(experiment_path, subfolder, 'GS_results') 
            geom_score = run_GS(R, E, os.path.join(GS_subfolder, 'original'), config['GS'])
            print('Geometry score: {}'.format(geom_score))
            res_dict['GS'].append(geom_score)
            
    # Save global scores
    res_dict['truncations']  = config['data']['truncations']
    with open(experiment_path + 'aggregated_results.pkl', 'wb') as f:
        pickle.dump(res_dict, f)
         
if args_opt.load_GeomCA:
    GeomCA_graph = GeomCA([], [], config['GeomCA'], load_existing=True, subfolder = 'truncation{:.1f}'.format(0.3))
    GeomCA_graph.plot_RE_components_quality(min_comp_size=100, annotate_largest=True, display_smaller=True)
    