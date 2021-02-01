import numpy as np 
import argparse
import os
import multiview_box_utils as utils
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

    # Load representations
    Rrepresentations, Rlabels = utils.load_formated_dataset(config['data']['path_to_train_dataset'])
    Erepresentations, Elabels = utils.load_formated_dataset(config['data']['path_to_holdout_dataset'])
    print('Data representations loaded.')
    
    # Extract initial R and E splits
    Eclasses = config['data']['Eclasses']
    R = utils.get_representations_by_class(
        Rrepresentations, Rlabels, config['data']['Rclasses'])
    E = utils.get_representations_by_class_with_random_subsampling(
        Erepresentations, Elabels, Eclasses)
    
    while len(Eclasses) <= config['data']['n_classes']:
        print('Current Eclasses: {0}'.format(Eclasses))
        
        for comp_quality in [0.0, 0.1, 0.3]:
            print('Current component quality threshold: {0:.2f}'.format(
                config['GeomCA']['comp_quality_threshold']))
            
            config['GeomCA']['comp_quality_threshold'] = comp_quality
            
            for comp_consisency in [0, 0.4, 0.6]:
                print('Current component consistency threshold: {0:.2f}'.format(
                    config['GeomCA']['comp_consistency_threshold']))

                config['GeomCA']['comp_consistency_threshold'] = comp_consisency
            
                subfolder = os.path.join('comp_ct{0:.2f}_qt{1:.2f}'.format(comp_consisency, comp_quality),
                                        'n_Eclasses{0}'.format(len(Eclasses)))
                # Run GeomCA
                GeomCA_graph = GeomCA(R, E, config['GeomCA'], load_existing=False, 
                                    subfolder=subfolder)
                GeomCA_results = GeomCA_graph.get_connected_components()
                component_stats, network_stats, network_params = GeomCA_results
                print('Done running GeomCA.')
                
                # Plot results
                GeomCA_graph.plot_connected_components()
                GeomCA_graph.plot_RE_components_consistency(min_comp_size=2, annotate_largest=True)
                GeomCA_graph.plot_RE_components_quality(min_comp_size=2, annotate_largest=True)
                print('Done visualizing GeomCA results.')
                
        # Add a class to E and obtain the data
        new_class = Eclasses[-1] + 1
        new_class_representations = utils.get_representations_by_class_with_random_subsampling(
            Erepresentations, Elabels, [new_class])
        Eclasses.append(Eclasses[-1] + 1)
        E = np.concatenate([E, new_class_representations])
        print('E set updated with class {0}'.format(new_class))


if args_opt.load_GeomCA:
    subfolder = os.path.join('comp_ct{0:.2f}_qt{1:.2f}'.format(0.6, 0.3), 'n_Eclasses{0}'.format(2))
    GeomCA_graph = GeomCA([], [], config['GeomCA'], load_existing=True, subfolder=subfolder)
    GeomCA_graph.plot_RE_components_quality(min_comp_size=5, annotate_largest=True, figsize=(10, 5))
    