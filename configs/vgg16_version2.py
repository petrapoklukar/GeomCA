#import numpy as np
#classes = np.random.choice(1000, size=10, replace=False)

config = {
    'data': {
        'path_to_dataset': 'datasets/vgg16/version2',
        'repr_levels': ['feat_lin1'], 
        'Rclasses': [118, 791, 318, 795, 406],
        'Eclasses': [120, 799, 297, 162, 618],
        'n_classes': 5
    },
    
    'GeomCA': {
        'experiment_path': 'experiments/vgg16/version2',
        'experiment_filename_prefix': 'perc10_gamma1_',
        'Rdist_ratio': 0.5,
        'Rdist_percentile': 10,
        'gamma': 1,
        'reduceR': True,             
        'reduceE': True,             
        'sparsify': True,            
        'n_Rsamples': None, 
        'n_Esamples': None,
        'log_reduced': True,               
        'comp_consistency_threshold': 0.0,
        'comp_quality_threshold': 0.0,  
        'random_seed': 1201,    
    },

    'IPR': {
        'nhood_sizes': [3], 
        'default': 3
    },
    
    'GS': {
        'L_0': 64, 
        'gamma': 1/128, 
        'i_max': 100, 
        'n': 1000
    }  
}