config = {
    'data': {
        'path_to_dataset': 'datasets/vgg16/version1', 
        'repr_levels': ['feat_lin1'], 
        'Rclasses': [530, 550, 567, 659, 827], # Kitchen stuff 
        'Eclasses': [174, 178, 182, 207, 214], # Dogs
        'n_classes': 5
    },
    
    'GeomCA': {
        'experiment_path': 'experiments/vgg16/version1',
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