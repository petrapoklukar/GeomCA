config = {
    'data': {
        'truncations': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'path_to_dataset': 'datasets/stylegan',
        'num_samples': ''
    },
    
    'GeomCA': {
        'experiment_path': 'experiments/stylegan/truncation_50k',
        'experiment_filename_prefix': 'perc10_gamma1_',
        'Rdist_ratio': 0.1,
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
        'gamma': (1/128)/10, 
        'i_max': 10, 
        'n': 1000
    }
}