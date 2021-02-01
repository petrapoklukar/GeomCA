config = {
    'data': {
        'truncations': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'path_to_dataset': 'datasets/stylegan',
        'num_samples': 5000
    },
    
    'GeomCA': {
        'experiment_path': 'experiments/stylegan/truncation_5k',
        'experiment_filename_prefix': 'perc30_gamma0p8_',
        'Rdist_ratio': 0.1,
        'Rdist_percentile': 30,
        'gamma': 0.8,
        'reduceR': True,              
        'reduceE': True,              
        'sparsify': True,                
        'n_Rsamples': None, 
        'n_Esamples': None,
        'log_reduced': True,               
        'comp_consistency_threshold': 0.0,      
        'comp_quality_threshold': 0.0,          
        'random_seed': 1201,    
    }
}