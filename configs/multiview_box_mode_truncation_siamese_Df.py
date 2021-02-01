model = 'siamese'
view = 'Df'

config = {
    'data': {
        'model': model, 
        'view': view, 
        'path_to_train_dataset': 'datasets/multiview_box/{0}_{1}_train.pkl'.format(model, view),
        'path_to_holdout_dataset': 'datasets/multiview_box/{0}_{1}_holdout.pkl'.format(model, view), 
        'Rclasses': [0, 1, 2, 3, 4, 5, 6], 
        'Eclasses': [0], 
        'n_classes': 12
    },
    
    'GeomCA': {
        'experiment_path': 'experiments/multiview_boxes/{0}/{1}/mode_truncation'.format(model, view),
        'experiment_filename_prefix': 'perc1_gamma0p5_',
        'Rdist_ratio': 0.8,
        'Rdist_percentile': 1,
        'gamma': 0.5, 
        'reduceR': True,  
        'reduceE': True,              
        'sparsify': True,                
        'n_Rsamples': None, 
        'n_Esamples': None,
        'log_reduced': True,               
        'comp_consistency_threshold': 0.75,
        'comp_quality_threshold': 0.45,    
        'random_seed': 1201,    
    },
    
    'IPR': {
        'nhood_sizes': [3], 
        'default': 3
    },
    
    'GS': {
        'L_0': 64, 
        'gamma': 1/128, 
        'i_max': 10, 
        'n': 1000
    }
}