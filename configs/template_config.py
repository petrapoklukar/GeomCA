config = {
    'GeomCA': {
        'experiment_path': 'path/to/your/experiment',
        'experiment_filename_prefix': 'name_added_to_all_experiment_files_',
        'Rdist_ratio': 0.5,                     # Percentage of R points to use for epsilon estimation
        'Rdist_percentile': 10,                 # Percentile of R distances D determining epsilon estimate
        'gamma': 1,                             # Proportion of epsilon to use for sparsification: delta = gamma * epsilon(p)
        'reduceR': True,                        # Whether to sparsify R
        'reduceE': True,                        # Whether to sparsify E
        'sparsify': True,                       # Way to reduce number of representations: sampling or sparsification
        'n_Rsamples': None,                     # number of R points to sample if reducing by sampling
        'n_Esamples': None,                     # number of E points to sample if reducing by sampling
        'log_reduced': True,                    # Whether to save sparsified representations 
        'comp_consistency_threshold': 0.0,      # Component consistency threshold
        'comp_quality_threshold': 0.0,          # Component quality threshold
        'random_seed': 1201,                    # Random seed for reproducibility
    }, 

    'IPR': {
        'nhood_sizes': [3, 5, 10],              # Neighbourhood sizes to compute
        'default': 3                            # Default neighbourhood sizes to display results from
    },
    
    'GS': {                                     # Geometry Score hyperparameters, see Algorithm 1 in their
        'L_0': 64,                              # paper http://proceedings.mlr.press/v80/khrulkov18a.html
        'gamma': 1/128, 
        'i_max': 100, 
        'n': 1000
    }        

}