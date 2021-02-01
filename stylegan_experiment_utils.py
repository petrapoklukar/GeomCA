import os
import numpy as np
import pickle

def create_splits(num_samples):
    """
    Sumbsamples original 50000 representations to create smaller splits. Used
    in time and sample robustness analysis.
    :params num_samples: number of samples to subsample. 
    """
    path_to_dataset = 'datasets/FFHQ_stylegan/stylegan_truncation'
    idxs = np.random.choice(50000, size=num_samples, replace=False)
    for truncation in np.arange(0.0, 1.01, 0.1):
        path = os.path.join(path_to_dataset, 
                            'stylegan_truncation{:.1f}_representations.pkl'.format(truncation))
        temp_dict = {}
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.items():
            if k in ['ref_features', 'eval_features']:
                temp_dict[k] = v[idxs]
            else: 
                temp_dict[k] = v
        temp_dict['idxs'] = idxs 
        path_to_save = os.path.join(
            path_to_dataset, 
            'stylegan_truncation{0:.1f}_{1}representations.pkl'.format(truncation, num_samples))
        assert(temp_dict['eval_features'].shape[0] == num_samples)
        assert(temp_dict['ref_features'].shape[0] == num_samples)
        with open(path_to_save, 'wb') as f:
            pickle.dump(temp_dict, f)
            

