import os
import numpy as np

def get_comp_points(component_stats, R_repr, R_paths, sparseR, 
                    E_repr, E_paths, sparseE, path):
    """
    Extracts images corresponding to representations in specific components.
    :param component_stats: dictionary of GeomCA component statistics.
    :param R_repr: initial R representations.
    :param R_paths: paths to R representations.
    :param sparseR: array of sparsified R representations.
    :param E_repr: initial E representations.
    :param E_paths: paths to E representations.
    :param sparseE: array of sparsified E representations.
    :param path: path to save the list of paths to the extracted images.
    """
    isolated_imgs = []
    for comp_id, v in component_stats.items():
        merged_imgs = []
        if len(v['Ridx']) == 0 and len(v['Eidx']) == 1:
            isolated_Erepr = sparseE[v['Eidx'].item()]
            is_equal = np.apply_along_axis(
                lambda Erepr: np.array_equal(Erepr, isolated_Erepr), 
                1, E_repr)
            path_to_image = E_paths[np.where(is_equal == True)[0].item()]
            isolated_imgs.append(path_to_image)
            
        elif len(v['Ridx']) > 0 and len(v['Eidx']) > 0:
            comp_Ereprs = sparseE[v['Eidx'], :]
            for comp_Erepr in comp_Ereprs:
                is_equal = np.apply_along_axis(
                    lambda t: np.array_equal(t, comp_Erepr), 1, E_repr)
                path_to_image = E_paths[np.where(is_equal == True)[0].item()]
                merged_imgs.append(path_to_image)
                
            comp_Rreprs = sparseR[v['Ridx'], :]
            for comp_Rrepr in comp_Rreprs:
                is_equal = np.apply_along_axis(
                    lambda t: np.array_equal(t, comp_Rrepr), 1, R_repr)
                path_to_image = R_paths[np.where(is_equal == True)[0].item()]
                merged_imgs.append(path_to_image)
            
            save_path = os.path.join(path, 'comp{}_paths.txt'.format(comp_id))
            with open(save_path, 'w') as f:
                f.writelines([path + '\n' for path in merged_imgs])
            
    save_path = os.path.join(path, 'paths_to_isolated_img.txt')
    with open(save_path, 'w') as f:
        f.writelines([path + '\n' for path in isolated_imgs])
        
