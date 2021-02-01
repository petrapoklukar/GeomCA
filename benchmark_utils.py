import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf 
import sys
sys.path.append('../geometry-score')
import gs 
sys.path.append('../improved-precision-and-recall-metric')
import precision_recall as ipr


def run_ipr(R, E, subfolder, config):
    """
    Runs Iproved Precision and Recall score.
    :param R: array of R points.
    :param E: array of E points.
    :param subfolder: subfolder to save results to.
    :param config: dictionary containing method hyperparameters.
    :return: precision and recall scores.
    """
    nhood_sizes = config['nhood_sizes'] 
    default = config['default']
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    sess = tf.Session()
    with sess.as_default():
        pr = ipr.knn_precision_recall_features(R, E, nhood_sizes=nhood_sizes,
                    row_batch_size=500, col_batch_size=100, num_gpus=1)
    print('IPR precision: {0}, recall: {1}'.format(pr['precision'], pr['recall']))
    
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    txt_file_path = os.path.join(subfolder, 'IPR_results.txt')
    pkl_file_path = os.path.join(subfolder, 'IPR_results.pkl')
    
    with open(txt_file_path, 'a') as f:
        f.writelines([
            '\n--- IPR stats\n',
            'nhood_sizes: {0}\n'.format(nhood_sizes)
        ])
        f.writelines(['{0}: {1}\n'.format(k, v) for k, v in pr.items()])
        print('IPR scores saved.')
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(pr, f)
    
    default_idx = nhood_sizes.index(default)
    return pr['precision'][default_idx], pr['recall'][default_idx]


def run_GS(R, E, subfolder, config):
    """
    Runs Geometry Score.
    :param R: array of R points.
    :param E: array of E points.
    :param subfolder: subfolder to save results to.
    :param config: dictionary containing method hyperparameters.
    :return: geometry score.
    """
    L_0 = config['L_0']
    gamma = config['gamma']
    i_max = config['i_max']
    n = config['n']
    
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    txt_file_path = os.path.join(subfolder, 'GS_results.txt')
    pkl_file_path = os.path.join(subfolder, 'GS_results.pkl')
    
    Rrlts = gs.rlts(R, L_0=L_0, gamma=gamma, i_max=i_max, n=n)
    Erlts = gs.rlts(E, L_0=L_0, gamma=gamma, i_max=i_max, n=n)
    
    R_color, E_color = 'C0', 'C1'
    mrltR = np.mean(Rrlts, axis=0)
    gs.fancy_plot(mrltR, label='MRLT of R', color=R_color)
    plt.legend()
    plt.savefig(os.path.join(subfolder, 'Rmrlt'))
    plt.clf()
    plt.close()
    
    mrltE = np.mean(Erlts, axis=0)
    gs.fancy_plot(mrltE, label='MRLT of E', color=E_color)
    plt.legend()
    plt.savefig(os.path.join(subfolder, 'Emrlt'))
    plt.clf()
    plt.close()
    
    gs.fancy_plot(mrltR, label='MRLT of R', color=R_color)
    gs.fancy_plot(mrltE, label='MRLT of E', color=E_color)
    plt.legend()
    plt.savefig(os.path.join(subfolder, 'REmrlt'))
    plt.clf()
    plt.close()
    
    GS_score = np.sum((mrltR - mrltE) ** 2)
    print('Obtained GS score: {}'.format(GS_score))
    with open(txt_file_path, 'a') as f:
        f.writelines([
            '\n--- GS stats\n',
            'GS score: {0}\n'.format(GS_score)])
        print('GS scores saved.')
        
    with open(pkl_file_path, 'wb') as f:
        pickle.dump({'GS_score': GS_score, 'gamma': gamma,
                    'mrltR': mrltR, 'mrltE': mrltE}, f)

    return GS_score