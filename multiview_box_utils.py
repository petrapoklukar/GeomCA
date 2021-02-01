import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def load_formated_dataset(path_to_dataset):
    """
    Loads pkl file containing representations.
    :param path_to_dataset: path to pkl file to load.
    :return: representations and labels.
    """
    with open(path_to_dataset, 'rb') as f:
        tuple_dataset = pickle.load(f)
    return flatten_box_dataset(tuple_dataset)

def flatten_box_dataset(tuple_dataset):
    """
    Flattens the representations from the original tuple format.
    :param tuple_dataset: list of tuples containing representations.
    :return: flattened representations and their labels.
    """
    representations, labels = [], []
    for repr1, repr2, _, label1, label2 in tuple_dataset:
        representations += [repr1, repr2]
        labels += [label1, label2]
    return np.array(representations), make_nice_labels(labels)

def make_nice_labels(labels):
    """
    Creates labels in range 0-11.
    :param labels: list original labels.
    :return: array of labeles.
    """
    unique_labels = sorted(np.unique(labels, axis=0))
    nice_labels = [unique_labels.index(l) for l in labels]
    return np.array(nice_labels)
  
def t_snes_plot(representations, labels, model_name, dataset_view):
    """
    Visualizes the representations using TSNE projection.
    :param representations: array of representations to visualize.
    :param labels: array of their labels.
    :param model_name: name of the model that produced the representations.
    :param dataset_view: camera view of images from the dataset.
    """
    tsne = TSNE(n_components=2, random_state=0)
    target_ids = np.unique(labels, axis=0)
    repr_2D = tsne.fit_transform(representations) 
    colors = cm.rainbow(np.linspace(0, 1, len(target_ids)))
    for id in target_ids:
        idxs = np.where(labels == id)[0]
        plt.scatter(repr_2D[idxs, 0], repr_2D[idxs, 1], color=colors[id], label=id)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Model: {0}, Dataset view: {1}'.format(model_name, dataset_view))
    plt.show()
    plt.clf()
    plt.close()  

def get_representations_by_class(representations, labels, Rclasses, Eclasses=None):
    """
    Extracts representations of a particular class from a given array.
    :param representations: array of representations.
    :param Rclasses: list of R classes to extract.
    :param Eclasses: list of E classes to extract.
    :return: array of filtered representations.
    """
    R = []
    for Rc in Rclasses:
        idxs = np.where(labels == Rc)[0]
        R.append(representations[idxs])
        
    if Eclasses is not None:
        E = []
        for Ec in Eclasses:
            idxs = np.where(labels == Ec)[0]
            E.append(representations[idxs])
        return np.vstack(R), np.vstack(E)
    else:
        return np.vstack(R)

def get_representations_by_class_with_random_subsampling(representations, labels, Eclasses):
    """
    Extracts random number of representations of a particular class from a given array.
    :param representations: array of representations.
    :param labels: list of their labeles.
    :param Eclasses: list of E classes to extract.
    :return: array of filtered representations.
    """
    E = []
    for Ec in Eclasses:
        idxs = np.where(labels == Ec)[0]
        perc_to_add = np.random.choice(np.arange(0.01, 0.6, 0.05))
        n_class_samples = int(idxs.shape[0] * perc_to_add)
        idxs_subsampled = np.random.choice(idxs, n_class_samples, replace=False)
        E.append(representations[idxs_subsampled])
    return np.vstack(E)


def create_epsilon_splits(model1, model2, view):
    """
    Creates the sets R and V used in epsilon experiment.
    :param model1: name of the first model to compare.
    :param model2: name of the second model to compare.
    :param view: camera view of images from the dataset.
    """
    model1_R, model2_R = get_classes_per_splits_epsilon_experiment(model1, model2, view, 'train')
    model1_E, model2_E = get_classes_per_splits_epsilon_experiment(model1, model2, view, 'holdout')
        
    with open('datasets/multiview_box/epsilon_{0}_{1}.pkl'.format(model1, view), 'wb') as f:
        pickle.dump({'R': model1_R, 'E': model1_E}, f)
    
    with open('../datasets/multiview_box/epsilon_{0}_{1}.pkl'.format(model2, view), 'wb') as f:
        pickle.dump({'R': model2_R, 'E': model2_E}, f)    
    

def get_classes_per_splits_epsilon_experiment(model1, model2, view, split):
    """
    Extracts representatons of the same images for both models.
    :param model1: name of the first model to compare.
    :param model2: name of the second model to compare.
    :param view: camera view of images from the dataset.
    :param split: dataset split, either train or holdout.
    :return: array of filtered representations from model1 and model2, respectively.
    """
    data_path_model1 = 'datasets/multiview_box/{0}_{1}_{2}.pkl'.format(model1, view, split)
    data_path_model2 = 'datasets/multiview_box/{0}_{1}_{2}.pkl'.format(model2, view, split)
    representations1, labels1 = load_formated_dataset(data_path_model1)
    representations2, labels2 = load_formated_dataset(data_path_model2)
    assert(np.array_equal(labels1, labels2))
    class_d = idxs_by_class(labels1, labels2, np.arange(7), n=250)
    
    repr1, repr2 = [], []
    for c, v in class_d.items():
        repr1.append(representations1[v['sub_idxs']])
        repr2.append(representations2[v['sub_idxs']])
    model1_epsilon_repr = np.concatenate(repr1)
    model2_epsilon_repr = np.concatenate(repr2)
    return model1_epsilon_repr, model2_epsilon_repr

def idxs_by_class(labels, labels_check, classes, n=250):
    """
    Extracts indices of representations corresponding to given labels.
    :param labels: array of labels to filter.
    :param labels_check: array of labels from another model that should be the same.
    :param classes: list of classes to filter.
    :param n: number of samples to sample from representations of each class.
    :return: dictionary of original and subsampled indices.
    """
    idx_dict = {c: {} for c in classes}
    for c in classes:
        idxs = np.where(labels == c)[0]
        assert(np.array_equal(idxs, np.where(labels_check == c)[0]))
        sub_idxs = np.random.choice(idxs, size=n, replace=False)
        idx_dict[c]['idxs'] = idxs
        idx_dict[c]['sub_idxs'] = sub_idxs
    return idx_dict

