import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os
import pickle
import numpy as np

class CustomImageNetDataset(Dataset):
    """
    Custom ImageNet dataset returning also image paths for diagnosing outliers.
    """
    def __init__(self, split):
        # Set the path to the dataset location
        dataset_path = os.path.join('/local_storage/datasets/imagenet/', split)
        self.image_list = torchvision.datasets.ImageFolder(dataset_path).imgs
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])


    def __getitem__(self, index):
        img_path, label = self.image_list[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, img_path

    def __len__(self):
        return len(self.image_list)


def get_imagenet_dataloader(split):
    """
    Creates a dataloader for VGG16.
    :param split: ImageNet dataset split to consider.
    :return: dataloder object.
    """
    imagenet_data = CustomImageNetDataset(split)
    dataloader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=1,
        shuffle=False,
        num_workers=0, 
        drop_last=False)
    return dataloader


def get_imagenet_representations_by_class(Rclasses, Eclasses, save_path):
    """
    Extracts VGG16 representations of ImageNet images of specific class.
    :param Rclasses: list of R classes.
    :param Eclasses: list of E classes.
    :param save_path: path to save the extracted representations to.
    :return: dictionaries containing extracted R and E representations, respectively.
    """
        
    def forward(img, label, img_path, feat_dict):
        """
        Forward pass through pretrained VGG16.
        :param img: batch of images.
        :param label: batch of labels.
        :param img_path: paths to processed images.
        :param feat_dict: dictionary to save extracted representations to.
        """
        feat = vgg16.features(img.cuda())
        feat1 = vgg16.avgpool(feat)
        feat1 = torch.flatten(feat1, 1)
        feat_dict['feat'].append(feat1.cpu().detach().numpy())
        for i in range(4):
            feat1 = vgg16.classifier[i](feat1)
            if i == 0:
                print(vgg16.classifier[i])
                feat_dict['feat_lin1'].append(feat1.cpu().detach().numpy())
            if i == 3:
                print(vgg16.classifier[i])
                feat_dict['feat_lin2'].append(feat1.cpu().detach().numpy())
        feat_dict['labels'].append(label.cpu().numpy().item())
        feat_dict['paths'].append(img_path[0])
     
    def save_representations(repr_dict, path):
        """
        Saves representation dictionary.
        :param repr_dict: dictionary containing representations.
        :param path: path to save the dictionary.
        """
        with open(path, 'wb') as f:
            pickle.dump(repr_dict, f)
    
    def concatenate_representations(repr_dict):
        """
        Flattens extracted representations.
        :param repr_dict: dictionary containing representations.
        :return: flattened dictionary.
        """
        for k, v in repr_dict.items():
            if k == 'labels':
                repr_dict[k] = np.array(v)
            elif k != 'paths':
                repr_dict[k] = np.concatenate(v)
        return repr_dict
    
    # Get representations
    Rfeatures = {'feat': [], 'feat_lin1': [], 'feat_lin2': [], 'labels': [], 'paths': []}
    Efeatures = {'feat': [], 'feat_lin1': [], 'feat_lin2': [], 'labels': [], 'paths': []}
    dataloader = get_imagenet_dataloader('train')
    vgg16 = models.vgg16(pretrained=True).cuda()
    for img, label, img_path in dataloader:
        print(img.shape)
        if label in Rclasses:
            forward(img, label, img_path, Rfeatures)
        if label in Eclasses:
            forward(img, label, img_path, Efeatures)
    R_dict = concatenate_representations(Rfeatures)
    E_dict = concatenate_representations(Efeatures)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_representations(R_dict, os.path.join(save_path, 'Tfeatures.pkl'))     
    save_representations(E_dict, os.path.join(save_path, 'Vfeatures.pkl'))        
    return R_dict, E_dict

def main():
    # Set classes and path to imagenet dataset
    get_imagenet_representations_by_class([], [], '')