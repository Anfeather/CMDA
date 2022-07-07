import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
from PIL import Image
import copy
import cv2
class DatasetSplit(Dataset):
    """
    return the client split of datasets
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        img, caption, caplen = self.dataset[self.idxs[item]]
        return img.clone().detach(), caption.clone().detach(), caplen.clone().detach()


class HiddenLayerDataset(Dataset):
    def __init__(self, imgs, caps, caplens):
        self.hiddens = imgs
        self.caps = caps
        self.caplens = caplens

    def __getitem__(self, i):

        hid = torch.tensor(self.hiddens[i])
        cps = torch.LongTensor(self.caps[i])
        cpls = torch.LongTensor(self.caplens[i])

        return hid, cps, cpls

    def __len__(self):

        return len(self.hiddens)




class adjust_FederatedCaptionDataset(Dataset):

    def __init__(self, global_data):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """

        # Captions per image

        
        self.cpi = 5

        # Load encoded captions (completely into memory)
        self.captions, self.caplens, self.class_id = zip(*global_data)
        # print(self.caplens)
        # print(type(self.captions))
        # print(type(self.caplens))

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor(self.caplens[i])

        class_id = torch.LongTensor([self.class_id[i]])


        return caption, caplen, class_id
        

    def __len__(self):
        return self.dataset_size




class adjust_FederatedCaptionDataset_Second(Dataset):

    def __init__(self, global_data):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """

        # Captions per image

        
        self.cpi = 5

        # Load encoded captions (completely into memory)
        self.imgs, self.captions, self.caplens = zip(*global_data)
        # print(self.caplens)
        # print(type(self.captions))
        # print(type(self.caplens))

        # Total number of datapoints
        self.dataset_size = len(self.captions)
        

    def __getitem__(self, i):
        

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor(self.caplens[i])

        img = torch.tensor(self.imgs[i])


        return img.cuda(), caption.cuda(), caplen.cuda()
        

    def __len__(self):
        return self.dataset_size

    # def reconstruct(self):
    #     return zip(self.imgs, self.captions, self.caplens)








class FederatedCaptionDataset(Dataset):

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size





class non_iid_MSCOCODataset(Dataset):

    def __init__(self, data_folder, data_name, flag, user_dict, user_id,transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.caplens = []
        self.captions = []
        if flag == "train":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + '.json'), 'r') as j:
                    self.caplens.extend(json.load(j))
        elif flag == "test":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name+ "_test" + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "_test" + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + "_test" + '.json'), 'r') as j:
                    self.caplens.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.imgs[i//self.cpi] 
        # print("1", img.shape)
        if self.transform is not None:
            # print(type(img))
            img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
            img = self.transform(img)
        # print(len(self.captions[i]))
        caption =  torch.FloatTensor(self.captions[i])

        caplen = torch.FloatTensor([self.caplens[i]])


        return img, caption, caplen


    def __len__(self):
        return self.dataset_size


class non_iid_MSCOCODataset_image(Dataset):

    def __init__(self, data_folder, data_name, flag, user_dict, user_id,transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.caplens = []
        self.captions = []
        if flag == "train":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                # with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + '.json'), 'r') as j:
                #     self.captions.extend(json.load(j))

                # Load caption lengths (completely into memory)
                # with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + '.json'), 'r') as j:
                #     self.caplens.extend(json.load(j))
        elif flag == "test":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + "_test" + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])            
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.imgs)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        #img = torch.FloatTensor( self.imgs[i] / 255. )
        img = self.imgs[i] #/ 255.
        # print("1", img.shape)
        if self.transform is not None:
            # print(type(img))
            img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
            img = self.transform(img)
            
            # img = self.transform(img)
            
        return img


    def __len__(self):
        return self.dataset_size



def get_iid_split(dataset, num_users):
    """
        Sample I.I.D. client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def get_noniid_split(dataset, num_users):
    raise NotImplementedError()


def get_federated_dataset_train_and_valid(data_folder, data_name, transform, is_iid, num_users):
    train_dataset = FederatedCaptionDataset(data_folder, data_name, 'TRAIN', transform=transform)
    valid_dataset = FederatedCaptionDataset(data_folder, data_name, 'VAL', transform=transform)

    if is_iid:
        user_groups = get_iid_split(dataset=train_dataset, num_users=num_users)
    else:
        user_groups = get_noniid_split(dataset=train_dataset, num_users=num_users)

    return train_dataset, valid_dataset, user_groups


def get_small_federated_dataset_train_only(data_folder, data_name, transform, is_iid, num_users):
    train_dataset = FederatedCaptionDataset(data_folder, data_name, 'TRAIN', transform=transform)
    small_train_sample = np.random.choice(len(train_dataset), 40000)
    small_train_dataset = DatasetSplit(train_dataset, small_train_sample)

    if is_iid:
        user_groups = get_iid_split(dataset=small_train_dataset, num_users=num_users)
    else:
        user_groups = get_noniid_split(dataset=small_train_dataset, num_users=num_users)

    return small_train_dataset, small_train_dataset, user_groups

class TwoCropcaption:
    """Create two crops of the same image"""

    def __init__(self):
        self.transform = None

    def __call__(self, x):
        # print("2", self.transform(x).shape)
        return [x, x]



class non_iid_MSCOCODataset_text(Dataset):

    def __init__(self, data_folder, data_name, flag, user_dict, user_id,transform,transform_image):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.caplens = []
        self.captions = []
        if flag == "train":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # Load caption lengths (completely into memory)
                # with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + '.json'), 'r') as j:
                #     self.caplens.extend(json.load(j))
        elif flag == "test":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name+ "_test" + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "_test" + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))

                # Load caption lengths (completely into memory)
                # with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + "_test" + '.json'), 'r') as j:
                #     self.caplens.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.transform_image = transform_image
        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.imgs[i//self.cpi] 
        # print("1", img.shape)
        # if self.transform is not None:
        #     # print(type(img))
        img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
        img = self.transform_image(img)
        # print(len(self.captions[i]))
        caption = self.transform( torch.FloatTensor(self.captions[i]) )

        #caplen = torch.FloatTensor([self.caplens[i]])


        return caption, img


    def __len__(self):
        return self.dataset_size




class classification_MSCOCODataset(Dataset):

    def __init__(self, data_folder, data_name, flag, user_dict, user_id,transform):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        label_dict = {}
        for a,b in enumerate(data_distribution):
            label_dict[b] = a
        # Captions per image
        self.cpi = 5
        self.imgs = []
        self.labels = []
        self.captions = []

        if flag == "train":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))
               
                self.labels.extend([label_dict[i] for index in range(len(self.captions))])
                # Load caption lengths (completely into memory)
                # with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + '.json'), 'r') as j:
                #     self.caplens.extend(json.load(j))
        elif flag == "test":
            for i in data_distribution:
                data_train_path = data_folder + str(i)

                # Open hdf5 file where images are stored
                self.h = h5py.File(os.path.join(data_train_path,  '_IMAGES_' + data_name+ "_test" + '.hdf5'), 'r')
                self.imgs.extend(self.h['images'])

                # Load encoded captions (completely into memory)
                with open(os.path.join(data_train_path,  '_CAPTIONS_' + data_name + "_test" + "vector" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))
                self.labels.extend([label_dict[i] for index in range(len(self.captions))])
                # Load caption lengths (completely into memory)
                # with open(os.path.join(data_train_path,  '_CAPLENS_' + data_name + "_test" + '.json'), 'r') as j:
                #     self.caplens.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.imgs[i//self.cpi] 
        # print("1", img.shape)
        # if self.transform is not None:
        #     # print(type(img))
        img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
        img = self.transform(img)
        # print(len(self.captions[i]))
        caption = torch.FloatTensor(self.captions[i]) 
        label = torch.LongTensor([self.labels[i]])
        #caplen = torch.FloatTensor([self.caplens[i]])


        return caption, img, label


    def __len__(self):
        return self.dataset_size



class Flower_detection(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform=None,F="N"):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 10
        self.imgs = []
        self.captions = []
        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image"  + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = "/home/ray/preject/data/Flower_102/jpg/" + self.imgs[i][-15:] 

        # print("1", img.shape)
        if self.transform is not None:
            # print(type(img))
            # img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
            # img_raw = scipy.misc.imread( os.path.join(img_dir, name) )
            # img = tl.prepro.imresize(img_raw, size=[64, 64])    # (64, 64, 3)
            # img = img.astype(np.float32)
            img = cv2.imread(img)
            img = Image.fromarray(np.uint8(img))
            # img = self.transform_(img)
            
            # print(img.shape)
            img = self.transform(img)
        # print(len(self.captions[i]))
        caption =  torch.FloatTensor(self.captions[i][0])

        return caption, img


    def __len__(self):
        return self.dataset_size




class Flower_detection_text(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,transform_text,transform=None,F="N"):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        data_distribution = user_dict[user_id]
        # Captions per image
        self.cpi = 10
        self.imgs = []
        self.captions = []
        if F == "N":
            if flag == "train":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
            elif flag == "test":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        elif F == "OOD":
                for i in data_distribution:
                    data_train_path = data_folder + str(i)

                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image"  + '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))


                    # Load encoded captions (completely into memory)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        self.imgs.extend(json.load(j))

                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.transform_text = transform_text
        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # print("client data:", self.dataset_size)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = "/home/ray/preject/data/Flower_102/jpg/" + self.imgs[i][-15:] 

        # print("1", img.shape)
        if self.transform is not None:
            # print(type(img))
            # img = Image.fromarray(np.uint8(img.transpose(1,2,0)))
            # img_raw = scipy.misc.imread( os.path.join(img_dir, name) )
            # img = tl.prepro.imresize(img_raw, size=[64, 64])    # (64, 64, 3)
            # img = img.astype(np.float32)
            img = cv2.imread(img)
            img = Image.fromarray(np.uint8(img))
            # img = self.transform_(img)
            
            # print(img.shape)
            img = self.transform(img)
        # print(len(self.captions[i]))

        caption = self.transform_text( torch.FloatTensor(self.captions[i][0]) )     
        return caption, img


    def __len__(self):
        return self.dataset_size



