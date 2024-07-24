
import numpy as np
import pandas as pd
import torch
import torchvision.transforms

from torch.utils import data
from torchvision.transforms import transforms
from typing import Tuple, Union

class NN2Dataset(data.Dataset):

    # Mean BGR values from resnet50_ft.prototxt
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])                  
    
    # Defining the slots for the class attributes
    __slots__ = ['_images', '_transform', '_id_label_dict']
    
    # Constructor
    def __init__(
        self, 
        images : Union[np.ndarray, torch.Tensor],
        image_list_csv : str, 
        transform : bool = True
    ):  
        '''This function initializes the dataset with the images and image list csv file

        Parameters
        ----------
        images : Union[np.ndarray, torch.Tensor]
            The images to be used in the dataset (numpy array or torch tensor)
        image_list_csv : str
            The path to the image list csv file
        transform : bool, optional
            The flag to transform the images, by default True
            
        Notes
        -----
        The image list csv file should have the following columns:
        - id : The id of the image (e.g. 0, 1, 2, ...)
        - file_name : The name of the image file (e.g. datasets/TestSet/data/n000236/0001_01.jpg, etc.)
        - name : The name of the image (e.g. n000236, etc.)
        - label : The label of the image (e.g. 221, ...)
        '''
        # Save the images and transform flag
        self._images = images if isinstance(images, np.ndarray) else images.numpy()
        self._transform = transform
        # Read the image list csv file and create a dictionary of id and label
        image_list_csv = pd.read_csv(
            filepath_or_buffer='datasets/TestSet/image_list_file.csv',   # File path
            sep=';',                                            # Separator
            skipinitialspace=True,                              # Skip spaces after delimiter
            engine='python'                                     # Parser engine
        )
        self._id_label_dict = dict(zip(image_list_csv['id'], image_list_csv['label']))

    def __len__(self) -> int:
        '''This function returns the length of the dataset

        Returns
        -------
        int
            The length of the dataset
        '''
        return len(self._images)

    def __getitem__(self, index : int) -> Tuple[torch.Tensor, int]:
        '''This function returns the image and label associated with the index

        Parameters
        ----------
        index : int
            Index of the image in the dataset

        Returns
        -------
        Tuple[torch.Tensor, int]
            _description_
        '''
        img = self._images[index]
        img = transforms.ToPILImage()(((torch.tensor(img + 1.0) / 2.0)))
        img = torchvision.transforms.Resize(256)(img)
        img = torchvision.transforms.CenterCrop(224)(img)
        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3  # assumes color images and no alpha channel

        # Get the label associated with the image
        label = self._id_label_dict[index]
        if self._transform:
            return self.transform(img), label
        else:
            return img, label

    def transform(self, img : np.ndarray) -> torch.Tensor:
        '''This function transforms the image to the required format.

        Parameters
        ----------
        img : np.ndarray
            The image to be transformed

        Returns
        -------
        torch.Tensor
            The transformed image
        '''
        img = img[:, :, ::-1]                           # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)                    # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def untransform(self, img : torch.Tensor, lbl : int) -> Tuple[np.ndarray, int]:
        '''This function untransforms the image to the required format.

        Parameters
        ----------
        img : torch.Tensor
            The image to be untransformed
        lbl : int
            The label associated with the image

        Returns
        -------
        Tuple[np.ndarray, int]
            The untransformed image and label
        '''
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl