import cv2
import numpy as np
from functools import wraps
from tqdm import tqdm

class ImageProcessor:
    '''Class to preprocess images using various image processing techniques.'''

    def __init__(self) -> None:
        '''Initialize the ImageProcessor class.'''
        super().__init__()
                
    def image_preprocessor(func: callable) -> callable:
        '''Decorator function to preprocess input images using a specified preprocessing function.

        Parameters
        ----------
        func : callable
            The preprocessing function used to preprocess the input image.

        Returns
        -------
        callable
            The wrapper function that preprocesses the input image using the specified preprocessing function.
            
        Notes
        -----
        Using the @wraps decorator ensures that the wrapper function retains the original function's metadata (e.g., name, docstring, etc.).
        In other words, the wrapper function behaves as if it were the original preprocessing function; that is, the signature, name, 
        and docstring of the original preprocessing function are preserved in the wrapper function. The 'func' parameter 
        is the original preprocessing function that preprocesses the input image.
                
        The input image is passed as an argument to the preprocessing function. It is not passed as an argument to the 'func' class. 
        When the @image_preprocessor decorator is used, the input image is automatically passed to the preprocessing function when
        preprocessing the image.
        
        Example
        -------
        >>> @image_preprocessor
        ... def bilateral_filter(self, image: np.ndarray, **kwargs) -> np.ndarray:
        ...     return cv2.bilateralFilter(image, **kwargs)
        '''
            
        @wraps(func)
        def wrapper(self, image: np.ndarray, **kwargs) -> np.ndarray:
            '''Function that preprocesses the input image using the specified preprocessing function.
            It is a wrapper function that calls the specified preprocessing function and preprocesses the input image using the preprocessing function.

            Parameters
            ----------
            image : np.ndarray
                The input image to be preprocessed.

            Returns
            -------
            np.ndarray
                The preprocessed image generated using the specified preprocessing function.
            '''
            # Convert image from the range [-1, 1] into the range [0, 1], if necessary
            image = (image + 1) / 2 if np.min(image) < 0 else image
            
            # Transpose the image from the format (C, H, W) to the CV2 format (H, W, C), if necessary
            image = image.transpose(1, 2, 0) if image.shape[0] == 3 else image
            
            # Convert the image to float32 format and in the range [0, 255]
            image = (image * 255).astype(np.float32) if func.__name__ != 'denoising' else (image * 255).astype(np.uint8)
            
            # Apply the preprocessing function to the input image
            preprocessed_image = func(self, image, **kwargs)
            
            # Convert the preprocessed image back to float32 format and in the range [0, 1]
            preprocessed_image = preprocessed_image.astype(np.float32) / 255 * 2 - 1
            
            # Transpose the preprocessed image back to the format (C, H, W), if necessary
            preprocessed_image = preprocessed_image.transpose(2, 0, 1) if preprocessed_image.shape[2] == 3 else preprocessed_image
            
            return preprocessed_image
        
        return wrapper    
    
    def apply_vectorized_function(self, images: np.ndarray, func: callable, debug : bool = True, **kwargs) -> np.ndarray:
        '''Apply a vectorized function to each image in the batch.

        Parameters
        ----------
        images : np.ndarray
            The batch of images to apply the function to.
        func : callable
            The function to apply to each image.
        kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        np.ndarray
            The batch of images with the function applied.
        '''
        # Check if the input images is 4D numpy array
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        # Debugging information about the function and arguments
        # if debug:
        #     logger.info(f'Function: {func.__name__}')
        #     logger.info(f'Arguments: {kwargs}')
        
        # Apply the function to each image in the batch
        processed_images = []
        for image in tqdm(images, desc='Applying Vectorized Function', leave=False):
            processed_image = func(image, **kwargs)
            processed_images.append(processed_image)
        
        return np.array(processed_images)
    
    @image_preprocessor
    def bilateral_filter(self, image: np.ndarray, **kwargs) -> np.ndarray:
        '''Apply bilateral filter to the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image to which to apply bilateral filter.

        Returns
        -------
        np.ndarray
            The image with bilateral filter applied.
        '''
        return cv2.bilateralFilter(image, **kwargs)

    @image_preprocessor
    def denoising(self, image: np.ndarray, **kwargs) -> np.ndarray:
        '''Apply denoising to the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image to which to apply denoising.

        Returns
        -------
        np.ndarray
            The image with denoising applied.
        '''
        return cv2.fastNlMeansDenoisingColored(image, **kwargs)

    @image_preprocessor
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, **kwargs) -> np.ndarray:
        '''Apply Gaussian blur to the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image to which to apply Gaussian blur.
        kernel_size : int, optional
            The size of the Gaussian kernel (default is 5).

        Returns
        -------
        np.ndarray
            The image with Gaussian blur applied.
        '''
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), **kwargs)

    @image_preprocessor
    def median_blur(self, image: np.ndarray, kernel_size: int = 5, **kwargs) -> np.ndarray:
        '''Apply median blur to the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image to which to apply median blur.
        kernel_size : int, optional
            The size of the median kernel (default is 5).

        Returns
        -------
        np.ndarray
            The image with median blur applied.
        '''
        return cv2.medianBlur(image, kernel_size, **kwargs)