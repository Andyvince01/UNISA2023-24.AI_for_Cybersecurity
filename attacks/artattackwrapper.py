# Import the required packages.
import numpy as np

# Import the required packages from their parent modules.
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod, FastGradientMethod, DeepFool, CarliniL2Method
from functools import wraps

def generate_adversarial_samples(attack_func : callable) -> callable:
    '''This function generates adversarial samples using the specified attack function and the provided target image.

    Parameters
    ----------
    attack_func : callable
        The attack function used to generate adversarial samples.

    Returns
    -------
    callable
        The wrapper function that generates adversarial samples using the specified attack function.
        
    Notes
    -----
    By using the @wraps decorator, the wrapper function retains the original function's metadata (e.g., name, docstring, etc.).
    In other words, the wrapper function behaves as if it were the original attack function (e.g., FGSM, PGD, etc.); that is, 
    the signature, name, and docstring of the original attack function are preserved in the wrapper function. In particular, 
    the 'attack_func' parameter is the original attack function (e.g., FGSM, PGD, etc.) that generates adversarial samples.
            
    The target image and one-hot encoded label are passed as arguments to the attack function. They are not passed as arguments
    to the 'attack_func' class. Given that the @generate_adversarial_samples decorator is used, the target image and one-hot 
    encoded label are automatically passed to the attack function (e.g., FGSM, PGD, etc.) when generating adversarial samples.
        
    The parameters (e.g., epsilon, etc.) are members of the kwargs dictionary. The decorator returns the wrapper function that
    generates adversarial samples using the specified attack function.
    
    Example
    -------
    >>> @generate_adversarial_samples
    ... def fgsm_attack(self, target_image : np.ndarray, one_hot_targeted_label : np.ndarray = None, **kwargs : dict) -> np.ndarray:
    ...     return FastGradientMethod(self.classifier, **kwargs)    
    '''

    @wraps(attack_func)
    def wrapper(self, target_image : np.ndarray, one_hot_targeted_label : np.ndarray = None, **kwargs : dict) -> np.ndarray:
        '''This function generates adversarial samples using the specified attack function and the provided target image.
        It is a wrapper function that calls the specified attack function and generates adversarial samples using the attack instance.

        Parameters
        ----------
        target_image : np.ndarray
            The target image for which to generate the adversarial sample.
        one_hot_targeted_label : np.ndarray, optional
            The target label (optional) one-hot encoded.

        Returns
        -------
        np.ndarray
            The adversarial samples generated using the specified attack function.
        '''
        # Generate the attack instance (example: PGD, FGSM, etc.)
        attack_instance = attack_func(self, target_image, one_hot_targeted_label, **kwargs)        
        # Generate the adversarial samples using the attack instance
        adv_samples = attack_instance.generate(target_image, one_hot_targeted_label)
        
        # Return callable function and adversarial samples
        return adv_samples
    
    return wrapper

class ARTAttackWrapper:
    '''The ARTAttackWrapper class is a wrapper class for the Adversarial Robustness Toolbox (ART) library.
    It provides a simple interface to generate adversarial samples using different attack methods available in ART.
    In particular, it supports the following attack methods: Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM),
    Projected Gradient Descent (PGD), DeepFool, and Carlini and Wagner L2 Attack.
    
    Attributes
    ----------
    classifier : object
        The ART classifier used to generate the attacks.

    Methods
    -------
    fgsm_attack(target_image, one_hot_targeted_label, **kwargs)
        Generates adversarial samples using the Fast Gradient Sign Method (FGSM).
    bim_attack(target_image, one_hot_targeted_label, **kwargs)
        Generates adversarial samples using the Basic Iterative Method (BIM).
    pgd_attack(target_image, one_hot_targeted_label, **kwargs)
        Generates adversarial samples using the Projected Gradient Descent (PGD) method.
    deepfool_attack(target_image, one_hot_targeted_label, **kwargs)
        Generates adversarial samples using the DeepFool method.
    carlini_attack(target_image, one_hot_targeted_label, **kwargs)
        Generates adversarial samples using the Carlini and Wagner L2 Attack.
    '''
    # Define the __slots__ attribute to restrict the creation of new attributes
    __slots__ = ['classifier']
    
    def __init__(self, classifier : object) -> None:
        ''' Initializes the ARTAttackWrapper class with the specified classifier.

        Parameters
        ----------
        classifier : object
            The ART classifier used to generate the attacks (e.g., PyTorchClassifier, TensorFlowClassifier, etc.).
        '''
        self.classifier = classifier

    @generate_adversarial_samples
    def fgsm_attack(self, target_image : np.ndarray, one_hot_targeted_label : np.ndarray = None, **kwargs : dict) -> np.ndarray:
        '''This function generates adversarial samples using the Fast Gradient Sign Method (FGSM).

        Parameters
        ----------
        target_image : np.ndarray
            The target image for which to generate the adversarial sample.
        one_hot_targeted_label : np.ndarray, optional
            The target label (optional) one-hot encoded.

        Returns
        -------
        np.ndarray
            The adversarial samples generated using the FGSM method.
        '''       
        return FastGradientMethod(self.classifier, **kwargs)

    @generate_adversarial_samples
    def bim_attack(self, target_image : np.ndarray, one_hot_targeted_label : np.ndarray = None, **kwargs : dict) -> np.ndarray:
        '''This function generates adversarial samples using the Basic Iterative Method (BIM).
        
        Parameters
        ----------
        target_image : np.ndarray
            The target image for which to generate the adversarial sample.
        one_hot_targeted_label : np.ndarray, optional
            The target label (optional) one-hot encoded.
        
        Returns
        -------
        np.ndarray
            The adversarial samples generated using the BIM method.
        '''
        return BasicIterativeMethod(self.classifier, **kwargs)

    @generate_adversarial_samples
    def pgd_attack(self, target_image : np.ndarray, one_hot_targeted_label : np.ndarray = None, **kwargs : dict) -> np.ndarray:
        '''This function generates adversarial samples using the Projected Gradient Descent (PGD) method.
        
        Parameters
        ----------
        target_image : np.ndarray
            The target image for which to generate the adversarial sample.
        one_hot_targeted_label : np.ndarray, optional
            The target label (optional) one-hot encoded.
        
        Returns
        -------
        np.ndarray
            The adversarial samples generated using the PGD method.
        '''
        return ProjectedGradientDescent(self.classifier, **kwargs)

    @generate_adversarial_samples
    def deepfool_attack(self, target_image : np.ndarray, one_hot_targeted_label : np.ndarray = None, **kwargs : dict) -> np.ndarray:
        '''This function generates adversarial samples using the DeepFool method.
        
        Parameters
        ----------
        target_image : np.ndarray
            The target image for which to generate the adversarial sample.
        one_hot_targeted_label : np.ndarray, optional
            The target label (optional) one-hot encoded.
        
        Returns
        -------
        np.ndarray
            The adversarial samples generated using the DeepFool method.
        '''
        return DeepFool(self.classifier, **kwargs)

    @generate_adversarial_samples
    def carlini_attack(self, target_image : np.ndarray, one_hot_targeted_label : np.ndarray = None, **kwargs : dict) -> np.ndarray:
        '''This function generates adversarial samples using the Carlini and Wagner L2 Attack.

        Parameters
        ----------
        target_image : np.ndarray
            The target image for which to generate the adversarial sample.
        one_hot_targeted_label : np.ndarray, optional
            The target label (optional) one-hot encoded.

        Returns
        -------
        np.ndarray
            The adversarial samples generated using the Carlini and Wagner L2 Attack.
        '''
        return CarliniL2Method(self.classifier, **kwargs)
    
if __name__ == '__main__':
    print('This module is not intended to be executed as a script. Please import this module in another script.')