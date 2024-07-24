# Import the necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
torch.backends.cudnn.benchmark = True   # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
torch.set_num_threads(12)               # This parameter allows you to set the number of OpenMP threads used for parallelizing CPU operations
torch.set_num_interop_threads(12)       # This parameter allows you to set the number of OpenMP threads used for parallelizing CPU operations for interop

# Import the necessary modules from their respective libraries
from artattackwrapper import ARTAttackWrapper
from tqdm import tqdm

def free():
    ''' This function is used to free the GPU memory after the attack is completed '''
    # Free the GPU memory
    torch.cuda.empty_cache()
    print(f"GPU Memory freed: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

def target_image(target_class : int = 498, LABELS : np.ndarray = None):
    # Target Label
    targeted_label = target_class * np.ones(1)

    # One-hot encoding of the target label
    one_hot_targeted_label = tf.keras.utils.to_categorical(targeted_label, num_classes = len(LABELS))
    # Repeat the one-hot encoding for the batch size (1 -> 1000)
    batch_size = len(x_test_aligned)
    one_hot_targeted_label = np.tile(one_hot_targeted_label, (batch_size, 1))

    return one_hot_targeted_label

def load_labels() -> np.ndarray:
    ''' This function is used to load the labels of the VGGFace2 dataset
    
    Returns
    -------
    np.ndarray
        Labels of the VGGFace2 dataset
    '''
    # Download the labels of the VGGFace2 dataset
    fpath = tf.keras.utils.get_file(
        fname='rcmalli_vggface_labels_v2.npy',
        origin="https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy",
        cache_dir="./"
    )
    # Load the labels of the VGGFace2 dataset
    LABELS = np.load(fpath)
    return LABELS

def main(classifier : torch.nn.Module, x_test_aligned : torch.utils.data.DataLoader, cw_parameters : dict, binary_search_steps_range : list, y_true : torch.Tensor) -> None:
    ''' This function is used to generate adversarial examples using the Carlini attack.

    Parameters
    ----------
    classifier : torch.nn.Module
        The classifier model
    x_test_aligned : torch.utils.data.DataLoader
        The aligned test data
    cw_parameters : dict
        The parameters of the Carlini-Wagner attack
    binary_search_steps_range : list
        The epsilon range
    y_true : torch.Tensor
        The true labels of the test data
    '''
    # Get the aligned test data
    LABELS = load_labels()

    # Initialize the attack wrapper for the Carlini attack
    attacks = ARTAttackWrapper(classifier)

    # Error Generic
    # -------------

    # Initialize the list to store the number of correctly classified images for the Carlini attack
    nb_correct_nn1_g = list()
    perturbation_nn1_g = list()

    x_test_aligned = x_test_aligned.numpy() if isinstance(x_test_aligned, torch.Tensor) else x_test_aligned

    # Evaluate the model for each epsilon value
    for binary_search_steps in tqdm(binary_search_steps_range[1:], desc='Generating UnTargeted Examples with Carlini', leave=False):
        # Change the epsilon value into the Carlini parameters
        cw_parameters['binary_search_steps'] = binary_search_steps
        # Generate adversarial examples using the Carlini attack with the current epsilon value
        adv_images = []
        for image in tqdm(x_test_aligned, desc='Generating Adversarial Examples image-per-image', leave=False):
            adv_images.append(np.squeeze(attacks.carlini_attack(np.expand_dims(image, 0), **cw_parameters), 0))
        x_test_adv = np.array(adv_images) 
        # x_test_adv = attacks.carlini_attack(x_test_aligned, **cw_parameters)
        torch.save(x_test_adv, 'attacks/Carlini/Generic/x_test_adv_nn1_g_{0}.pt'.format(binary_search_steps))
        # Predict the labels of the adversarial examples for the current epsilon value
        x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)   
        # Get the predicted labels for the adversarial examples for the current epsilon value
        y_test_adv_cw = LABELS[x_test_adv_pred]
        # Count the number of correctly classified images for the current epsilon value
        nb_correct_nn1_g += [np.sum(np.array(y_true) == np.array(y_test_adv_cw))]
        print(f"\t • Number of correctly classified images for binary_search_steps={binary_search_steps}: {nb_correct_nn1_g[-1]}")
        # Save the perturbation for the current epsilon value
        perturbation_nn1_g += [np.mean(np.abs(x_test_adv - x_test_aligned))]
        print(f"\t • Perturbation for binary_search_steps={binary_search_steps}: {perturbation_nn1_g[-1]}")
        
        
    # Error Specific
    # --------------

    # Define the target class for the targeted attack
    cw_parameters['targeted'] = True
    target_class = 498
    one_hot_targeted_label = target_image(target_class, LABELS)

    # Initialize the list to store the number of correctly classified images for the Carlini attack
    nb_correct_nn1_s = list()
    perturbation_nn1_s = list()
        
    # Evaluate the model for each epsilon value
    for binary_search_steps in tqdm(binary_search_steps_range[1:], desc='Generating Targeted Examples with Carlini', leave=False):
        # Change the epsilon value into the Carlini parameters
        cw_parameters['binary_search_steps'] = binary_search_steps
        # Generate adversarial examples using the Carlini attack with the current epsilon value
        adv_images = []
        for image in tqdm(x_test_aligned, desc='Generating Adversarial Examples image-per-image', leave=False):
            adv_images.append(np.squeeze(attacks.carlini_attack(np.expand_dims(image, 0), one_hot_targeted_label=one_hot_targeted_label, **cw_parameters), 0))
        x_test_adv = np.array(adv_images) 
        torch.save(x_test_adv, 'attacks/Carlini/Specific/x_test_adv_nn1_s_{0}.pt'.format(binary_search_steps))
        # Predict the labels of the adversarial examples for the current epsilon value
        x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)   
        # Get the predicted labels for the adversarial examples for the current epsilon value
        y_test_adv_cw = LABELS[x_test_adv_pred]
        # Count the number of correctly classified images for the current epsilon value
        nb_correct_nn1_s += [np.sum(np.tile(LABELS[target_class], 1000) == np.array(y_test_adv_cw))]
        print(f"\t • Number of correctly classified images as Andrew Lincoln for binary_search_steps={binary_search_steps}: {nb_correct_nn1_s[-1]}")
        # Save the perturbation for the current epsilon value
        perturbation_nn1_s += [np.mean(np.abs(x_test_adv - x_test_aligned))]
        print(f"\t • Perturbation for binary_search_steps={binary_search_steps}: {perturbation_nn1_s[-1]}")
        
    # Save the number of correctly classified images for the Carlini attack
    torch.save(nb_correct_nn1_g, 'security_evaluation_curve/Carlini/nb_correct_nn1_g.pt')
    torch.save(perturbation_nn1_g, 'security_evaluation_curve/Carlini/perturbation_nn1_g.pt')
    torch.save(nb_correct_nn1_s, 'security_evaluation_curve/Carlini/nb_correct_nn1_s.pt')
    torch.save(perturbation_nn1_s, 'security_evaluation_curve/Carlini/perturbation_nn1_s.pt')

if __name__ == "__main__":
    # Nested function to collate the samples in the dataloader
    def collate_fn(x : list) -> tuple:
        ''' This function is used to collate the samples in the dataloader 
        
        Parameters
        ----------
        x : list
            List of samples
            
        Returns
        -------
        tuple
            Tuple containing the input and the target
            
        Example
        -------
        >>> collate_fn([(torch.tensor([1, 2, 3]), 0), (torch.tensor([4, 5, 6]), 1)])
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([0, 1]))
        '''
        return x[0]

    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Carlini Attack')
    parser.add_argument('--classifier_path', type=str, default='classifier.pt', help='Path to the classifier model')
    parser.add_argument('--dataloader_aligned_path', type=str, default='dataloader_aligned.pt', help='Path to the aligned dataloader')
    parser.add_argument('--cw_parameters', type=str, default='{"binary_search_steps": 10, "max_iter": 7, "confidence": 0.3, "learning_rate": 0.05, "initial_const": 0.01, "batch_size": 1, "verbose": false}', help='Carlini attack parameters')
    parser.add_argument('--binary_search_steps_range', type=str, default='[0, 2, 5, 10]', help='Epsilon range')
    parser.add_argument('--y_true_path', type=str, default='y_true.pt', help='Path to the true labels')

    args = parser.parse_args()

    classifier = torch.load(args.classifier_path)
    cw_parameters = json.loads(args.cw_parameters.replace("'", '"'))                                        # Fix JSON format if necessary
    dataloader_aligned = torch.load(args.dataloader_aligned_path)   
    binary_search_steps_range = json.loads(args.binary_search_steps_range.replace("'", '"'))                # Fix JSON format if necessary
    y_true = torch.load(args.y_true_path)
    
    # Convert binary_search_steps_range to float values
    binary_search_steps_range = [int(eps) for eps in binary_search_steps_range]
    
    # Get the x_test_aligned and y_test_aligned from the dataloader and convert them to tensors
    x_test_aligned, y_test_aligned = zip(*[(sample[0], sample[1]) for sample in dataloader_aligned])
    x_test_aligned = torch.stack(x_test_aligned)
    y_test_aligned = torch.tensor(y_test_aligned)
    
    # Call the free function to free the GPU memory
    free()
    
    # Call the main function
    main(classifier=classifier, x_test_aligned=x_test_aligned, cw_parameters=cw_parameters, binary_search_steps_range=binary_search_steps_range, y_true=y_true)
    
    # Call the free function to free the GPU memory
    free()