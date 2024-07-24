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

def main(classifier : torch.nn.Module, x_test_aligned : torch.utils.data.DataLoader, df_parameters : dict, eps_range : list, y_true : torch.Tensor) -> None:
    ''' This function is used to generate adversarial examples using the DeepFool attack.

    Parameters
    ----------
    classifier : torch.nn.Module
        The classifier model
    x_test_aligned : torch.utils.data.DataLoader
        The aligned test data
    df_parameters : dict
        The parameters of the DeepFool attack
    eps_range : list
        The epsilon range
    y_true : torch.Tensor
        The true labels of the test data
    '''
    # Get the aligned test data
    LABELS = load_labels()

    # Initialize the attack wrapper for the DeepFool attack
    attacks = ARTAttackWrapper(classifier)

    # Initialize the list to store the number of correctly classified images for the DeepFool attack
    nb_correct_nn1_g = list()
    perturbation_nn1_g = list()
    
    # Convert the aligned test data to a numpy array
    x_test_aligned = x_test_aligned.numpy() if isinstance(x_test_aligned, torch.Tensor) else x_test_aligned

    # Evaluate the model for each epsilon value
    for eps in tqdm(eps_range[1:], desc='Generating UnTargeted Examples with DeepFool', leave=False, position=30):
        # Change the epsilon value into the DeepFool parameters
        df_parameters['epsilon'] = eps
        # Generate adversarial examples using the DeepFool attack with the current epsilon value
        adv_images = []
        for image in tqdm(x_test_aligned, desc='Generating Adversarial Examples image-per-image', leave=False):
            adv_images.append(np.squeeze(attacks.deepfool_attack(np.expand_dims(image, 0), **df_parameters), 0) )
        x_test_adv = np.array(adv_images)
        # x_test_adv = attacks.deepfool_attack(x_test_aligned, **df_parameters)
        torch.save(x_test_adv, 'attacks/DeepFool/Generic/x_test_adv_nn1_g_{0}.pt'.format(eps))
        # Predict the labels of the adversarial examples for the current epsilon value
        x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)   
        # Get the predicted labels for the adversarial examples for the current epsilon value
        y_test_adv_df = LABELS[x_test_adv_pred]
        # Count the number of correctly classified images for the current epsilon value
        nb_correct_nn1_g += [np.sum(np.array(y_true) == np.array(y_test_adv_df))]
        print(f"\t • Number of correctly classified images for epsilon={eps}: {nb_correct_nn1_g[-1]}")
        # Save the perturbation for the current epsilon value
        perturbation_nn1_g += [np.mean(np.abs(x_test_adv - x_test_aligned))]
        print(f"\t • Number of correctly classified images for epsilon={eps}: {perturbation_nn1_g[-1]}")
        
    # Save the number of correctly classified images for the DeepFool attack
    torch.save(nb_correct_nn1_g, 'security_evaluation_curve/DeepFool/nb_correct_nn1_g.pt')
    torch.save(perturbation_nn1_g, 'security_evaluation_curve/DeepFool/perturbation_nn1_g.pt')

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
    
    parser = argparse.ArgumentParser(description='DeepFool Attack')
    parser.add_argument('--classifier_path', type=str, default='classifier.pt', help='Path to the classifier model')
    parser.add_argument('--dataloader_aligned_path', type=str, default='dataloader_aligned.pt', help='Path to the aligned dataloader')
    parser.add_argument('--df_parameters', type=str, default='{"epsilon": 0.058823529411764705, "max_iter": 10, "nb_grads": 10, "batch_size": 16, "verbose": false}', help='DeepFool attack parameters')
    parser.add_argument('--eps_range', type=str, default='[0.0, 0.00392156862745098, 0.0392156862745098, 0.09803921568627451, 0.39215686274509803, 1.0]', help='Epsilon range')
    parser.add_argument('--y_true_path', type=str, default='y_true.pt', help='Path to the true labels')

    args = parser.parse_args()

    classifier = torch.load(args.classifier_path)
    dataloader_aligned = torch.load(args.dataloader_aligned_path) 
    df_parameters = json.loads(args.df_parameters.replace("'", '"'))                                        # Fix JSON format if necessary
    eps_range = json.loads(args.eps_range.replace("'", '"'))                                                # Fix JSON format if necessary
    y_true = torch.load(args.y_true_path)
    
    # Convert eps_range to float values
    eps_range = [float(eps) for eps in eps_range]
    
    # Get the x_test_aligned and y_test_aligned from the dataloader and convert them to tensors
    x_test_aligned, y_test_aligned = zip(*[(sample[0], sample[1]) for sample in dataloader_aligned])
    x_test_aligned = torch.stack(x_test_aligned)
    y_test_aligned = torch.tensor(y_test_aligned)
    
    # Call the free function to free the GPU memory
    free()
    
    # Call the main function
    main(classifier=classifier, x_test_aligned=x_test_aligned, df_parameters=df_parameters, eps_range=eps_range, y_true=y_true)
    
    # Call the free function to free the GPU memory
    free()