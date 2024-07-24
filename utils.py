# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn

# Import the required packages from their parent modules.
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

def evaluate_model(model: nn.Module, dataloader: DataLoader, LABELS: np.ndarray) -> Tuple[list, list]:
    '''This function evaluates the model on the test set and returns the true and predicted labels.

    Parameters
    ----------
    model : nn.Module
        This is the model that we want to evaluate on the test set.
    dataloader : DataLoader
        This is the DataLoader object that contains all the test samples (images and labels) aligned by the MTCNN model.
    LABELS : np.ndarray
        This is the array that contains the class labels for the dataset.

    Returns
    -------
    Tuple[list, list]
        This function returns a tuple containing the true and predicted labels.
    '''
    # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)
        
    # Initialize the variables
    y_true = []
    y_pred = []
        
    # Iterate over the test set
    with torch.no_grad():
        for sample in tqdm(dataloader, total=len(dataloader), desc='Evaluating the model', leave=False):
            # Get image and label from the sample
            image, label = sample[0].to(device), sample[1].to(device)
            
            # Predicted output from the model
            image = torch.unsqueeze(image, 0) if len(image.shape) == 3 else image
            output = model(image).cpu()
            pred_class = int(torch.argmax(output, dim=1).item())
        
            # Append true and predicted labels to the lists
            y_true.append(dataloader.dataset.idx_to_class[label.item()])
            y_pred.append(LABELS[pred_class])          
            
    return y_true, y_pred

def plot_predicted_images(x_test : torch.Tensor, y_true : list, y_pred : list, title : str = "Plot Prediction", id_test_images : list = [1, 4, 32, 33, 59, 73], image_idx : int = 3):
    '''This function plots the predicted images with the true and predicted labels.

    Parameters
    ----------
    x_test : torch.Tensor
        This is the test image tensor.
    y_true : list
        This is the list containing the true labels.
    y_pred : list
        This is the list containing the predicted labels.
    id_test_images : list
        This is the list containing the image indices to plot.
        Default is [1, 4, 32, 33, 59, 73].
    image_idx : int
        This is the image index to plot.
        Default is 3.
    '''
    # Inner function to invert the standardization
    def invert_standardization(image : np.ndarray) -> np.ndarray:
        '''This function converts the image to range [0, 1] from [-1, 1].
        That is, it converts the pixels from the the range [-128, 128] to [0, 255]. 
        
        Parameters
        ----------
        image : torch.Tensor
            This is the image tensor that we want to invert the standardization.
        
        Returns
        -------
        torch.Tensor
            This function returns the image tensor with the standardization inverted.
        '''
        image = (image + 1) / 2 if image.min() < 0 else image
        return image

    # Convert x_test to numpy if it is a tensor
    x_test = x_test.numpy() if isinstance(x_test, torch.Tensor) else x_test

    # Create the figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21, 12))
    fig.suptitle(title, fontsize=21, color='black', backgroundcolor='#FFFF9b', fontweight='bold', fontname='Comic Sans MS')
    
    x_test = invert_standardization(x_test)

    for i, idx in enumerate(id_test_images):
        # Get the axes
        ax : plt.Axes = axes[i // 3, i % 3]
        # Get the image
        image = x_test[10 * (idx) + image_idx].transpose(1, 2, 0)
        np.clip(image, 0, 1, out=image)
        # Plot the image with the true and predicted labels
        ax.imshow(image)
        ax.set_title('True label: {0}\nPredicted label: {1}'.format(y_true[10*(idx) + image_idx], y_pred[10*(idx) + image_idx]))
        ax.axis('off')
        
def security_evaluation_curves(
        params_range: list, 
        nb_correct_generic: list, 
        nb_correct_specific: list, 
        perturbation_nn1_g: list, 
        perturbation_nn1_s: list, 
        title: str,
        x_axis_label: str = '$\epsilon$'
    ) -> None:
    '''This function plots the Security Evaluation Curves for the generic and specific cases.

    Parameters
    ----------
    params_range : list
        List containing the range of parameters for the analysis of the security evaluation curves.
    nb_correct_generic : list
        List containing the number of correct predictions for the generic case.
    nb_correct_specific : list
        List containing the number of correct predictions for the specific case.
    perturbation_nn1_g : list
        List containing the perturbation sizes for the generic case.
    perturbation_nn1_s : list
        List containing the perturbation sizes for the specific case.
    title : str
        Title of the plot.
    x_axis_label : str
        Label for the x-axis.
    '''
    # Plot the Security Evaluation Curves
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    fig.suptitle('Security Evaluation Curve: {0} Case'.format(title), fontsize=20, color='black', backgroundcolor='#FFFF9b', fontweight='bold')
    fig.tight_layout(pad=2.5, w_pad=6.0, h_pad=5.0)

    # Error Generic Case
    ax[0, 0].plot(np.array(params_range), np.array(nb_correct_generic) / 1000, color='navy', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier')
    ax[0, 0].grid(True, color='grey', linestyle=':', linewidth=0.5)
    ax[0, 0].set_title('Error Generic Case', fontsize=10)
    ax[0, 0].set_xlabel(x_axis_label, fontsize=10)
    ax[0, 0].set_ylabel('Accuracy', fontsize=10)
    ax[0, 0].legend(loc='best', shadow=False, fontsize='large', facecolor='lightgrey')
    ax[0, 0].set_facecolor('#dbe4ec')

    # Perturbation Generic Case
    ax[0, 1].plot(np.array(params_range), np.array(perturbation_nn1_g), color='navy', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier')
    ax[0, 1].grid(True, color='grey', linestyle=':', linewidth=0.5)
    ax[0, 1].set_title('Perturbation Generic Case', fontsize=10)
    ax[0, 1].set_xlabel(x_axis_label, fontsize=10)
    ax[0, 1].set_ylabel('Perturbation $\\ell_1$', fontsize=10)
    ax[0, 1].legend(loc='best', shadow=False, fontsize='large', facecolor='lightgrey')
    ax[0, 1].set_facecolor('#dbe4ec')

    # Check if the specific case is not None
    if not nb_correct_specific or not perturbation_nn1_s:
        # Delete the second row of the plot
        fig.delaxes(ax[1, 0])
        fig.delaxes(ax[1, 1])
        plt.show()
        return

    # Error Specific Case
    ax[1, 0].plot(np.array(params_range), np.array(nb_correct_specific) / 1000, color='darkred', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier')
    ax[1, 0].grid(True, color='grey', linestyle=':', linewidth=0.5)
    ax[1, 0].set_title('Error Specific Case', fontsize=10)
    ax[1, 0].set_xlabel(x_axis_label, fontsize=10)
    ax[1, 0].set_ylabel('Accuracy', fontsize=10)
    ax[1, 0].legend(loc='best', shadow=False, fontsize='large', facecolor='lightgrey')
    ax[1, 0].set_facecolor('#dbe4ec')

    # Perturbation Specific Case
    ax[1, 1].plot(np.array(params_range), np.array(perturbation_nn1_s), color='darkred', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier')
    ax[1, 1].grid(True, color='grey', linestyle=':', linewidth=0.5)
    ax[1, 1].set_title('Perturbation Specific Case', fontsize=10)
    ax[1, 1].set_xlabel(x_axis_label, fontsize=10)
    ax[1, 1].set_ylabel('Perturbation $\\ell_1$', fontsize=10)
    ax[1, 1].legend(loc='best', shadow=False, fontsize='large', facecolor='lightgrey')
    ax[1, 1].set_facecolor('#dbe4ec')

    plt.show()
    
def compare_sec(
        params_range: list,
        nn1_accuracy : float,
        nn2_accuracy : float,
        nb_correct_generic_nn1: list, 
        nb_correct_generic_nn2: list, 
        nb_correct_specific_nn1: list = None,
        nb_correct_specific_nn2: list = None, 
        title: str = "Comparison of Security Evaluation Curves for NN1 and NN2",
        x_axis_label: str = '$\epsilon$'
    ) -> None:
    '''This function plots the Security Evaluation Curves for the generic and specific cases for two different models (NN1 and NN2).

    Parameters
    ----------
    params_range : list
        This is the list containing the perturbation sizes.
    nn1_accuracy : float
        This is the accuracy of the NN1 model.
    nn2_accuracy : float
        This is the accuracy of the NN2 model.
    nb_correct_generic_nn1 : list
        This is the list containing the number of correct predictions for the generic case for NN1.
    nb_correct_generic_nn2 : list
        This is the list containing the number of correct predictions for the generic case for NN2.
    nb_correct_specific_nn1 : list
        This is the list containing the number of correct predictions for the specific case for NN1.
    nb_correct_specific_nn2 : list
        This is the list containing the number of correct predictions for the specific case for NN2.
    title : str
        Title of the plot
    '''
    cols = 3 if (nb_correct_specific_nn1 is not None or nb_correct_specific_nn2 is not None) else 2
    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(12, 7))
    fig.suptitle(title, fontsize=20, color='black', backgroundcolor='#FFFF9b', fontweight='bold')
    fig.tight_layout(pad=2.5, w_pad=6.0, h_pad=5.0)

    # Plot the accuracy of the models on a bar chart
    ax[0].plot([0, 0], [0, 100], color='#6bb56b', linestyle='-', linewidth=2)   
    ax[0].plot([0], nn1_accuracy * 100, 'navy', markersize=7, marker='o', markerfacecolor='white', markeredgewidth=2)
    ax[0].plot([0], nn2_accuracy * 100, 'darkred', markersize=7, marker='o', markerfacecolor='white', markeredgewidth=2)
    ax[0].annotate(f'NN1\n({nn1_accuracy * 100}, 0)', (0, nn1_accuracy * 100), textcoords="offset points", xytext=(-25, -9), ha='center', fontsize=9)
    ax[0].annotate(f'NN2\n({nn2_accuracy * 100}, 0)', (0, nn2_accuracy * 100), textcoords="offset points", xytext=(25, -9), ha='center', fontsize=9)
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(min(nn1_accuracy, nn2_accuracy) * 100 - 3, 100)    
    ax[0].grid(False)
    ax[0].get_xaxis().set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].set_facecolor('#eafff5')

    # Error Generic Case for NN1
    ax[1].plot(np.array(params_range), np.array(nb_correct_generic_nn1) / 1000, color='navy', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier')
    ax[1].plot(np.array(params_range), np.array(nb_correct_generic_nn2) / 1000, color='darkred', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN2 classifier')
    ax[1].grid(True, color='grey', linestyle=':', linewidth=0.5)
    ax[1].set_title('Error Generic Case', fontsize=10)
    ax[1].set_xlabel(x_axis_label, fontsize=10)
    ax[1].set_ylabel('Accuracy', fontsize=10)
    ax[1].legend(loc='best', shadow=False, fontsize='large', facecolor='lightgrey')
    ax[1].set_facecolor('#dbe4ec')

    if nb_correct_specific_nn1 is not None or nb_correct_specific_nn2 is not None:
        if nb_correct_specific_nn1:
            ax[2].plot(np.array(params_range), np.array(nb_correct_specific_nn1) / 1000, color='navy', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier')
        if nb_correct_specific_nn2:
            ax[2].plot(np.array(params_range), np.array(nb_correct_specific_nn2) / 1000, color='darkred', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN2 classifier')
        ax[2].grid(True, color='grey', linestyle=':', linewidth=0.5)
        ax[2].set_title('Error Specific Case', fontsize=10)
        ax[2].set_xlabel(x_axis_label, fontsize=10)
        ax[2].set_ylabel('Accuracy', fontsize=10)
        ax[2].legend(loc='best', shadow=False, fontsize='large', facecolor='lightgrey')
        ax[2].set_facecolor('#dbe4ec')

    plt.show()
    
    
def compare_sec_smoothed(
        params_range: list,
        nb_correct_generic_nn1: list, 
        nb_correct_generic_nn1_smoothed: list, 
        nb_correct_specific_nn1: list = None,
        nb_correct_specific_nn1_smoothed: list = None, 
        title: str = "Comparison of Security Evaluation Curves for NN1 and NN2",
        x_axis_label: str = '$\epsilon$'
    ) -> None:
    '''This function plots the Security Evaluation Curves for the generic and specific cases in order to compare the smoothed and unsmoothed curves for the NN1 model.

    Parameters
    ----------
    params_range : list
        This is the list containing the perturbation sizes.
    nb_correct_generic_nn1 : list
        This is the list containing the number of correct predictions for the generic case for NN1.
    nb_correct_generic_nn1_smoothed : list
        This is the list containing the number of correct predictions for the generic case for NN1 (Smoothed).
    nb_correct_specific_nn1 : list
        This is the list containing the number of correct predictions for the specific case for NN1.
    nb_correct_specific_nn1_smoothed : list
        This is the list containing the number of correct predictions for the specific case for NN1 (Smoothed).
    title : str
        Title of the plot
    '''
    cols = 2 if (nb_correct_specific_nn1 is not None or nb_correct_specific_nn1_smoothed is not None) else 1
    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(12, 7))
    fig.suptitle(title, fontsize=20, color='black', backgroundcolor='#FFFF9b', fontweight='bold')
    fig.tight_layout(pad=2.5, w_pad=6.0, h_pad=5.0)

    # Convert ax to list if cols == 1
    ax = [ax] if cols == 1 else ax

    # Error Generic Case
    ax[0].plot(np.array(params_range), np.array(nb_correct_generic_nn1) / 1000, color='navy', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier')
    ax[0].plot(np.array(params_range), np.array(nb_correct_generic_nn1_smoothed) / 1000, color='darkred', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier (Smoothed)')
    ax[0].grid(True, color='grey', linestyle=':', linewidth=0.5)
    ax[0].set_title('Error Generic Case', fontsize=10)
    ax[0].set_xlabel(x_axis_label, fontsize=10)
    ax[0].set_ylabel('Accuracy', fontsize=10)
    ax[0].legend(loc='best', shadow=False, fontsize='large', facecolor='lightgrey')
    ax[0].set_facecolor('#dbe4ec')

    # Error Specific Case
    if nb_correct_specific_nn1 is not None or nb_correct_specific_nn1_smoothed is not None:
        if nb_correct_specific_nn1:
            ax[1].plot(np.array(params_range), np.array(nb_correct_specific_nn1) / 1000, color='navy', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier')
        if nb_correct_specific_nn1_smoothed:
            ax[1].plot(np.array(params_range), np.array(nb_correct_specific_nn1_smoothed) / 1000, color='darkred', linestyle='-', marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, linewidth=2, label='NN1 classifier (Smoothed)')
        ax[1].grid(True, color='grey', linestyle=':', linewidth=0.5)
        ax[1].set_title('Error Specific Case', fontsize=10)
        ax[1].set_xlabel(x_axis_label, fontsize=10)
        ax[1].set_ylabel('Accuracy', fontsize=10)
        ax[1].legend(loc='best', shadow=False, fontsize='large', facecolor='lightgrey')
        ax[1].set_facecolor('#dbe4ec')

    plt.show()