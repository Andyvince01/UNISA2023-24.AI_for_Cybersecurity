# Import the required packages.
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import shutil

import warnings
warnings.filterwarnings('ignore')

# Import the required packages from their parent modules.
from PIL import Image
from matplotlib import pyplot as plt
from random import random
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score

# Change the current working directory
# os.chdir(os.path.dirname(os.path.abspath('')))

# Import logging modules.
import logging

# Create a custom logger.
logger = logging.getLogger('AIC_Logger')

# Check if the logger has handlers already.
if not logger.handlers:
    # Set the logging level for the custom logger to INFO.
    logger.setLevel(logging.INFO)

    # Create a console handler.
    ch = logging.StreamHandler()

    # Create a formatter and set it for the handler.
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the custom logger.
    logger.addHandler(ch)

    # Disable propagation to the root logger.
    logger.propagate = False

# Log an info message using the custom logger.
logger.info('Logging is set up.')