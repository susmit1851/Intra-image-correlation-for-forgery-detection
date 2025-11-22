import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import SegformerModel, SegformerConfig
from torch.optim import AdamW
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import os
import random
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
from tqdm import tqdm