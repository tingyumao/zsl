import os
import random
import cv2

from model import *
from config import Config
from dataset import *

def train():
    
    config = Config()
    
    # Define the model
    model = ZSL(mode="training", config=config, model_dir="./model/")
    model.keras_model.summary()

    # Prepare the dataset
    if sys.platform == 'darwin':
        data_path = os.path.join(os.getcwd(), '../data/train_test_a/zsl_a_animals_train_20180321')
    else:
        data_path = os.path.join(os.getcwd(), '../cai_zsl/data/train_test_a/zsl_a_animals_train_20180321')
        
    train_dataset = CAIData(root_dir=data_path, mode="train")
    train_dataset.prepare()

    val_dataset = CAIData(root_dir=data_path, mode="validation")
    val_dataset.prepare()
    
    # Training
    model.train(train_dataset, val_dataset, 
            learning_rate=config.LEARNING_RATE, 
            epochs=15)
    
    
if __name__ == "__main__":
    train()
    
