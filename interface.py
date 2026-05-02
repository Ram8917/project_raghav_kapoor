# interface.py

from model import SimpleUNet as TheModel
from train import train_model as the_trainer
from predict import predict_temperature_classes as the_predictor
from dataset import ClimateDataset as TheDataset
from dataset import get_dataloader as the_dataloader
from config import batch_size as the_batch_size
from config import number_of_epochs as total_epochs