import os

# general configuration
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets')

# model dirs
CURRICULARFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'CurricularFace_Backbone.pth')
ARCFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'arcface.pth')
COSFACE_MODEL_PATH = os.path.join(MODEL_DIR, 'cosface.pth')

# data dirs
LFW_GENERAL_DATA_DIR = os.path.join(DATASET_DIR, 'LFW')
LFW_UPDATE_GENERAL_DATA_DIR = os.path.join(DATASET_DIR, 'LFW')
YALEB_GENERAL_DATA_DIR = os.path.join(DATASET_DIR, 'YaleFaceCroppedB')