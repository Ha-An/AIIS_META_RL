
import os
import shutil
from envs.config_SimPy import *
from envs.config_RL import *
from multiprocessing import Process, current_process
def DEFINE_FOLDER(folder_name):
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        folder_name = os.path.join(folder_name, f"Train_{len(file_list)+1}")
        os.makedirs(folder_name)
    else:
        folder_name = os.path.join(folder_name, "Train_1")
        os.makedirs(folder_name)
    return folder_name


def save_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create a new folder
    os.makedirs(path)
    return path

if "MainProcess" == current_process().name:
    # Define parent dir's path
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    PAR_FOLDER = os.path.join(
            parent_dir, "Tensorboard_logs")
    # Define dir's path
    if RL_EXPERIMENT:
        TENSORFLOW_LOGS = DEFINE_FOLDER(PAR_FOLDER)
        # Saved Model
        SAVED_MODEL_PATH = DEFINE_FOLDER(os.path.join(parent_dir, "Saved_Model"))
        SAVE_MODEL = True
        HYPERPARAMETER_LOG = DEFINE_FOLDER(os.path.join(parent_dir, "Optuna_result"))
    
        CSV_LOG = DEFINE_FOLDER(os.path.join(
                parent_dir, "FEW_SHOT_RESULT"))
    else:
        CSV_LOG = DEFINE_FOLDER(os.path.join(
                parent_dir, "SSPOLICY_RESULT"))