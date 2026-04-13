
import os
import shutil
import re
from envs.config_SimPy import *
from envs.config_RL import *
from multiprocessing import Process, current_process
def DEFINE_FOLDER(folder_name):
    if os.path.exists(folder_name):
        # Only count directories matching Train_<int>, then pick max + 1.
        # This avoids skipped numbering when non-directory files exist.
        next_idx = 1
        for entry in os.listdir(folder_name):
            full_path = os.path.join(folder_name, entry)
            if not os.path.isdir(full_path):
                continue
            m = re.fullmatch(r"Train_(\d+)", entry)
            if m:
                next_idx = max(next_idx, int(m.group(1)) + 1)
        folder_name = os.path.join(folder_name, f"Train_{next_idx}")
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
    meta_root = os.path.join(parent_dir, "AIIS_META")
    PAR_FOLDER = os.path.join(meta_root, "Tensorboard_logs")
    # Define dir's path
    if RL_EXPERIMENT:
        TENSORFLOW_LOGS = DEFINE_FOLDER(PAR_FOLDER)
        # Saved Model
        SAVED_MODEL_PATH = DEFINE_FOLDER(os.path.join(meta_root, "Saved_model"))
        SAVE_MODEL = True
        HYPERPARAMETER_LOG = None
        CSV_LOG = None
    else:
        CSV_LOG = None
