
from pathlib import Path
import shutil
import random


TARGET_DIR = Path("App") / "examples"
local_dir = Path("eu-moths-dataset") / "images"

def copy_examples(directory_path):
    # Convert the directory path to a Path object
    dir_path = Path(directory_path)
    files = []

    # Iterate over all files in the directory and its subdirectories using glob pattern **/*
    for file in dir_path.glob('**/*.jpg'):
        files.append(file)
    
    for file in random.sample(files, k=10):
        shutil.copy(file, TARGET_DIR)
        

copy_examples(local_dir)

