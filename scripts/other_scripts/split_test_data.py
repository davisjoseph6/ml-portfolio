#!/usr/bin/env python3

import os
import shutil
import random

def split_data(source_dir, dest_dir1, dest_dir2, split_ratio=0.5):
    if not os.path.exists(dest_dir1):
        os.makedirs(dest_dir1)
    if not os.path.exists(dest_dir2):
        os.makedirs(dest_dir2)
    
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding directories in dest_dir1 and dest_dir2
        relative_path = os.path.relpath(root, source_dir)
        if relative_path == ".":
            relative_path = ""
        for dir in dirs:
            os.makedirs(os.path.join(dest_dir1, relative_path, dir), exist_ok=True)
            os.makedirs(os.path.join(dest_dir2, relative_path, dir), exist_ok=True)
        
        # Split files
        for file in files:
            file_path = os.path.join(root, file)
            if random.random() < split_ratio:
                shutil.copy(file_path, os.path.join(dest_dir1, relative_path, file))
            else:
                shutil.copy(file_path, os.path.join(dest_dir2, relative_path, file))

if __name__ == "__main__":
    # Define paths
    supervised_test = "/home/davis/ml-portfolio/data/supervised/test/"
    unsupervised_test = "/home/davis/ml-portfolio/data/unsupervised/test/"
    
    # Destination directories
    supervised_test_dest1 = "/home/davis/ml-portfolio/data/supervised/test_split1/"
    supervised_test_dest2 = "/home/davis/ml-portfolio/data/supervised/test_split2/"
    
    unsupervised_test_dest1 = "/home/davis/ml-portfolio/data/unsupervised/test_split1/"
    unsupervised_test_dest2 = "/home/davis/ml-portfolio/data/unsupervised/test_split2/"
    
    # Split supervised test data
    split_data(supervised_test, supervised_test_dest1, supervised_test_dest2, split_ratio=0.5)
    
    # Split unsupervised test data
    split_data(unsupervised_test, unsupervised_test_dest1, unsupervised_test_dest2, split_ratio=0.5)
    
    print("Test data split completed.")

