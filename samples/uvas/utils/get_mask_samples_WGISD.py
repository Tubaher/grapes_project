import glob
import argparse
import os
import shutil

"""
    This module helps to filter only the images 
    that have binary masks within WGISD dataset 
"""

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
		description='Filter masked data from WGISD dataset')

    parser.add_argument('--data_dir', required=True,
                        metavar="/path/to/dataset",
                        help="Path to dataset")        

    args = parser.parse_args()
    
    data_dir = os.path.join(args.data_dir,"data")
    train_dir = os.path.join(args.data_dir,"train")
    val_dir = os.path.join(args.data_dir,"val")

    #Create paths if not exists
    if not os.path.exists(train_dir):
      os.makedirs(train_dir)
      
    if not os.path.exists(val_dir):
      os.makedirs(val_dir)      

    #Open the train_masked.txt 
    train_file = open(os.path.join(args.data_dir,"train_masked.txt"), 'r') 
    for name in train_file.readlines():
      
      name = name.strip()
      extentions = [".jpg", ".txt", ".npz"]

      for ext in extentions:
        # Copying files
        src = os.path.join(data_dir,name+ext)
        dst = os.path.join(train_dir,name+ext)
        shutil.copy2(src,dst)

    #Open the test_masked.txt 
    train_file = open(os.path.join(args.data_dir,"test_masked.txt"), 'r') 
    for name in train_file.readlines():
      
      name = name.strip()
      extentions = [".jpg", ".txt", ".npz"]

      for ext in extentions:
        # Copying files
        src = os.path.join(data_dir,name+ext)
        dst = os.path.join(val_dir,name+ext)
        shutil.copy2(src,dst)
