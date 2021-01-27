import glob
import argparse
import os
import shutil

"""
    This module helps to filter only the images 
    that have binary masks within a dataset 
"""

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
		description='Filter masked data from dataset')

    parser.add_argument('--data_dir', required=True,
                        metavar="/path/to/dataset",
                        help="Path to dataset")        

    args = parser.parse_args()
    
    data_dir = os.path.join(args.data_dir,"data")
    output_dir = os.path.join(args.data_dir,"data2")

    #Create paths if not exists
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    #Open the train_masked.txt 
    file_base_names = open(os.path.join(args.data_dir,"base_names.txt"), 'r') 
    for name in file_base_names.readlines():
      
      name = name.strip()
      extentions = [".jpg", ".npz"]

      for ext in extentions:
        # Copying files
        src = os.path.join(data_dir,name+ext)
        dst = os.path.join(output_dir,name+ext)
        shutil.copy2(src,dst)