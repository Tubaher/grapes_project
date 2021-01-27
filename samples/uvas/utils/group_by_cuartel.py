import shutil
import glob
import argparse
import os 

campo_name = "cuartel"

if __name__ == '__main__':

	# Parse command line arguments
  parser = argparse.ArgumentParser(
  description='Train Mask R-CNN to detect grapes.')
  parser.add_argument('--locations_dir', required=True,
          metavar="/path/to/pickles.pkl",
          help="Location pickles")
                      
  args = parser.parse_args()

  print("Location dir:",args.locations_dir)
  #Open each pickle file and order by folder
  for dir_file in glob.glob(args.locations_dir+"*.pkl"):

      file_name = dir_file.split("/")[-1]
      _, campo_id, cuartel_id, _, _ = file_name.split("_")

      folder_name = campo_name + "_" + campo_id + "_" + cuartel_id
      folder_dir = os.path.join(args.locations_dir,folder_name)
      if not os.path.exists(folder_dir):
          os.makedirs(folder_dir)

      shutil.move(dir_file,os.path.join(folder_dir,file_name))