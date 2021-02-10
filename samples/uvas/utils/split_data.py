import os
import random
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
		description='Get npz mask from .json')

    parser.add_argument('--dataset_dir', required=True,
                        metavar="/path/to/dataset",
                        help="Path to dataset")

    parser.add_argument('--basenames', required=False,
                        default= None,
                        metavar="/path/to/base_names.txt",
                        help="Path to base_names file text")     

    args = parser.parse_args()

    if args.basenames is None:
        basenames_file = os.path.join(args.dataset_dir,"base_names.txt")
    else:
        basenames_file = args.basenames
        
    # Creating directories
    dirs = ["train","val"]

    for d in dirs:
        path = os.path.join(args.dataset_dir,d)
        if not os.path.exists(path):
            os.makedirs(path)

    # define the percentage for train, validation
    train_p = 0.9 # val_p = 1.0 - train_p
    
    # get the basenames in a list
    file1 = open(basenames_file, 'r') 
    content = file1.readlines()
    content = [x.strip() for x in content] 

    # random shuffle
    random.shuffle(content)

    # define the lengths and intervals (len[i],len[i+1])
    len_train = int(train_p * len(content))
    lens = [0,len_train, len(content)]
    print("Lens: ", lens)

    # split data
    for i in range(2):
        
        print("Content from {} to {}".format(lens[i],lens[i+1]))
        data = content[lens[i]:lens[i+1]]
        # moving data 
        for base_name in data:
            
            #moving images from data to train or val
            src = os.path.join(args.dataset_dir,"data/" + base_name + ".jpg")
            dst = os.path.join(args.dataset_dir, dirs[i] +"/" + base_name + ".jpg")
            shutil.copy2(src,dst)

            #moving npz from data to train or val 
            src = os.path.join(args.dataset_dir,"data/" + base_name + ".npz")
            dst = os.path.join(args.dataset_dir, dirs[i] +"/" + base_name + ".npz")
            shutil.copy2(src,dst) 

    print("Ending the split data")