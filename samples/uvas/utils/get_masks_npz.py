import json
import argparse
import numpy as np
import os
from skimage import io, draw
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
		description='Get npz mask from .json')

    parser.add_argument('--dataset_dir', required=True,
                        metavar="/path/to/dataset",
                        help="Path to dataset")        

    args = parser.parse_args()
        
    annotations = json.load(open(os.path.join(args.dataset_dir, "data/via_region_data.json")))
    annotations = list(annotations.values())  # don't need the dict keys

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # Storing the save name in .txt file
    base_names = []

    #Add images
    for a in annotations:
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. These are stores in the
        # shape_attributes (see json format above)
        # The if condition is needed to support VIA versions 1.x and 2.x.
        if type(a['regions']) is dict:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
        else:
            polygons = [r['shape_attributes'] for r in a['regions']]
        
        file_name = a['filename'].split(".")
        base_name = '.'.join(file_name[:-1])
        image_path = os.path.join(args.dataset_dir, "data/"+ a['filename'])
        image = io.imread(image_path)
        height, width = image.shape[:2]

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)
        for i, p in enumerate(polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        #Compress to npz file
        compress_path = os.path.join(args.dataset_dir, "data/" + base_name + ".npz")
        np.savez_compressed(compress_path, arr_0=mask)

        #Append basenames
        base_names.append(base_name)

    #Storing base_names in a .txt file
    with open(os.path.join(args.dataset_dir,'base_names.txt'), 'a') as file:
        for b_name in base_names:
            file.write(b_name +"\n")