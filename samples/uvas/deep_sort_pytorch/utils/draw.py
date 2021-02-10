import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0), class_label=None):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(j) for j in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0  
        # color = compute_color_for_labels(id)
        color = (102, 0, 255) #pink
        label = '{}: {:d}'.format(class_label, id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 3 , 1)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,4)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 3, [255,255,255], 1)
    return img



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
