import cv2
import numpy as np

def draw_text_area(image, string, window_size, y_offset = 0, location = "left-top", margin = 5, padding = 5, alpha = 0.65,
  font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, thickness = 1, baseline = 2, rectangle_color = [255, 255, 255], text_color = [0,0,0]):
    window_w, window_h  = window_size
    text_w, text_h = cv2.getTextSize(string, font, fontScale, thickness)[0]
    y_offset = text_h * y_offset + 2 * padding * y_offset + margin * y_offset if y_offset != 0 else 0

    #Start the bbox from the left-top corner
    box_xmin = 0
    box_xmax = text_w + 2 * padding
    box_ymin = 0  + y_offset
    box_ymax = text_h + 2 * padding + y_offset

    if location == "left-top":
      box_xmin += margin
      box_xmax += margin
      box_ymin += margin
      box_ymax += margin
    elif location == "left-bottom":
      box_xmin += margin
      box_xmax += margin
      box_ymin = window_h - margin - box_ymax
      box_ymax = window_h - margin

    elif location == "rigth-bottom":
      box_xmin = window_w - margin - box_xmax
      box_xmax = window_w - margin
      box_ymin = window_h - margin - box_ymax
      box_ymax = window_h - margin

    elif location == "rigth-top":
      box_xmin = window_w - margin - box_xmax
      box_xmax = window_w - margin
      box_ymin += margin
      box_ymax += margin
    
    # print("[INFO] Frame type: ", type(image))
    # print("[INFO] Frame dtype: ", image.dtype)
    # print("[INFO] Max values ", np.amax(image))
    # print("[INFO] Max values ", np.amin(image))
    overlay = image.copy()
    
    cv2.rectangle(overlay, (box_xmin, box_ymin), (box_xmax, box_ymax), rectangle_color, thickness = -1)
    cv2.putText(overlay, string, (box_xmin + padding, box_ymax - padding), font, fontScale, text_color, thickness, baseline)    
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image
