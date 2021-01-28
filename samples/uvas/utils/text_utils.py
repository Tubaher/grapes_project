import cv2

def draw_text_area(image, string, window_size, location = "left-top", margin = 5, padding = 5, alpha = 0.65,
  font = cv2.FONT_HERSHEY_SIMPLEX, thickness = 1, baseline = 2, rectangle_color = [255, 255, 255], text_color = [0,0,0]):
    print("location: ", location)
    window_w, window_h  = window_size
    text_w, text_h = cv2.getTextSize(string, font, thickness, baseline)[0]

    box_xmin = 0
    box_xmax = text_w + 2 * padding
    box_ymin = 0
    box_ymax = text_h + 2 * padding

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

    overlay = image.copy()
    cv2.rectangle(overlay, (box_xmin, box_ymin), (box_xmax, box_ymax), rectangle_color, -1)
    cv2.putText(overlay, string, (box_xmin + padding, box_ymax - padding), font, thickness, text_color, baseline)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


# # img = cv2.imread("picture.jpg")
# # h, w, c = img.shape
# # print(w,h)

# # # img = draw_text_area(img, "Hola", (w,h), location="left-top")
# # # img = draw_text_area(img, "Hola", (w,h), location="left-bottom")
# # # img = draw_text_area(img, "Hola", (w,h), location="rigth-bottom")
# # img = draw_text_area(img, "Hola", (w,h), location="rigth-top")

# # cv2.imwrite("test.jpg",img)
# # #cv2.imshow("window_name", img) 
  
# # #waits for user to press any key  
# # #(this is necessary to avoid Python kernel form crashing) 
# # #cv2.waitKey(0) 