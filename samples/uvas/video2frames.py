import cv2
from tqdm import tqdm
import os.path
import numpy as np
video_path = 'videos/Prueba3_4K_30fps.MOV'
outFolder = 'frames/'
video = cv2.VideoCapture(video_path)               # argumento: path al video en el cual se harán las detecciones

fps = video.get(cv2.CAP_PROP_FPS)     
totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print('Number of frames =%d '%totalFrames)
duration =(totalFrames/fps)
minutes = int(duration/60)
seconds = int(duration%60)
#rint(type(video.get(cv2.CAP_PROP_POS_MSEC)))
print('Duration (M:S) = ' + str(minutes) + ':' + str(seconds))
count = 0
print('Writing video frames as jpg...')
np.random.seed(0)

start_frame  = 700
end_frame = totalFrames/2.
wanted_frames = 10
delta = int((end_frame - start_frame) / wanted_frames)
print("Saving every {} frames".format(str(delta)))
for frameCount in range(totalFrames):
    
    success, img = video.read()
    if frameCount < start_frame or not success: 
        continue
    if frameCount > end_frame:
        break
    if(frameCount%delta != 0):
        continue

    img = cv2.resize(img, (1280,720))
    file_name = outFolder+video_path.split('.')[0].split('/')[-1]+'_f'+str(frameCount)+'.jpg'
    if os.path.isfile(file_name):
        continue
    print(file_name)
    if('rotated' in video_path):
        img = np.rot90(img, k=3)
    cv2.imwrite(file_name,img) 
    count += 1
    if(count == wanted_frames):
        break
video.release()
print(count)