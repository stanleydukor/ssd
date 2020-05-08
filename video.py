import cv2
from PIL import Image
import numpy as np
from detect import *
import imageio

method = "live"
currentFrame = 0
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.0.102:8080/video")

if method == "live":
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = detect(frame, min_score=0.5, max_overlap=0.1, top_k=200)
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        currentFrame += 1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

else:
    file_name = "test2"
    reader = imageio.get_reader('vid/'+file_name+'.mp4') # We open the video.
    fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
    writer = imageio.get_writer('output/'+file_name+'.mp4', fps = fps) # We create an output video with this same fps frequence.
    for i, frame in enumerate(reader): # We iterate on the frames of the output video:
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = detect(frame, min_score=0.5, max_overlap=0.1, top_k=200)
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.append_data(frame)
        print(i) # We print the number of the processed frame.
    writer.close() 