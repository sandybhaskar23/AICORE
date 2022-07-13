import cv2
from keras.models import load_model
import numpy as np
##load the model from the teachable machine. 
model = load_model('keras_model.h5')
##use camera to capture the image
cap = cv2.VideoCapture(0)
##create container to hold data
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True: 
    ##take input from camera image
    ret, frame = cap.read()
    ###make sure image is scaled down - note same as container/array
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    ##display the image in the frame
    cv2.imshow('frame', frame)
    # Press q to close the window
    print(prediction) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()