import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

interpreter = tf.lite.Interpreter(model_path="/home/anmol/Documents/change/tf_model/resnet18/resnet18_float16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
class_names = {0: "classA", 1: "classB"}
roi = None
selecting_roi = False
start_point = (0, 0)

def mouse_callback(event, x, y, flags, param):
    global roi, selecting_roi, start_point
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        selecting_roi = True
    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, start_point, (x, y), (255, 0, 0), 2)
        cv2.imshow("test Classification", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        selecting_roi = False
        roi = (min(start_point[0], x), min(start_point[1], y), abs(x - start_point[0]), abs(y - start_point[1]))

cv2.namedWindow("test Classification")
cv2.setMouseCallback("test Classification", mouse_callback)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    if roi is not None:
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        if w > 0 and h > 0:
            image = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((224, 224))
            image = np.array(image, dtype=np.float32)
            image = (image / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            image = np.expand_dims(image, axis=0).astype(np.float32)
            
            interpreter.set_tensor(input_details[0]["index"], image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])
            predicted_class = np.argmax(output_data[0])
            confidence = output_data[0][predicted_class]
            
            if confidence > 0.6:
                predicted_label = class_names[predicted_class]
                cv2.putText(frame, f"{predicted_label}: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("test Classification", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()