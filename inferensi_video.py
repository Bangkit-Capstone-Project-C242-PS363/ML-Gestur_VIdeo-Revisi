import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

label_map = {'spoon': 0, 'theater': 1, 'repeat': 2, 'pet': 3, 'recognize': 4, 'orange': 5, 'sausage': 6, 'mine': 7, 'invite': 8,
'goodbye': 9, 'fail': 10, 'approve': 11, 'moon': 12, 'ticket': 13, 'bet': 14, 'apartment': 15, 'bully': 16, 'gymnastics': 17,
'seldom': 18, 'find': 19, 'rose': 20, 'punish': 21, 'bored': 22, 'individual': 23, 'lemon': 24, 'pig': 25, 'south america': 26,
'classroom': 27, 'broke': 28, 'fact': 29, 'network': 30, 'lady': 31, 'second': 32, 'excited': 33, 'work': 34, 'sue': 35,
'statistics': 36, 'seem': 37, 'cuba': 38, 'gift': 39, 'question': 40, 'engagement': 41, 'inspect': 42, 'blend': 43, 'sad': 44,
'heavy': 45, 'sentence': 46, 'weight': 47, 'center': 48, 'pumpkin': 49}

reversed_label_map = {v: k for k, v in label_map.items()}

model = load_model('sign_language_video_model.h5')

target_width = 224
target_height = 224
buffer_size = 60
buffer = []
inference_delay = 0.5

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (target_width, target_height))
    frame_normalized = frame_resized / 255.0
    return frame_normalized

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat membuka kamera.")
    exit()

print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat menangkap frame.")
        break

    processed_frame = preprocess_frame(frame)
    buffer.append(processed_frame)

    if len(buffer) == buffer_size:
        input_data = np.expand_dims(buffer, axis=0)
        predictions = model.predict(input_data)
        pred_label = np.argmax(predictions)

        cv2.putText(frame, f"Prediksi: {reversed_label_map[pred_label]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        buffer.pop(0)

    cv2.imshow('Video Real-Time', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
