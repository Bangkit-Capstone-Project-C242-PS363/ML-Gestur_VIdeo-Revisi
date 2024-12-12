import asyncio
import websockets
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow
import json
import base64

class SignLanguageServer:
    def __init__(self):
        self.model = load_model('sign_language_video_model.h5')
        self.target_width = 224
        self.target_height = 224
        self.buffer_size = 60
        self.buffer = []
        self.label_map = {
            'spoon': 0, 'theater': 1, 'repeat': 2, 'pet': 3, 'recognize': 4,
            'orange': 5, 'sausage': 6, 'mine': 7, 'invite': 8, 'goodbye': 9,
            'fail': 10, 'approve': 11, 'moon': 12, 'ticket': 13, 'bet': 14,
            'apartment': 15, 'bully': 16, 'gymnastics': 17, 'seldom': 18,
            'find': 19, 'rose': 20, 'punish': 21, 'bored': 22, 'individual': 23,
            'lemon': 24, 'pig': 25, 'south america': 26, 'classroom': 27,
            'broke': 28, 'fact': 29, 'network': 30, 'lady': 31, 'second': 32,
            'excited': 33, 'work': 34, 'sue': 35, 'statistics': 36, 'seem': 37,
            'cuba': 38, 'gift': 39, 'question': 40, 'engagement': 41,
            'inspect': 42, 'blend': 43, 'sad': 44, 'heavy': 45, 'sentence': 46,
            'weight': 47, 'center': 48, 'pumpkin': 49
        }
        self.reversed_label_map = {v: k for k, v in self.label_map.items()}

    def preprocess_frame(self, frame_data):
        jpg_original = base64.b64decode(frame_data.split(',')[1])
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
        
        frame_resized = cv2.resize(frame, (self.target_width, self.target_height))
        frame_normalized = frame_resized / 255.0
        return frame_normalized

    async def process_video(self, websocket, path):
        try:
            async for message in websocket:
                frame = self.preprocess_frame(message)
                self.buffer.append(frame)

                if len(self.buffer) == self.buffer_size:
                    input_data = np.expand_dims(self.buffer, axis=0)
                    predictions = self.model.predict(input_data)
                    pred_label = np.argmax(predictions)
                    
                    response = {
                        'prediction': self.reversed_label_map[pred_label],
                        'confidence': float(predictions[0][pred_label])
                    }
                    await websocket.send(json.dumps(response))
                    
                    self.buffer.pop(0)
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")

async def main():
    server = SignLanguageServer()
    async with websockets.serve(server.process_video, "localhost", 8080):
        print("WebSocket server started on ws://localhost:8080")
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(main())