import cv2
import os
from ultralytics import YOLO
from flask import Flask, request, jsonify

app = Flask(__name__)

model = YOLO()

@app.route('/get_items_from_video', methods=['POST'])
def get_items_from_video():
    data = request.json
    print("Received JSON Data:", data)
    video_path = data.get("video")
    print("Video path:", video_path)

    if video_path is None:
        return jsonify({"error": "video path not provided."}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3333)

print("listening on 3333")
