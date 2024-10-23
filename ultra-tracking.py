import cv2
import os
from ultralytics import YOLO

model = YOLO()

# Create capture object and get fps
cap = cv2.VideoCapture("example_video.MOV")
fps = cap.get(cv2.CAP_PROP_FPS)

# Dictionary to keep track of saved object IDs -> conf
saved_objects = {}

# Create output directory for object images if it doesn't exist
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Frame skip is the fps divided by intended frame rate
frame_skip = int(fps / 30.0)
frame_count = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break
            
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device="mps")

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # box coordinates
        cls = int(box.cls[0])  # class id
        conf = float(box.conf[0])  # confidence score

        if box.id is None:
            print(f"Untracked object: Class {cls}, Confidence {conf:.2f}")
        else:
            track_id = int(box.id[0])
            if track_id and (track_id not in saved_objects or saved_objects[track_id] < conf):
                pad_x1, pad_y1, pad_x2, pad_y2 = 50, 50, 50, 50
                if x1 - pad_x1 < 0:
                    pad_x1 += x1 - pad_x1
                if x2 + pad_x2 > frame.shape[1]:
                    pad_x2 = frame.shape[1] - x2
                if y1 - pad_y1 < 0:
                    pad_y1 += y1 - pad_y1
                if y2 + pad_y2 > frame.shape[0]:
                    pad_y2 = frame.shape[0] - y2

                # Crop the detected object from the frame
                cropped_img = frame[(y1 - pad_y1):(y2 + pad_y2), (x1 - pad_x1):(x2 + pad_x2)]

                # Save the cropped image
                output_path = os.path.join(
                    output_dir, f'{results[0].names[cls]}_{track_id}.jpg')
                cv2.imwrite(output_path, cropped_img)

                # Mark this object ID as saved
                saved_objects[track_id] = conf
            print(
                f"Object ID: {track_id}, Class: {cls}, Confidence: {conf:.2f}")
            

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
