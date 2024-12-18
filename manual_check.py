import os
import random
import cv2
from ultralytics import YOLO

# Define paths to the models and video directory
pt_model_path = '/home/manvendra/Documents/yolo_tensorrt/yolov8s.pt'
trt_model_path = '/home/manvendra/Documents/yolo_tensorrt/yolov8s.engine'
gt_model_path = '/home/manvendra/Documents/yolo_tensorrt/yolov8x.pt'
video_directory = '/media/manvendra/Elements/project_traffic/extract_videos_1000/videos/aggrMp4/gamma_1.75/cam_5rain_/'
output_directory = '/home/manvendra/Documents/yolo_tensorrt/manual_check/detections'

# Load models
model_pt = YOLO(pt_model_path)
model_trt = YOLO(trt_model_path, task="detect")
model_gt = YOLO(gt_model_path)

# Create output folders for PT, TRT, and GT models
for model_name in ['pt', 'trt', 'gt']:
    model_dir = os.path.join(output_directory, model_name)
    os.makedirs(model_dir, exist_ok=True)

# Pick 5 random videos from the video directory
video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]
selected_videos = random.sample(video_files, 5)

# Helper function to process a single frame with a given model
def detect_and_save_frame(frame, model, output_path):
    results = model.predict(frame)
    annotated_frame = results[0].plot()  # Assuming `plot` method exists to annotate frame
    cv2.imwrite(output_path, annotated_frame)

# Process each selected video
for video_name in selected_videos:
    video_path = os.path.join(video_directory, video_name)
    
    # Create subdirectories for each video in the model folders
    for model_name in ['pt', 'trt', 'gt']:
        video_output_dir = os.path.join(output_directory, model_name, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

    # Read the video
    cap = cv2.VideoCapture(video_path)
    frame_number = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Define paths to save the frame images with detections
        pt_output_path = os.path.join(output_directory, 'pt', video_name, f'frame_{frame_number}.jpg')
        trt_output_path = os.path.join(output_directory, 'trt', video_name, f'frame_{frame_number}.jpg')
        gt_output_path = os.path.join(output_directory, 'gt', video_name, f'frame_{frame_number}.jpg')

        # Run detection on the frame for each model and save the results
        detect_and_save_frame(frame, model_pt, pt_output_path)
        detect_and_save_frame(frame, model_trt, trt_output_path)
        detect_and_save_frame(frame, model_gt, gt_output_path)
        
        frame_number += 1
    
    cap.release()

print("Detections saved for 5 random videos.")
