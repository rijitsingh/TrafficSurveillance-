import os
import csv
import cv2
from ultralytics import YOLO

# Define paths to the models
pt_model_path = '/home/manvendra/Documents/yolo_tensorrt/withoutcoco/yolov8s.pt'
trt_model_path = '/home/manvendra/Documents/yolo_tensorrt/yolov8s.engine'
ground_truth_model_path = '/home/manvendra/Documents/yolo_tensorrt/yolov8x.pt'

# Load models
model_pt = YOLO(pt_model_path)
model_trt = YOLO(trt_model_path, task="detect")
model_gt = YOLO(ground_truth_model_path)

# Helper function to calculate bounding box coordinates
def get_bounding_box_coordinates(image_width, image_height, x_center, y_center, width, height):
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    half_width = width * image_width / 2
    half_height = height * image_height / 2

    xmin = int(x_center_pixel - half_width)
    ymin = int(y_center_pixel - half_height)
    xmax = int(x_center_pixel + half_width)
    ymax = int(y_center_pixel + half_height)

    return xmin, ymin, xmax, ymax

# Helper function to detect objects in video using YOLOv8 and return class labels and box coordinates
def detect_objects_yolo(video_path, model):
    results = model.predict(source=video_path, task='detect', stream=True)
    return results

# Function to process video and add bounding box coordinates and class labels to CSV
def process_video(video_file, pt_model_size, trt_model_size, gt_model_size):
    results_list = []

    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for frame_index, result_gt in enumerate(detect_objects_yolo(video_file, model_gt)):
        # Ground truth results (YOLOv8x)
        gt_detections = len(result_gt.boxes)
        gt_boxes = [get_bounding_box_coordinates(frame_width, frame_height, box[0], box[1], box[2], box[3]) for box in result_gt.boxes.xywh]  # Extract boxes using xywh format
        gt_classes = [model_gt.names[int(cls)] for cls in result_gt.boxes.cls]  # Extract class labels

        # PyTorch YOLOv8s results
        result_pt = next(detect_objects_yolo(video_file, model_pt))
        pt_detections = len(result_pt.boxes)
        pt_boxes = [get_bounding_box_coordinates(frame_width, frame_height, box[0], box[1], box[2], box[3]) for box in result_pt.boxes.xywh]  # Extract boxes
        pt_classes = [model_pt.names[int(cls)] for cls in result_pt.boxes.cls]  # Extract class labels

        # TensorRT YOLOv8s results
        result_trt = next(detect_objects_yolo(video_file, model_trt))
        trt_detections = len(result_trt.boxes)
        trt_boxes = [get_bounding_box_coordinates(frame_width, frame_height, box[0], box[1], box[2], box[3]) for box in result_trt.boxes.xywh]  # Extract boxes
        trt_classes = [model_trt.names[int(cls)] for cls in result_trt.boxes.cls]  # Extract class labels

        # Append results to the list
        results_list.append({
            'video': os.path.basename(video_file),
            'frame_index': frame_index,
            'gt_detections': gt_detections,
            'gt_boxes': str(gt_boxes),
            'gt_classes': str(gt_classes),  # Add ground truth class labels
            'pt_detections': pt_detections,
            'pt_boxes': str(pt_boxes),
            'pt_classes': str(pt_classes),  # Add PyTorch class labels
            'trt_detections': trt_detections,
            'trt_boxes': str(trt_boxes),
            'trt_classes': str(trt_classes),  # Add TensorRT class labels
            'pt_model_size': pt_model_size,
            'trt_model_size': trt_model_size,
            'gt_model_size': gt_model_size
        })

    return results_list

# Function to compare YOLOv8 models and save results to CSV
def compare_yolo_models_in_folder(folder_path):
    csv_file = os.path.join(folder_path, 'model_comparison_with_location_and_classes.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['video', 'frame_index', 'gt_detections', 'gt_boxes', 'gt_classes', 'pt_detections', 'pt_boxes', 'pt_classes',
                      'trt_detections', 'trt_boxes', 'trt_classes', 'pt_model_size', 'trt_model_size', 'gt_model_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for video_file in os.listdir(folder_path):
            if video_file.endswith('.mp4'):
                full_video_path = os.path.join(folder_path, video_file)
                frame_results = process_video(full_video_path, pt_model_size, trt_model_size, gt_model_size)
                for result in frame_results:
                    writer.writerow(result)

# Get model sizes in MiB
pt_model_size = os.path.getsize(pt_model_path) / (1024 * 1024)
trt_model_size = os.path.getsize(trt_model_path) / (1024 * 1024)
gt_model_size = os.path.getsize(ground_truth_model_path) / (1024 * 1024)

# Specify the folder containing videos
folder_path = '/media/manvendra/Elements/project_traffic/extract_videos_1000/videos/aggrMp4/gamma_1.75/cam_5rain_'
# '/home/manvendra/Documents/yolo_tensorrt/extracted_videos'
compare_yolo_models_in_folder(folder_path)
