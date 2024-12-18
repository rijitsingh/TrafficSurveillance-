import csv

# Helper function to calculate the Intersection over Union (IoU) of two bounding boxes
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# Function to compare detections and save results in a CSV file
def compare_detections_with_iou_to_csv(csv_file, output_csv_file, iou_threshold=0.5):
    with open(csv_file, 'r') as file, open(output_csv_file, 'w', newline='') as out_file:
        reader = csv.DictReader(file)

        fieldnames = ['Video', 'Frame Index', 'GT Detections', 'TRT Detections', 
                      'Extra TRT Detections', 'Extra TRT Detection Locations', 'Extra TRT Class Labels', 
                      'Extra GT Detections', 'Extra GT Detection Locations', 'Extra GT Class Labels',
                      'Significant Overlap']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            frame_index = row.get('frame_index')
            video_name = row.get('video')
            gt_detections = int(row.get('gt_detections'))
            trt_detections = int(row.get('trt_detections'))
            gt_boxes = eval(row.get('gt_boxes'))
            trt_boxes = eval(row.get('trt_boxes'))
            gt_classes = eval(row.get('gt_classes'))
            trt_classes = eval(row.get('trt_classes'))

            # List to store non-overlapping extra boxes
            non_overlapping_trt_boxes = []
            non_overlapping_trt_classes = []
            non_overlapping_gt_boxes = []
            non_overlapping_gt_classes = []

            # Compare extra TRT detections (beyond GT)
            extra_trt_boxes = trt_boxes[gt_detections:]
            extra_trt_classes = trt_classes[gt_detections:]

            for i, extra_box in enumerate(extra_trt_boxes):
                is_overlapping = False
                for gt_box in gt_boxes:
                    iou = calculate_iou(extra_box, gt_box)
                    if iou >= iou_threshold:
                        is_overlapping = True
                        break

                if not is_overlapping:
                    non_overlapping_trt_boxes.append(extra_box)
                    non_overlapping_trt_classes.append(extra_trt_classes[i])

            # Compare extra GT detections (beyond TRT)
            extra_gt_boxes = gt_boxes[trt_detections:]
            extra_gt_classes = gt_classes[trt_detections:]

            for i, extra_box in enumerate(extra_gt_boxes):
                is_overlapping = False
                for trt_box in trt_boxes:
                    iou = calculate_iou(extra_box, trt_box)
                    if iou >= iou_threshold:
                        is_overlapping = True
                        break

                if not is_overlapping:
                    non_overlapping_gt_boxes.append(extra_box)
                    non_overlapping_gt_classes.append(extra_gt_classes[i])

            actual_trt_detection_difference = len(non_overlapping_trt_boxes)
            actual_gt_detection_difference = len(non_overlapping_gt_boxes)

            writer.writerow({
                'Video': video_name,
                'Frame Index': frame_index,
                'GT Detections': gt_detections,
                'TRT Detections': trt_detections,
                'Extra TRT Detections': actual_trt_detection_difference,
                'Extra TRT Detection Locations': str(non_overlapping_trt_boxes),
                'Extra TRT Class Labels': str(non_overlapping_trt_classes),
                'Extra GT Detections': actual_gt_detection_difference,
                'Extra GT Detection Locations': str(non_overlapping_gt_boxes),
                'Extra GT Class Labels': str(non_overlapping_gt_classes),
                'Significant Overlap': 'No' if (actual_trt_detection_difference > 0 or actual_gt_detection_difference > 0) else 'Yes'
            })

# Path to the CSV file generated in the previous step
csv_file_path = '/media/manvendra/Elements/project_traffic/extract_videos_1000/videos/aggrMp4/gamma_1.75/cam_5rain_/model_comparison_with_location_and_classes.csv'
# '/home/manvendra/Documents/yolo_tensorrt/extracted_videos/model_comparison_with_location_and_classes.csv'

# Output CSV file to save the comparison results with IoU analysis
output_csv_file_path = '/media/manvendra/Elements/project_traffic/extract_videos_1000/videos/aggrMp4/gamma_1.75/cam_5rain_/detection_comparison_iou_gt_vs_trt.csv'

# Compare the detections and check IoU with ground truth, save to CSV
compare_detections_with_iou_to_csv(csv_file_path, output_csv_file_path)


