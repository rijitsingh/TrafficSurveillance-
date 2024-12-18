from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(
    format="engine",
    dynamic=True, 
    batch=8, 
    workspace=4,
    half=True,
    int8=True,
    data="coco.yaml",
)

model_rt_op = YOLO("yolov8s.engine", task="detect") # load the model
# model_rt_op.info()
model_rt_op.predict('bus.jpg',show=True,save=True)

