from ultralytics import YOLO

model = YOLO("runs/train_cars_20250422_090512/yolov8n/weights/best.pt")  

results = model.predict(
    source="./detect_img",  
    conf=0.25,         
    imgsz=640,         
    device=0,          
    show=True,         
    save=True          
)
