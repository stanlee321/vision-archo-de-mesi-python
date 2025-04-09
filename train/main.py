# YOLOv11 example inference script
import time
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np



class DrawerManager:
    def __init__(self, fontScale=0.5, thickness=1):
        self.label_annotator = sv.LabelAnnotator(
            text_scale=fontScale,
            text_thickness=thickness
        )
        
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.corner_annotator = sv.BoxCornerAnnotator()
        self.mask_annotator = sv.MaskAnnotator(opacity = 0.4)
        self.trace_annotator = sv.TraceAnnotator()
        
        self.halo_annotator = sv.HaloAnnotator()
        
        self.ellipse_annotator = sv.EllipseAnnotator()

        self.heat_map_annotator = sv.HeatMapAnnotator()

        
    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections, labels: list, segmentation: bool = False) -> np.ndarray:
        
        annotated_frame = self.trace_annotator.annotate(
            scene= frame.copy(), 
            detections=detections
        )

        annotated_frame = self.label_annotator.annotate(
            scene= annotated_frame, detections=detections, labels=labels,
        )

        annotated_frame = self.ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        

        return annotated_frame
    
    
# model = YOLO("yolo11n.pt")
model = YOLO("best_11n.pt")


drawer = DrawerManager()

# keypoints annotator landmark
smoother = sv.DetectionsSmoother()

tracker = sv.ByteTrack(
    frame_rate=20,
    track_activation_threshold=0.5
)

        
# pose annotator
# Use PointAnnotator for keypoints, not PolygonAnnotator

# Capture video from webcam
cap = cv2.VideoCapture("/Users/stanleysalvatierra/Desktop/2025/vision/vision_1/train/data/videos/new_1.mp4")

# Write video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Set annotator to show the frame
while True:
    time1 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    # resize frame
    
    frame = cv2.resize(frame, (640, 480))
    
    # Get prediction results (it's a list)
    results = model.predict(frame, save=False, device="mps") # Changed save=True to save=False as we handle annotation
    
    # # Convert the first result to supervision Detections
    detections = sv.Detections.from_ultralytics(results[0])

    # detections = keypoints.as_detections()
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)
    detections = detections.with_nmm(threshold=0.5)
        
    if len(detections.xyxy) != 0:
        # # Annotate the frame using supervision
        annotated_frame = drawer.annotate_frame(frame, detections, ["ball"]*len(detections.class_id))
    else:
        annotated_frame = frame
        
    # annotated_frame = results[0].plot(kpt_radius=2)

    time2 = time.time()
    fps = 1 / (time2 - time1)
    annotated_frame = cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Show the frame
    # Label the frame
    cv2.imshow("Frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()



