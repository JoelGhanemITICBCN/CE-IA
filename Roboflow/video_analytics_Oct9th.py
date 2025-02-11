import cv2
import time
import numpy as np
import argparse
from pathlib import Path
import supervision as sv
from ultralytics import YOLO

path = Path('video4.mp4')
#Defining the area we want to study 
#Supervision docs (tools->PolygonZone)
'''

POLYGON = np.array([])
LINE_1_START = sv.Point(0,1)
LINE_1_END = sv.Point(0,1)
LINE_2_START = sv.Point(0,1)
LINE_2_END = sv.Point(0,1)
'''
print(path)
if path == Path('video3.mp4'):
    CLASSES = [0]
    POLYGON = np.array([
    [146.5523156089194,955.943396226415],
    [144.9056603773585,149.08233276157802],
    [1335.4373927958834,147.43567753001713],
    [1360.1372212692968,957.5900514579758]
    ],dtype=np.int32)
    LINE_1_START = sv.Point(78,706)
    LINE_1_END = sv.Point(1207,735)
    LINE_2_START = sv.Point(976,438)
    LINE_2_END = sv.Point(1333,436)

elif path == Path('video.mp4'):
    CLASSES = [2,3]
    POLYGON = np.array([
    [0.8164665523156089,495.4777015437393],
    [511.9245283018868,266.0506003430532],
    [662.9708404802744,270.1329331046312],
    [816.466552315609,534.6680960548886],
    [3.2658662092624358,530.5857632933105]
    ], dtype=np.int32)

    LINE_1_START = sv.Point(261,517)
    LINE_1_END = sv.Point(791,513)
    LINE_2_START = sv.Point(545,275)
    LINE_2_END = sv.Point(497,275)

elif path == Path('video4.mp4'):
    CLASSES = [0]
    POLYGON = np.array([
   [0,0.40222984562605874],
   [0,432.28902229845625],
   [761.6912521440823,433.5977701543739],
   [760.3825042881647,1.0566037735848923]
   ], dtype=np.int32)

    LINE_1_START = sv.Point(49,6)
    LINE_1_END = sv.Point(48,430)
    LINE_2_START = sv.Point(701,16)
    LINE_2_END = sv.Point(701,414)


LINE_1_ZONE = sv.LineZone(
        start=LINE_1_START,
        end = LINE_1_END,
        triggering_anchors=(sv.Position.BOTTOM_CENTER,)
        )

LINE_2_ZONE = sv.LineZone(
    start=LINE_2_START,
    end = LINE_2_END,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
    )


#Cuadro que te hace el recuento de que hay dentro y que no 
line_zone_annotator_multiclass = sv.LineZoneAnnotatorMulticlass(
        text_scale=0,
        text_thickness=0,
        table_margin=0
        )

#Defining the classes we want to study (car and motorcycle are 2 and 3)
#CLASSES = [0,1,2,3]
model = YOLO("yolov8m.pt")

#Tracker to see the id of the vehicles
tracker = sv.ByteTrack(minimum_consecutive_frames=3)


#Annotators in order to clearly visualize the dta we are fetching
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

#Tracer to see the path the objects have followed (Tracker id's are mandatory for tracer)
trace_annotator = sv.TraceAnnotator(trace_length=30)

line_zone_annotator = sv.LineZoneAnnotator(text_orient_to_line=False)
polygon_zone = sv.PolygonZone(polygon=POLYGON,triggering_anchors=(sv.Position.CENTER,))


'''
1. Iterate each frame
    2. Detect with Yolo
    3. Defining the region (mask) and filter
    4. Filter by class
    5. Track objects over the frames
'''
def main():
    frame_generator = sv.get_video_frames_generator(path)
    for i, frame in enumerate(frame_generator):
        result = model(frame, device="cuda",verbose=False,imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(result) #Comprueba si el elemento esta o no donde marcamos

        detections = detections[polygon_zone.trigger(detections)]
        detections = detections[np.isin(detections.class_id,CLASSES)]
        detections = tracker.update_with_detections(detections)
        print(detections)


        LINE_1_ZONE.trigger(detections=detections)
        LINE_2_ZONE.trigger(detections=detections)

        annotated_frame = frame.copy()
        
        ''' 
        labels = [
                f"{tracker_id}"
                for tracker_id 
                #Id's every object
                in detections.tracker_id
        ]
        Cuadro que muestra el trozo de video que estudio
        annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=POLYGON,
                color=sv.Color.GREEN,
                thickness=2
        )

        Cuadro que muestra el total
        annotated_frame = box_annotator.annotate(
               scene=annotated_frame,
               detections=detections
        )
        '''

        
        annotated_frame = label_annotator.annotate(
               scene=annotated_frame,
               detections=detections,
        )

        annotated_frame = trace_annotator.annotate(
               scene=annotated_frame,
               detections=detections
        )
        annotated_frame = line_zone_annotator.annotate(
               annotated_frame, line_counter=LINE_1_ZONE,
        )
        annotated_frame = line_zone_annotator_multiclass.annotate(
                annotated_frame, line_zones=[LINE_1_ZONE]
        )

        annotated_frame = line_zone_annotator.annotate(
               annotated_frame, line_counter=LINE_2_ZONE,
        )
        annotated_frame = line_zone_annotator_multiclass.annotate(
                annotated_frame, line_zones=[LINE_2_ZONE]
        )
        cv2.imshow("Processed Video",annotated_frame)
        if cv2.waitKey(1) and 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

main()
