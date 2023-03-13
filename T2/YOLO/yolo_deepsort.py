import cv2
from ultralytics import YOLO
from tracker import Tracker
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='path of video', required=True)
args = parser.parse_args()

def main():
    video_path = args.video
    video = cv2.VideoCapture(video_path)

    model = YOLO("model/best.pt")

    tracker = Tracker()

    csv_column = ['frame_number','x_center','y_center','x_size','y_size']
    csv_list = []
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("result_video.mp4", fourcc, 30, (1280, 720))
    frame_id = 0
    while True:
        success, frame = video.read()

        if not success:
            break
        # results = model(frame, stream=True, iou=0.1, conf=0.45, classes=32)
        results = model(frame, stream=True, iou=0.5, conf=0.6)

        for r in results:
            detections = []
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                detections.append([x1, y1, x2, y2, box.conf[0]])
            
            if detections:
                tracker.update(frame, detections)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 3)
                csv_list.append([frame_id, int((x1+x2)/2), int((y1+y2)/2), int(x2-x1), int(y2-y1)])
        
        frame_id += 1
        cv2.imshow('frame', frame)
        out.write(frame)
        # Exit if q pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pd_result = pd.DataFrame(columns=csv_column, data=csv_list)
    pd_result.to_csv('tracking_result.csv', index=False)

    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
