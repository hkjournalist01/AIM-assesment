import cv2
from ultralytics import YOLO
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='path of video', required=True)
args = parser.parse_args()

def main():
    model = YOLO('model/best.pt')
    
    video_path = args.video

    video = cv2.VideoCapture(video_path)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    while True:
        success, frame = video.read()
        if not success:
            break

        # results = model(frame, stream=True, iou=0.1, conf=0.45, classes=32)
        results = model(frame, stream=True, iou=0.5)
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 3)
                cv2.putText(frame, str(float(box.conf[0])), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 3)

        cv2.imshow('frame', frame)
        
        # Exit if q pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
