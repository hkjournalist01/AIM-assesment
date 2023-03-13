import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='path of video', required=True)
args = parser.parse_args()

def main():
    video_path = args.video
    video = cv2.VideoCapture(video_path)

    model = YOLO('model/best.pt')
    # model = YOLO('model/yolov8m.pt')

    # initial_target_box = [729, 238, 764, 339]
    initial_target_box = [881, 537, 988, 636]

    initial_box_state = xyxy_to_xywh(initial_target_box)
    initial_state = np.array([[initial_box_state[0], initial_box_state[1],
                            initial_box_state[2], initial_box_state[3], 0, 0]]).T  # [x,y,w,h,dx,dy]

    IOU_Threshold = 0.3

    # State transition matrix
    A = np.array([[1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])

    # Observation matrix
    H = np.eye(6)

    # Process noice covariance
    # In tracking, process noice can come from sudden acceleration, deceleration, turning, etc.
    Q = np.eye(6) * 0.1

    # measurement noise covariance
    R = np.eye(6) * 1

    # State covariance
    P = np.eye(6)

    X_posterior = np.array(initial_state)
    P_posterior = np.array(P)
    Z = np.array(initial_state)
    trace_list = []

    while (True):
        success, frame = video.read()

        if not success:
            break
        
        last_box_posterior = xywh_to_xyxy(X_posterior[0:4])
        plot_one_box(last_box_posterior, frame, color=(255, 255, 255), target=False)

        results = model(frame, stream=True, classes=0)
        max_iou = IOU_Threshold
        max_iou_matched = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)
                xyxy = np.array([x1,y1,x2,y2], dtype="float")
                plot_one_box(xyxy, frame)
                iou = cal_iou(xyxy, xywh_to_xyxy(X_posterior[0:4]))
                if iou > max_iou:
                    target_box = xyxy
                    max_iou = iou
                    max_iou_matched = True
        if max_iou_matched == True:
            plot_one_box(target_box, frame, target=True)
            xywh = xyxy_to_xywh(target_box)
            box_center = (int((target_box[0] + target_box[2]) // 2), int((target_box[1] + target_box[3]) // 2))
            trace_list = updata_trace_list(box_center, trace_list, 100)
            cv2.putText(frame, "Tracking", (int(target_box[0]), int(target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

            dx = xywh[0] - X_posterior[0]
            dy = xywh[1] - X_posterior[1]

            Z[0:4] = np.array([xywh]).T
            Z[4::] = np.array([dx, dy])

        if max_iou_matched:
            # Prior probability
            X_prior = np.dot(A, X_posterior)
            box_prior = xywh_to_xyxy(X_prior[0:4])
            # plot_one_box(box_prior, frame, color=(0, 0, 0), target=False)
            # State Covariance
            P_prior_1 = np.dot(A, P_posterior)
            P_prior = np.dot(P_prior_1, A.T) + Q
            # Kalman gain
            k1 = np.dot(P_prior, H.T)
            k2 = np.dot(np.dot(H, P_prior), H.T) + R
            K = np.dot(k1, np.linalg.inv(k2))
            # Posterior probability
            X_posterior_1 = Z - np.dot(H, X_prior)
            X_posterior = X_prior + np.dot(K, X_posterior_1)
            box_posterior = xywh_to_xyxy(X_posterior[0:4])
            # Update state covariance
            P_posterior_1 = np.eye(6) - np.dot(K, H)
            P_posterior = np.dot(P_posterior_1, P_prior)
        else:
            # IoU match failed, iterate directly
            X_posterior = np.dot(A, X_posterior)
            box_posterior = xywh_to_xyxy(X_posterior[0:4])
            box_center = (
            (int(box_posterior[0] + box_posterior[2]) // 2), int((box_posterior[1] + box_posterior[3]) // 2))
            trace_list = updata_trace_list(box_center, trace_list, 20)
            cv2.putText(frame, "Lost", (box_center[0], box_center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)

        draw_trace(frame, trace_list)


        cv2.putText(frame, "ALL BOXES(Green)", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, "TRACKED BOX(Red)", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Last frame best estimation(White)", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('track', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
