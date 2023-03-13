# Task 2: Soccer ball tracking

## Dependencies

* numpy
* scikit-learn==0.22.1
* opencv-python
* tensorflow==2.11.0
* torch
* ultralytics

## Running OpenCV trackers
Under `opencv_trackers` directory:
```
python opencv_trackers --video ball_tracking_video.mp4 --tracker TRACKER_TYPE
```

## Running SiamRPN++ tracker

Under `pysot` directory:
```
python tools/demo.py --config experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml --snapshot experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth --video ../ball_tracking_video.mp4
```

The output video and csv file will be saved under the same directory.

## Running YOLOv8 DeepSORT
Under `YOLO` directory:
```
python yolo_deepsort.py --video ../ball_tracking_video.mp4
```

The output video and csv file will be saved under the same directory.