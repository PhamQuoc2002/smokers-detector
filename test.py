from ultralytics import YOLO
import cv2 as cv
import os

video = cv.VideoCapture('Actors That Make Cigarette Smoking Look Cool - Part 2.mp4')
model = YOLO('goatmodel.pt')

classes = ['person', 'smoking']
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
video_writer = cv.VideoWriter(os.path.join('data', 'output', 'output.mp4'), cv.VideoWriter_fourcc(*'mp4v'), fps = 30, frameSize = (height, width))

if not video_writer.isOpened():
    print("Error: Could not open VideoWriter.")
    exit()

while True:
    isTrue, frame = video.read()
    results = model(frame, stream = 'True')
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            clss = int(box.cls[0])
            cv.rectangle(frame, (x1, y1), (x2, y2), color = (0, 255, 0), thickness = 2)
            cv.putText(frame, str(classes[clss]),(x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow('Testing', frame)
    video_writer.write(frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



video.release()
video_writer.release()
cv.destroyAllWindows()
