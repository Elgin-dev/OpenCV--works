
import cv2
from pathlib import Path
#C:\\Users\\elgin\\Downloads\\annotated_counter_video (4).mp4
# Video path
video_path = "C:\\Users\\elgin\\Downloads\\annotated_counter_video (4).mp4"

# Open video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video started. Press 'q' to quit.")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("CCTV Video Viewer", frame)

    # Press q to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Video playback finished.")
