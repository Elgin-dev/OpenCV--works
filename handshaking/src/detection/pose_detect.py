
import cv2
import torch

from pathlib import Path
from ultralytics import YOLO


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_DIR
    / "dataset"
    / "merged"
    / "merged_dataset.mp4"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "pose"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "pose_output.mp4"


# =========================================================
# 2. MODEL CONFIGURATION
# =========================================================

MODEL_NAME = "yolov8n-pose.pt"

CONFIDENCE_THRESHOLD = 0.35

IMAGE_SIZE = 640

KEYPOINT_CONFIDENCE = 0.35


# COCO keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_ELBOW = 7
RIGHT_ELBOW = 8

LEFT_WRIST = 9
RIGHT_WRIST = 10


# =========================================================
# 3. INITIALIZE DEVICE AND MODEL
# =========================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

USE_HALF = torch.cuda.is_available()

print("\n========== MODEL INITIALIZATION ==========")

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

model = YOLO(MODEL_NAME)

print(f"Model loaded: {MODEL_NAME}")

print("===========================================\n")


# =========================================================
# 4. DRAW WRIST KEYPOINTS
# =========================================================

def draw_wrist_labels(frame, keypoints, keypoint_conf):

    wrist_indices = {
        LEFT_WRIST: "Left Wrist",
        RIGHT_WRIST: "Right Wrist"
    }

    for index, label in wrist_indices.items():

        x, y = keypoints[index]

        confidence = keypoint_conf[index]

        if confidence < KEYPOINT_CONFIDENCE:
            continue

        x = int(x)
        y = int(y)

        cv2.circle(
            frame,
            (x, y),
            6,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            f"{label}: {confidence:.2f}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )


# =========================================================
# 5. PROCESS VIDEO
# =========================================================

def run_pose_detection():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Input video not found: {VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    frame_number = 0

    total_people_detected = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # Run YOLO pose inference
            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE,
                half=USE_HALF,
                verbose=False
            )

            result = results[0]

            # Draw bounding boxes and complete pose skeletons
            annotated_frame = result.plot()

            person_count = 0

            if (
                result.keypoints is not None
                and result.boxes is not None
            ):

                keypoints_xy = (
                    result.keypoints.xy.cpu().numpy()
                )

                if result.keypoints.conf is not None:
                    keypoints_conf = (
                        result.keypoints.conf.cpu().numpy()
                    )
                else:
                    keypoints_conf = None

                for person_index, person_keypoints in enumerate(
                    keypoints_xy
                ):

                    person_count += 1

                    # Draw wrist labels only when confidence is available
                    if keypoints_conf is not None:

                        person_conf = keypoints_conf[
                            person_index
                        ]

                        draw_wrist_labels(
                            annotated_frame,
                            person_keypoints,
                            person_conf
                        )

            total_people_detected += person_count

            # Display frame information
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_number}/{total_frames}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"People detected: {person_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            writer.write(annotated_frame)

            # Show live output
            cv2.imshow(
                "YOLO Pose - Person and Wrist Detection",
                annotated_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nStopped by user.")
                break

            if frame_number % 100 == 0:

                print(
                    f"Processed {frame_number}/{total_frames} frames"
                )

    finally:

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\n========== POSE DETECTION SUMMARY ==========")

    print(f"Frames processed: {frame_number}")

    print(f"People detected across frames: {total_people_detected}")

    print(f"Output video: {OUTPUT_PATH}")

    print("============================================")


# =========================================================
# 6. MAIN
# =========================================================

if __name__ == "__main__":
    run_pose_detection()
