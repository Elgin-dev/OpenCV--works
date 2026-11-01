
import cv2
from pathlib import Path


# =========================================================
# 1. VIDEO CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_DIR
    / "dataset"
    / "merged"
    / "merged_dataset.mp4"
)


# =========================================================
# 2. VIDEO VALIDATION
# =========================================================

def validate_video(video_path):

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open the merged video."
        )

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

    duration = (
        total_frames / fps
        if fps > 0 else 0
    )

    print("\n========== VIDEO INFORMATION ==========")

    print(f"Video path       : {video_path}")
    print(f"Resolution       : {width} x {height}")
    print(f"FPS              : {fps:.2f}")
    print(f"Total frames     : {total_frames}")
    print(f"Duration         : {duration:.2f} seconds")

    print("========================================\n")

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid video properties.")

    return cap, fps, total_frames


# =========================================================
# 3. VIDEO PLAYBACK
# =========================================================

def play_video():

    cap, fps, total_frames = validate_video(
        VIDEO_PATH
    )

    frame_delay = max(1, int(1000 / fps))

    current_frame = 0

    print("Playback controls:")
    print("Q - Exit")
    print("SPACE - Pause / Resume")

    paused = False

    try:

        while True:

            if not paused:

                ret, frame = cap.read()

                if not ret:
                    print("\nVideo playback completed.")
                    break

                current_frame += 1

                cv2.putText(
                    frame,
                    f"Frame: {current_frame}/{total_frames}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    "Counter Hand Interaction - Dataset",
                    frame
                )

            key = cv2.waitKey(frame_delay) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                paused = not paused

    finally:

        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# 4. MAIN
# =========================================================

if __name__ == "__main__":
    play_video()
