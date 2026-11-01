
import cv2
from pathlib import Path


# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_DIR = PROJECT_DIR / "video"

OUTPUT_DIR = PROJECT_DIR / "merged"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "merged_dataset.mp4"


# =========================================================
# 2. INPUT VIDEOS - PROCESS IN THIS EXACT ORDER
# =========================================================

VIDEO_FILES = [
    "VID20261024182729.mp4",
    "VID20261024182900.mp4",
    "VID20261024183129.mp4",
    "VID20261024183839.mp4",
    "VID20261024184058.mp4"
]


# =========================================================
# 3. OUTPUT VIDEO CONFIGURATION
# =========================================================

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TARGET_FPS = 30

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


# =========================================================
# 4. RESIZE FRAME WHILE PRESERVING ASPECT RATIO
# =========================================================

def resize_frame(frame):

    height, width = frame.shape[:2]

    scale = min(
        TARGET_WIDTH / width,
        TARGET_HEIGHT / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = cv2.copyMakeBorder(
        resized,
        (TARGET_HEIGHT - new_height) // 2,
        (TARGET_HEIGHT - new_height + 1) // 2,
        (TARGET_WIDTH - new_width) // 2,
        (TARGET_WIDTH - new_width + 1) // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return canvas


# =========================================================
# 5. MERGE VIDEOS
# =========================================================
print("Script started")
def merge_videos():

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        FOURCC,
        TARGET_FPS,
        (TARGET_WIDTH, TARGET_HEIGHT)
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video.")

    total_frames_written = 0

    try:

        for video_index, video_name in enumerate(VIDEO_FILES, start=1):

            video_path = VIDEO_DIR / video_name

            print("\n" + "=" * 55)
            print(f"Processing video {video_index}/{len(VIDEO_FILES)}")
            print(f"File: {video_name}")

            if not video_path.exists():
                print(f"WARNING: File not found: {video_path}")
                continue

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print(f"WARNING: Could not open {video_name}")
                cap.release()
                continue

            source_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            duration = (
                frame_count / source_fps
                if source_fps > 0 else 0
            )

            print(f"Source FPS: {source_fps:.2f}")
            print(f"Frame count: {frame_count}")
            print(f"Duration: {duration:.2f} seconds")

            if source_fps <= 0 or frame_count <= 0:
                print("WARNING: Invalid video properties. Skipping.")
                cap.release()
                continue

            # Sample frames according to the target output FPS.
            output_frame_count = int(duration * TARGET_FPS)

            frames_written = 0

            for frame_index in range(output_frame_count):

                timestamp_ms = (
                    frame_index / TARGET_FPS
                ) * 1000

                cap.set(
                    cv2.CAP_PROP_POS_MSEC,
                    timestamp_ms
                )

                ret, frame = cap.read()

                if not ret:
                    break

                frame = resize_frame(frame)

                writer.write(frame)

                frames_written += 1
                total_frames_written += 1

            cap.release()

            print(f"Frames added: {frames_written}")
            print(f"Completed: {video_name}")

    finally:
        writer.release()

    if total_frames_written == 0:
        raise RuntimeError("No frames were written. Check the input videos.")

    total_duration = total_frames_written / TARGET_FPS

    print("\n" + "=" * 55)
    print("VIDEO MERGING COMPLETED")
    print("=" * 55)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total frames: {total_frames_written}")
    print(f"Output FPS: {TARGET_FPS}")
    print(f"Resolution: {TARGET_WIDTH} x {TARGET_HEIGHT}")
    print(f"Total duration: {total_duration:.2f} seconds")


# =========================================================
# 6. MAIN
# =========================================================

if __name__ == "__main__":
    merge_videos()
