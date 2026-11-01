
import cv2
import json
from pathlib import Path


# =========================================================
# 1. CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    PROJECT_DIR
    / "dataset"
    / "merged"
    / "merged_dataset.mp4"
)

OUTPUT_DIR = PROJECT_DIR / "configs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "camera_rois.json"

WINDOW_NAME = "Counter ROI Coordinate Selector"


# =========================================================
# 2. ROI CONFIGURATION
# =========================================================

rois = {
    "staff_roi": [],
    "customer_roi": [],
    "counter_boundary": []
}

current_mode = "staff_roi"

mode_labels = {
    "staff_roi": "STAFF ROI",
    "customer_roi": "CUSTOMER ROI",
    "counter_boundary": "COUNTER BOUNDARY"
}

colors = {
    "staff_roi": (0, 255, 0),
    "customer_roi": (255, 0, 0),
    "counter_boundary": (0, 0, 255)
}

completed = {
    "staff_roi": False,
    "customer_roi": False,
    "counter_boundary": False
}

frame = None


# =========================================================
# 3. MOUSE CALLBACK
# =========================================================

def mouse_callback(event, x, y, flags, param):

    global rois, current_mode

    if event == cv2.EVENT_LBUTTONDOWN:

        rois[current_mode].append([int(x), int(y)])

        print(
            f"{current_mode}: "
            f"({x}, {y})"
        )

    elif event == cv2.EVENT_RBUTTONDOWN:

        if rois[current_mode]:

            removed = rois[current_mode].pop()

            print(
                f"Removed point: {removed}"
            )


# =========================================================
# 4. DRAW SELECTED COORDINATES
# =========================================================

def draw_rois(image):

    output = image.copy()

    for roi_name, points in rois.items():

        color = colors[roi_name]

        # Draw selected points
        for index, point in enumerate(points):

            x, y = point

            cv2.circle(
                output,
                (x, y),
                6,
                color,
                -1
            )

            cv2.putText(
                output,
                str(index + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Draw polygon for staff/customer
        if roi_name in ["staff_roi", "customer_roi"]:

            if len(points) >= 2:

                for i in range(len(points) - 1):

                    cv2.line(
                        output,
                        tuple(points[i]),
                        tuple(points[i + 1]),
                        color,
                        2
                    )

            if completed[roi_name] and len(points) >= 3:

                cv2.polylines(
                    output,
                    [__import__("numpy").array(points, dtype="int32")],
                    True,
                    color,
                    2
                )

        # Draw counter boundary as an open line
        elif roi_name == "counter_boundary":

            if len(points) >= 2:

                cv2.polylines(
                    output,
                    [__import__("numpy").array(points, dtype="int32")],
                    False,
                    color,
                    3
                )

    return output


# =========================================================
# 5. SAVE COORDINATES
# =========================================================

def save_coordinates(width, height):

    data = {
        "image_width": width,
        "image_height": height,
        "staff_roi": rois["staff_roi"],
        "customer_roi": rois["customer_roi"],
        "counter_boundary": rois["counter_boundary"]
    }

    with open(OUTPUT_PATH, "w") as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print("\n========== COORDINATES SAVED ==========")

    print(json.dumps(data, indent=4))

    print(f"\nSaved to: {OUTPUT_PATH}")


# =========================================================
# 6. MAIN ROI SELECTOR
# =========================================================

def main():

    global frame, current_mode

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open merged video.")

    # Read the first frame
    ret, frame = cap.read()

    cap.release()

    if not ret:
        raise RuntimeError("Could not read video frame.")

    height, width = frame.shape[:2]

    print("\n========== ROI SELECTOR ==========")

    print(f"Frame resolution: {width} x {height}")

    print("\nMOUSE CONTROLS:")
    print("Left Click  : Add coordinate")
    print("Right Click : Undo last coordinate")

    print("\nKEYBOARD CONTROLS:")
    print("1 : Select Staff ROI")
    print("2 : Select Customer ROI")
    print("3 : Select Counter Boundary")
    print("C : Complete current selection")
    print("R : Reset current selection")
    print("S : Save all coordinates")
    print("Q : Quit")

    print("\nSelect the STAFF ROI first.")

    print("==================================\n")

    cv2.namedWindow(WINDOW_NAME)

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    while True:

        display = draw_rois(frame)

        # Display current selection information
        instructions = [
            f"MODE: {mode_labels[current_mode]}",
            f"Selected points: {len(rois[current_mode])}",
            "1: Staff | 2: Customer | 3: Boundary",
            "C: Complete | R: Reset | S: Save | Q: Quit"
        ]

        for i, text in enumerate(instructions):

            cv2.putText(
                display,
                text,
                (15, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        # Select Staff ROI
        if key == ord("1"):

            current_mode = "staff_roi"

            print("\nSelected STAFF ROI.")

        # Select Customer ROI
        elif key == ord("2"):

            current_mode = "customer_roi"

            print("\nSelected CUSTOMER ROI.")

        # Select Counter Boundary
        elif key == ord("3"):

            current_mode = "counter_boundary"

            print("\nSelected COUNTER BOUNDARY.")

        # Complete current ROI
        elif key == ord("c"):

            points = rois[current_mode]

            minimum = (
                2 if current_mode == "counter_boundary"
                else 3
            )

            if len(points) < minimum:

                print(
                    f"Select at least {minimum} points."
                )

            else:

                completed[current_mode] = True

                print(
                    f"{current_mode} completed."
                )

        # Reset current ROI
        elif key == ord("r"):

            rois[current_mode] = []

            completed[current_mode] = False

            print(
                f"{current_mode} reset."
            )

        # Save all coordinates
        elif key == ord("s"):

            if not all(completed.values()):

                print(
                    "Complete all three selections before saving."
                )

            else:

                save_coordinates(width, height)

        # Exit
        elif key == ord("q"):

            print("\nExiting ROI selector.")
            break

    cv2.destroyAllWindows()


# =========================================================
# 7. RUN
# =========================================================

if __name__ == "__main__":
    main()
