
import cv2
import csv
import json
import torch
import numpy as np

from pathlib import Path
from collections import deque
from datetime import datetime
from itertools import product
from ultralytics import YOLO


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    PROJECT_DIR
    / "dataset"
    / "merged"
    / "merged_dataset.mp4"
)

# The JSON is expected beside main.py.
ROI_PATH = PROJECT_DIR / "camera_rois.json"

OUTPUT_DIR = PROJECT_DIR / "outputs"
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
EVENT_OUTPUT_DIR = OUTPUT_DIR / "events"

VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATED_VIDEO_PATH = (
    VIDEO_OUTPUT_DIR / "annotated_output.mp4"
)

CSV_OUTPUT_PATH = (
    EVENT_OUTPUT_DIR / "interaction_events.csv"
)

JSON_OUTPUT_PATH = (
    EVENT_OUTPUT_DIR / "interaction_events.json"
)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "yolov8n-pose.pt"

TRACKER_CONFIG = "bytetrack.yaml"

IMAGE_SIZE = 640

PERSON_CONFIDENCE = 0.35

KEYPOINT_CONFIDENCE = 0.35

USE_HALF = torch.cuda.is_available()

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# 3. INTERACTION CONFIGURATION
# ============================================================

# Maximum distance between two wrists to consider
# a possible hand-contact interaction.
CONTACT_DISTANCE = 90.0

# Required duration of proximity before classifying
# an interaction as likely hand contact.
CONTACT_DURATION = 0.5

# Minimum duration for handshake-like classification.
HANDSHAKE_DURATION = 1.2

# Number of directional reversals required.
HANDSHAKE_REVERSALS = 2

# Time allowed for temporary separation before
# ending an interaction episode.
SEPARATION_GRACE = 0.5

# Cooldown prevents repeated events.
COOLDOWN_DURATION = 1.0

# Maximum wrist movement required to register
# a meaningful directional change.
MOVEMENT_THRESHOLD = 3.0

# Counter boundary crossing tolerance.
CROSSING_TOLERANCE = 30


# ============================================================
# 4. COCO KEYPOINT INDICES
# ============================================================

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_ELBOW = 7
RIGHT_ELBOW = 8

LEFT_WRIST = 9
RIGHT_WRIST = 10

WRIST_INDICES = [
    LEFT_WRIST,
    RIGHT_WRIST
]


# ============================================================
# 5. LOAD ROI CONFIGURATION
# ============================================================

def load_roi_configuration():

    if not ROI_PATH.exists():

        raise FileNotFoundError(
            f"ROI configuration not found: {ROI_PATH}\n"
            "Place camera_rois.json beside main.py."
        )

    with open(ROI_PATH, "r") as file:

        config = json.load(file)

    expected_width = config["image_width"]
    expected_height = config["image_height"]

    staff_roi = np.array(
        config["staff_roi"],
        dtype=np.int32
    )

    customer_roi = np.array(
        config["customer_roi"],
        dtype=np.int32
    )

    counter_boundary = np.array(
        config["counter_boundary"],
        dtype=np.int32
    )

    if len(staff_roi) < 3:
        raise ValueError("Staff ROI needs at least 3 points.")

    if len(customer_roi) < 3:
        raise ValueError("Customer ROI needs at least 3 points.")

    if len(counter_boundary) < 2:
        raise ValueError("Counter boundary needs at least 2 points.")

    return (
        staff_roi,
        customer_roi,
        counter_boundary,
        expected_width,
        expected_height
    )


# ============================================================
# 6. GEOMETRY FUNCTIONS
# ============================================================

def point_inside_polygon(point, polygon):

    x, y = int(point[0]), int(point[1])

    result = cv2.pointPolygonTest(
        polygon,
        (x, y),
        False
    )

    return result >= 0

def signed_distance_to_boundary(point, boundary):

    """
    Signed perpendicular distance from a point to
    the infinite line defined by the first and last
    boundary coordinates.
    """

    p1 = boundary[0].astype(float)
    p2 = boundary[-1].astype(float)

    point = np.array(point, dtype=float)

    line = p2 - p1

    length = np.linalg.norm(line)

    if length == 0:
        return 0.0

    return (
        line[0] * (point[1] - p1[1])
        - line[1] * (point[0] - p1[0])
    ) / length


def point_near_boundary_segment(point, boundary):

    """
    Check whether the point is close to the actual
    configured boundary segment.
    """

    p1 = boundary[0].astype(float)
    p2 = boundary[-1].astype(float)

    point = np.array(point, dtype=float)

    line = p2 - p1

    length_squared = np.dot(line, line)

    if length_squared == 0:
        return False

    t = np.dot(point - p1, line) / length_squared

    if not 0 <= t <= 1:
        return False

    closest = p1 + t * line

    distance = np.linalg.norm(point - closest)

    return distance <= CROSSING_TOLERANCE


# ============================================================
# 7. PERSON ROLE ASSIGNMENT
# ============================================================

def assign_person_role(
    bbox,
    staff_roi,
    customer_roi,
    previous_role=None
):

    """
    Use the bottom-center of the person bounding box
    as the approximate footpoint.

    Retain the previous role if the person is temporarily
    outside both configured regions.
    """

    x1, y1, x2, y2 = bbox

    footpoint = (
        int((x1 + x2) / 2),
        int(y2)
    )

    in_staff = point_inside_polygon(
        footpoint,
        staff_roi
    )

    in_customer = point_inside_polygon(
        footpoint,
        customer_roi
    )

    if in_staff and not in_customer:
        return "STAFF"

    if in_customer and not in_staff:
        return "CUSTOMER"

    return previous_role


# ============================================================
# 8. WRIST EXTRACTION
# ============================================================

def extract_wrists(keypoints, keypoint_confidences):

    wrists = {}

    for index in WRIST_INDICES:

        if index >= len(keypoints):
            continue

        confidence = float(
            keypoint_confidences[index]
        )

        if confidence < KEYPOINT_CONFIDENCE:
            continue

        x, y = keypoints[index]

        wrists[index] = {
            "x": float(x),
            "y": float(y),
            "confidence": confidence
        }

    return wrists


# ============================================================
# 9. PERSON TRACKING AND BOUNDARY CROSSING
# ============================================================

class PersonManager:

    def __init__(self):

        self.previous_roles = {}

        self.previous_wrist_sides = {}

        self.person_crossings = set()

    def process_person(
        self,
        track_id,
        bbox,
        keypoints,
        keypoint_confidences,
        staff_roi,
        customer_roi,
        boundary
    ):

        previous_role = self.previous_roles.get(
            track_id
        )

        role = assign_person_role(
            bbox,
            staff_roi,
            customer_roi,
            previous_role
        )

        if role is not None:
            self.previous_roles[track_id] = role

        wrists = extract_wrists(
            keypoints,
            keypoint_confidences
        )

        crossed = False

        for wrist_index, wrist in wrists.items():

            point = (
                wrist["x"],
                wrist["y"]
            )

            # Ignore points outside the finite boundary segment.
            if not point_near_boundary_segment(
                point,
                boundary
            ):
                continue

            side = signed_distance_to_boundary(
                point,
                boundary
            )

            current_sign = (
                1 if side > 0
                else -1 if side < 0
                else 0
            )

            wrist_key = (
                track_id,
                wrist_index
            )

            previous_sign = self.previous_wrist_sides.get(
                wrist_key
            )

            if (
                previous_sign is not None
                and current_sign != 0
                and previous_sign != 0
                and current_sign != previous_sign
            ):

                crossed = True

            if current_sign != 0:

                self.previous_wrist_sides[
                    wrist_key
                ] = current_sign

        return {
            "track_id": track_id,
            "role": role,
            "bbox": bbox,
            "wrists": wrists,
            "crossed": crossed
        }


# ============================================================
# 10. HANDSHAKE MOTION ANALYSIS
# ============================================================

def count_direction_reversals(points):

    """
    Count meaningful changes in the direction of
    the interaction midpoint along its dominant axis.
    """

    if len(points) < 4:
        return 0

    points = np.array(points, dtype=float)

    displacement = points[-1] - points[0]

    axis = 0 if abs(displacement[0]) >= abs(displacement[1]) else 1

    values = points[:, axis]

    differences = np.diff(values)

    directions = []

    for value in differences:

        if value > MOVEMENT_THRESHOLD:
            directions.append(1)

        elif value < -MOVEMENT_THRESHOLD:
            directions.append(-1)

    reversals = 0

    for previous, current in zip(
        directions,
        directions[1:]
    ):

        if previous != current:
            reversals += 1

    return reversals


# ============================================================
# 11. INTERACTION STATE MACHINE
# ============================================================

class InteractionStateMachine:

    def __init__(self):

        self.pairs = {}

        self.events = []

    def create_pair(self, pair_key):

        self.pairs[pair_key] = {
            "state": "NO_INTERACTION",

            "start_time": None,

            "last_contact_time": None,

            "cooldown_until": 0.0,

            "contact_frames": 0,

            "boundary_crossed": False,

            "midpoints": deque(maxlen=30),

            "event_index": None,

            "minimum_distance": None
        }

    def update(
        self,
        staff,
        customer,
        frame_number,
        timestamp,
        current_time
    ):

        pair_key = (
            staff["track_id"],
            customer["track_id"]
        )

        if pair_key not in self.pairs:

            self.create_pair(pair_key)

        pair = self.pairs[pair_key]

        if current_time < pair["cooldown_until"]:
            return

        staff_wrists = list(
            staff["wrists"].values()
        )

        customer_wrists = list(
            customer["wrists"].values()
        )

        if not staff_wrists or not customer_wrists:
            return

        distances = []

        for sw, cw in product(
            staff_wrists,
            customer_wrists
        ):

            distance = np.linalg.norm(
                np.array([sw["x"], sw["y"]])
                - np.array([cw["x"], cw["y"]])
            )

            distances.append(
                (distance, sw, cw)
            )

        distance, sw, cw = min(
            distances,
            key=lambda item: item[0]
        )

        if distance <= CONTACT_DISTANCE:

            if pair["state"] in [
                "NO_INTERACTION",
                "COOLDOWN",
                "UNCERTAIN"
            ]:

                pair["state"] = "HAND_APPROACHING"

                pair["start_time"] = current_time

                pair["last_contact_time"] = current_time

                pair["contact_frames"] = 0

                pair["boundary_crossed"] = False

                pair["midpoints"].clear()

                pair["event_index"] = None

                pair["minimum_distance"] = float(distance)

            pair["last_contact_time"] = current_time

            pair["contact_frames"] += 1

            pair["minimum_distance"] = min(
                pair["minimum_distance"],
                float(distance)
            )

            midpoint = (
                (sw["x"] + cw["x"]) / 2,
                (sw["y"] + cw["y"]) / 2
            )

            pair["midpoints"].append(midpoint)

            pair["state"] = "POSSIBLE_HAND_CONTACT"

            duration = (
                current_time - pair["start_time"]
            )

            reversals = count_direction_reversals(
                pair["midpoints"]
            )

            if (
                duration >= CONTACT_DURATION
                and pair["event_index"] is None
            ):

                pair["state"] = "LIKELY_HAND_CONTACT"

                event = {
                    "event_id": len(self.events) + 1,
                    "frame": frame_number,
                    "timestamp": timestamp,
                    "staff_track_id": staff["track_id"],
                    "customer_track_id": customer["track_id"],
                    "event_type": "LIKELY_HAND_CONTACT",
                    "state": pair["state"],
                    "duration_seconds": round(duration, 3),
                    "minimum_wrist_distance": round(
                        pair["minimum_distance"], 2
                    ),
                    "boundary_crossed": pair["boundary_crossed"],
                    "handshake_reversals": reversals
                }

                self.events.append(event)

                pair["event_index"] = len(
                    self.events
                ) - 1

            if (
                pair["event_index"] is not None
                and duration >= HANDSHAKE_DURATION
                and reversals >= HANDSHAKE_REVERSALS
            ):

                pair["state"] = "LIKELY_HANDSHAKE"

                event = self.events[
                    pair["event_index"]
                ]

                event["event_type"] = "LIKELY_HANDSHAKE"

                event["state"] = pair["state"]

                event["handshake_reversals"] = reversals

                event["duration_seconds"] = round(
                    duration, 3
                )

        else:

            # End interaction only after the grace period.
            if (
                pair["last_contact_time"] is not None
                and current_time - pair["last_contact_time"]
                > SEPARATION_GRACE
            ):

                pair["state"] = "COOLDOWN"

                pair["cooldown_until"] = (
                    current_time + COOLDOWN_DURATION
                )

                pair["start_time"] = None

                pair["last_contact_time"] = None

                pair["midpoints"].clear()

    def mark_boundary_crossing(self, staff_id, customer_id):

        pair_key = (staff_id, customer_id)

        if pair_key in self.pairs:

            self.pairs[pair_key][
                "boundary_crossed"
            ] = True

            event_index = self.pairs[pair_key][
                "event_index"
            ]

            if event_index is not None:

                self.events[event_index][
                    "boundary_crossed"
                ] = True


# ============================================================
# 12. EVENT LOGGING
# ============================================================

def save_events(events):

    fields = [
        "event_id",
        "frame",
        "timestamp",
        "staff_track_id",
        "customer_track_id",
        "event_type",
        "state",
        "duration_seconds",
        "minimum_wrist_distance",
        "boundary_crossed",
        "handshake_reversals"
    ]

    with open(
        CSV_OUTPUT_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(events)

    with open(
        JSON_OUTPUT_PATH,
        "w"
    ) as file:

        json.dump(
            events,
            file,
            indent=4
        )

    print("\n========== EVENT REPORT ==========")

    print(f"Total events: {len(events)}")

    print(f"CSV: {CSV_OUTPUT_PATH}")

    print(f"JSON: {JSON_OUTPUT_PATH}")

    print("==================================")


# ============================================================
# 13. DRAW CAMERA CONFIGURATION
# ============================================================

def draw_camera_regions(
    frame,
    staff_roi,
    customer_roi,
    boundary
):

    cv2.polylines(
        frame,
        [staff_roi],
        True,
        (0, 255, 0),
        2
    )

    cv2.polylines(
        frame,
        [customer_roi],
        True,
        (255, 0, 0),
        2
    )

    cv2.polylines(
        frame,
        [boundary],
        False,
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "STAFF REGION",
        (staff_roi[0][0], staff_roi[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "CUSTOMER REGION",
        (customer_roi[0][0], customer_roi[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    return frame


# ============================================================
# 14. MAIN VIDEO PROCESSING PIPELINE
# ============================================================

def main():

    print("\n========== SYSTEM INITIALIZATION ==========")

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print("Loading camera ROIs...")

    (
        staff_roi,
        customer_roi,
        boundary,
        expected_width,
        expected_height
    ) = load_roi_configuration()

    print("Camera ROIs loaded.")

    print("Loading YOLO Pose model...")

    model = YOLO(MODEL_NAME)

    print("YOLO Pose model loaded.")

    print("ByteTrack configured.")

    print("===========================================\n")

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Input video not found: {VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open input video."
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

    if (
        width != expected_width
        or height != expected_height
    ):

        cap.release()

        raise ValueError(
            "Video dimensions do not match ROI configuration. "
            f"Video: {width}x{height}, "
            f"ROI: {expected_width}x{expected_height}"
        )

    writer = cv2.VideoWriter(
        str(ANNOTATED_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create annotated output video."
        )

    person_manager = PersonManager()

    state_machine = InteractionStateMachine()

    frame_number = 0

    # Store IDs for which a crossing has already been logged
    crossing_logged = set()

    print("Starting video processing...\n")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            current_time = frame_number / fps

            timestamp = (
                datetime.utcfromtimestamp(
                    current_time
                ).strftime("%H:%M:%S.%f")[:-3]
            )

            # --------------------------------------------
            # YOLO POSE + BYTETRACK
            # --------------------------------------------

            results = model.track(
                source=frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                imgsz=IMAGE_SIZE,
                conf=PERSON_CONFIDENCE,
                device=DEVICE,
                half=USE_HALF,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()

            draw_camera_regions(
                annotated,
                staff_roi,
                customer_roi,
                boundary
            )

            people = []

            if (
                result.boxes is not None
                and result.keypoints is not None
                and result.boxes.id is not None
                and result.keypoints.conf is not None
            ):

                track_ids = (
                    result.boxes.id.cpu().numpy().astype(int)
                )

                boxes = (
                    result.boxes.xyxy.cpu().numpy()
                )

                keypoints_xy = (
                    result.keypoints.xy.cpu().numpy()
                )

                keypoints_conf = (
                    result.keypoints.conf.cpu().numpy()
                )

                for i, track_id in enumerate(track_ids):

                    person = person_manager.process_person(
                        track_id=track_id,
                        bbox=boxes[i],
                        keypoints=keypoints_xy[i],
                        keypoint_confidences=keypoints_conf[i],
                        staff_roi=staff_roi,
                        customer_roi=customer_roi,
                        boundary=boundary
                    )

                    people.append(person)

                    # ------------------------------------
                    # BOUNDARY CROSSING LOG
                    # ------------------------------------

                    if (
                        person["crossed"]
                        and track_id not in crossing_logged
                    ):

                        crossing_logged.add(track_id)

                        state_machine.events.append({

                            "event_id": len(
                                state_machine.events
                            ) + 1,

                            "frame": frame_number,

                            "timestamp": timestamp,

                            "staff_track_id": (
                                track_id
                                if person["role"] == "STAFF"
                                else None
                            ),

                            "customer_track_id": (
                                track_id
                                if person["role"] == "CUSTOMER"
                                else None
                            ),

                            "event_type": "WRIST_BOUNDARY_CROSSING",

                            "state": "BOUNDARY_CROSSED",

                            "duration_seconds": 0.0,

                            "minimum_wrist_distance": None,

                            "boundary_crossed": True,

                            "handshake_reversals": 0
                        })

                    # Draw track ID and role
                    x1, y1, x2, y2 = map(
                        int,
                        person["bbox"]
                    )

                    label = (
                        f"ID {track_id} | "
                        f"{person['role'] or 'UNKNOWN'}"
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    # Draw wrist points
                    for wrist in person["wrists"].values():

                        point = (
                            int(wrist["x"]),
                            int(wrist["y"])
                        )

                        cv2.circle(
                            annotated,
                            point,
                            7,
                            (0, 255, 255),
                            -1
                        )

            # --------------------------------------------
            # STAFF-CUSTOMER INTERACTION ANALYSIS
            # --------------------------------------------

            staff_people = [
                p for p in people
                if p["role"] == "STAFF"
            ]

            customer_people = [
                p for p in people
                if p["role"] == "CUSTOMER"
            ]

            for staff in staff_people:

                for customer in customer_people:

                    state_machine.update(
                        staff=staff,
                        customer=customer,
                        frame_number=frame_number,
                        timestamp=timestamp,
                        current_time=current_time
                    )

                    if (
                        staff["crossed"]
                        or customer["crossed"]
                    ):

                        state_machine.mark_boundary_crossing(
                            staff["track_id"],
                            customer["track_id"]
                        )

                    pair_key = (
                        staff["track_id"],
                        customer["track_id"]
                    )

                    pair = state_machine.pairs.get(
                        pair_key
                    )

                    if pair is not None:

                        state = pair["state"]

                        cv2.putText(
                            annotated,
                            f"Interaction: {state}",
                            (20, height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2
                        )

            # --------------------------------------------
            # FRAME INFORMATION
            # --------------------------------------------

            cv2.putText(
                annotated,
                f"Frame: {frame_number}/{total_frames}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"People: {len(people)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Events: {len(state_machine.events)}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            writer.write(annotated)

            cv2.imshow(
                "Counter Hand Interaction Detection",
                annotated
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                print("\nProcessing stopped by user.")

                break

            if frame_number % 100 == 0:

                print(
                    f"Processed {frame_number}/{total_frames} "
                    f"| Events: {len(state_machine.events)}"
                )

    finally:

        cap.release()

        writer.release()

        cv2.destroyAllWindows()

        save_events(
            state_machine.events
        )

    print("\n========== PROCESSING COMPLETED ==========")

    print(f"Frames processed: {frame_number}")

    print(f"Annotated video: {ANNOTATED_VIDEO_PATH}")

    print(f"CSV events: {CSV_OUTPUT_PATH}")

    print(f"JSON events: {JSON_OUTPUT_PATH}")

    print("===========================================")


# ============================================================
# 15. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
