# parking_zone_detector.py
import cv2
import numpy as np
import os
import json
import threading
from datetime import datetime
from playsound import playsound


class ParkingZoneDetector:
    def __init__(self):
        self.legal_zones = []      # List of legal parking zone polygons
        self.illegal_zones = []    # List of illegal parking zone polygons
        self.detection_history = {}  # Track detections over time
        self.persistence_threshold = 5  # Number of frames to confirm illegal parking
        self.zone_colors = {
            'legal': (0, 255, 0),     # Green for legal zones
            'illegal': (0, 0, 255),   # Red for illegal zones
            'warning': (0, 165, 255)  # Orange for warnings
        }
        self.config_dir = "./parking_zones/"
        os.makedirs(self.config_dir, exist_ok=True)

    def play_warning_sound(self, sound_file="sounds/alert.mp3"):
        """Plays warning sound in a separate thread"""
        threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()

    def add_zone(self, points, zone_type='legal'):
        """Add a new zone defined by points [(x1,y1), (x2,y2), ...]"""
        if zone_type == 'legal':
            self.legal_zones.append(np.array(points))
        elif zone_type == 'illegal':
            self.illegal_zones.append(np.array(points))
        else:
            raise ValueError("Zone type must be 'legal' or 'illegal'")

    def clear_zones(self, zone_type=None):
        """Clear all zones or specific zone type"""
        if zone_type is None or zone_type == 'legal':
            self.legal_zones = []
        if zone_type is None or zone_type == 'illegal':
            self.illegal_zones = []

    def save_zones(self, filename="parking_zones.json"):
        """Save defined zones to a JSON file"""
        data = {
            "legal_zones": [zone.tolist() for zone in self.legal_zones],
            "illegal_zones": [zone.tolist() for zone in self.illegal_zones]
        }

        filepath = os.path.join(self.config_dir, filename)
        print(filepath)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return filepath

    def load_zones(self, filename="parking_zones.json"):
        """Load zones from a JSON file"""
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.legal_zones = [np.array(zone) for zone in data.get("legal_zones", [])]
        self.illegal_zones = [np.array(zone) for zone in data.get("illegal_zones", [])]
        return True

    def is_point_in_any_zone(self, point, zone_list):
        """Check if a point is inside any zone in the list"""
        for zone in zone_list:
            if cv2.pointPolygonTest(zone, point, False) >= 0:
                return True
        return False

    def is_box_in_zone(self, box, zone):
        """Check if a bounding box intersects with a zone"""
        x1, y1, x2, y2 = box
        box_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        for point in box_polygon:
            if cv2.pointPolygonTest(zone, tuple(point), False) >= 0:
                return True

        for point in zone:
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                return True

        return False

    def check_illegal_parking(self, frame, detections):
        """
        Check for illegally parked cars
        detections: List of (box, confidence, class_id) where box is [x1, y1, x2, y2]
        Returns: Annotated frame and list of illegal parking events
        """
        illegal_events = []
        current_detections = set()
        timestamp = datetime.now().isoformat()

        overlay = frame.copy()
        for zone in self.legal_zones:
            cv2.polylines(overlay, [zone], True, self.zone_colors['legal'], 2)
            cv2.fillPoly(overlay, [zone], (*self.zone_colors['legal'], 40))

        for zone in self.illegal_zones:
            cv2.polylines(overlay, [zone], True, self.zone_colors['illegal'], 2)
            cv2.fillPoly(overlay, [zone], (*self.zone_colors['illegal'], 40))

        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        for bbox, confidence, class_id in detections:
            x1, y1, x2, y2 = map(int, bbox)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detection_id = f"{center_x}_{center_y}"
            current_detections.add(detection_id)

            in_illegal_zone = any(self.is_box_in_zone(bbox, zone) for zone in self.illegal_zones)
            in_legal_zone = any(self.is_box_in_zone(bbox, zone) for zone in self.legal_zones)

            is_illegal = in_illegal_zone or (not in_legal_zone and self.legal_zones)

            if detection_id not in self.detection_history:
                self.detection_history[detection_id] = {
                    'count': 1 if is_illegal else 0,
                    'illegal': is_illegal,
                    'first_seen': timestamp
                }
            else:
                if is_illegal:
                    self.detection_history[detection_id]['count'] += 1
                self.detection_history[detection_id]['illegal'] = is_illegal

            confirmed_illegal = (
                self.detection_history[detection_id]['count'] >= self.persistence_threshold
            )

            color = self.zone_colors['warning'] if is_illegal else (255, 255, 255)
            if confirmed_illegal:
                color = self.zone_colors['illegal']

                illegal_events.append({
                    'bbox': bbox,
                    'center': (center_x, center_y),
                    'first_detected': self.detection_history[detection_id]['first_seen'],
                    'current_time': timestamp
                })

                # Play warning sound
                self.play_warning_sound()

                cv2.putText(frame, "ILLEGAL PARKING", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        keys_to_remove = [k for k in self.detection_history.keys() if k not in current_detections]
        for k in keys_to_remove:
            del self.detection_history[k]

        return frame, illegal_events

    def draw_zones(self, frame):
        """Draw all zones on the frame for visualization"""
        overlay = frame.copy()

        for zone in self.legal_zones:
            cv2.polylines(overlay, [zone], True, self.zone_colors['legal'], 2)
            cv2.fillPoly(overlay, [zone], (*self.zone_colors['legal'], 80))

        for zone in self.illegal_zones:
            cv2.polylines(overlay, [zone], True, self.zone_colors['illegal'], 2)
            cv2.fillPoly(overlay, [zone], (*self.zone_colors['illegal'], 80))

        alpha = 0.4
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def draw_zone_editor(self, frame, current_zone_points, zone_type='legal'):
        """Draw current zone being edited on the frame"""
        edited_frame = self.draw_zones(frame.copy())

        color = self.zone_colors['legal'] if zone_type == 'legal' else self.zone_colors['illegal']

        points = np.array(current_zone_points, dtype=np.int32)
        if len(points) > 0:
            for point in points:
                cv2.circle(edited_frame, tuple(point), 5, color, -1)

            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(edited_frame, tuple(points[i]), tuple(points[i + 1]), color, 2)

                if len(points) >= 3:
                    cv2.line(edited_frame, tuple(points[-1]), tuple(points[0]), (255, 255, 255), 2, cv2.LINE_AA)

        return edited_frame
