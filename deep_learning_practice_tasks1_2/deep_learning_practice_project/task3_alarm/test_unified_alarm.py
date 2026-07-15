from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from task3_alarm.unified_alarm import (
    DEFAULT_SCHEMA_PATH,
    convert_detection,
    convert_image_detection,
    convert_video_detection,
    validate_unified_alarm,
)


INPUT_JSON = Path("synthetic_detection.json")


class UnifiedAlarmConversionTests(unittest.TestCase):
    def test_schema_file_is_valid_json(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0")
        self.assertIn("event", schema["$defs"])

    def test_image_detection_converts_multiple_boxes_to_one_event(self) -> None:
        detection = {
            "status": "detected",
            "timestamp": "2026-07-15 09:17:38",
            "source": "sample.jpg",
            "num_images": 1,
            "num_detections": 2,
            "has_yiwu": True,
            "class_counts": {"石块异物": 1, "塑料异物": 1},
            "objects": [
                {
                    "image": "sample.jpg",
                    "class_id": 0,
                    "class": "stone",
                    "class_name": "石块异物",
                    "confidence": 0.85,
                    "bbox_xyxy": [10, 10, 30, 40],
                },
                {
                    "image": "sample.jpg",
                    "class_id": 1,
                    "class": "plastic",
                    "class_name": "塑料异物",
                    "confidence": 0.75,
                    "bbox_xyxy": [40, 20, 80, 60],
                },
            ],
        }

        document = convert_image_detection(detection, INPUT_JSON)

        self.assertEqual(document["source"]["type"], "image")
        self.assertEqual(document["detection_summary"]["event_count"], 1)
        self.assertEqual(document["detection_summary"]["detection_box_count"], 2)
        self.assertEqual(len(document["events"][0]["objects"]), 2)
        self.assertEqual(document["events"][0]["objects"][0]["area"], 600.0)
        self.assertEqual(document["events"][0]["risk"]["status"], "pending")
        self.assertEqual(validate_unified_alarm(document), [])

    def test_video_detection_joins_event_to_key_frame(self) -> None:
        key_frame = "outputs/video/detected_frames/frame_0003.jpg"
        detection = {
            "status": "completed",
            "created_at": "2026-07-15 08:53:56",
            "video": "outputs/uploaded_videos/belt.mp4",
            "video_start_time": "2026-07-15 08:51:40",
            "video_end_time": "2026-07-15 08:52:05",
            "duration_seconds": 25.0,
            "positive_frames": 3,
            "num_detection_boxes": 4,
            "has_foreign_object": True,
            "num_events": 1,
            "class_counts": {"石块异物": 1},
            "events": [
                {
                    "event_id": 1,
                    "start_offset_seconds": 15.0,
                    "end_offset_seconds": 17.0,
                    "start_video_time": "00:00:15.000",
                    "end_video_time": "00:00:17.000",
                    "start_real_time": "2026-07-15 08:51:55",
                    "end_real_time": "2026-07-15 08:51:57",
                    "object_count": 1,
                    "class_counts": {"石块异物": 1},
                    "observed_classes": ["石块异物"],
                    "max_confidence": 0.9,
                    "positive_sample_count": 3,
                    "unique_object_count": 1,
                    "track_ids": [7],
                    "key_frame": key_frame,
                    "frame_images": [key_frame],
                    "tracks": [
                        {
                            "track_id": 7,
                            "representative_frame": key_frame,
                            "representative_object": {
                                "track_id": 7,
                                "class_id": 0,
                                "class": "stone",
                                "class_name": "石块异物",
                                "confidence": 0.9,
                                "bbox_xyxy": [100, 100, 200, 250],
                            },
                        }
                    ],
                }
            ],
            "detection_frames": [
                {
                    "offset_seconds": 16.0,
                    "image": key_frame,
                    "object_count": 1,
                    "max_confidence": 0.9,
                    "objects": [
                        {
                            "class_id": 0,
                            "class": "stone",
                            "class_name": "石块异物",
                            "confidence": 0.9,
                            "bbox_xyxy": [100, 100, 200, 250],
                        }
                    ],
                }
            ],
        }

        document = convert_video_detection(detection, INPUT_JSON)

        event = document["events"][0]
        self.assertEqual(document["source"]["type"], "video")
        self.assertEqual(event["duration_seconds"], 2.0)
        self.assertEqual(event["key_frame"], key_frame)
        self.assertEqual(len(event["objects"]), 1)
        self.assertEqual(event["objects"][0]["area"], 15000.0)
        self.assertEqual(event["objects"][0]["track_id"], 7)
        self.assertEqual(event["detection_summary"]["positive_sample_count"], 3)
        self.assertEqual(event["detection_summary"]["unique_object_count"], 1)
        self.assertEqual(event["detection_summary"]["track_ids"], [7])
        self.assertEqual(validate_unified_alarm(document), [])

    def test_empty_video_has_no_alarm_events(self) -> None:
        detection = {
            "status": "completed",
            "video": "empty.mp4",
            "has_foreign_object": False,
            "num_events": 0,
            "events": [],
            "detection_frames": [],
        }

        document = convert_detection(detection, INPUT_JSON, source_type="auto")

        self.assertFalse(document["detection_summary"]["has_foreign_object"])
        self.assertEqual(document["events"], [])
        self.assertEqual(validate_unified_alarm(document), [])

    def test_validator_rejects_event_count_mismatch(self) -> None:
        detection = {
            "status": "detected",
            "source": "sample.jpg",
            "objects": [
                {
                    "class": "stone",
                    "class_name": "石块异物",
                    "confidence": 0.8,
                    "bbox_xyxy": [0, 0, 10, 10],
                }
            ],
        }
        document = convert_image_detection(detection, INPUT_JSON)
        invalid = deepcopy(document)
        invalid["detection_summary"]["event_count"] = 99

        errors = validate_unified_alarm(invalid)

        self.assertTrue(any("event_count" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
