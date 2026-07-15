from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path


if importlib.util.find_spec("cv2") is None:
    sys.modules["cv2"] = types.ModuleType("cv2")
if importlib.util.find_spec("ultralytics") is None:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object
    sys.modules["ultralytics"] = ultralytics_stub


from video_detection import (  # noqa: E402
    LightweightTracker,
    filter_duplicate_objects,
    merge_detection_events,
    parse_roi,
    process_frame_objects,
    sample_source_index,
)


def raw_object(
    class_name: str,
    confidence: float,
    bbox: list[float],
    class_id: int = 0,
) -> dict:
    display_names = {
        "stone": "石块异物",
        "plastic": "塑料异物",
        "metal": "金属异物",
        "wood": "木块异物",
    }
    return {
        "predicted_class_id": class_id,
        "predicted_class": class_name,
        "predicted_class_name": display_names[class_name],
        "confidence": confidence,
        "bbox_xyxy": bbox,
    }


def event_frame(offset: float, image: str, objects: list[dict]) -> dict:
    return {
        "offset_seconds": offset,
        "video_time": f"00:00:{offset:06.3f}",
        "real_time": "2026-07-15 08:00:00",
        "image": image,
        "objects": objects,
        "object_count": len(objects),
        "class_counts": {},
        "max_confidence": max((item["confidence"] for item in objects), default=0.0),
    }


class VideoDetectionLogicTests(unittest.TestCase):
    def test_sampling_4fps_contains_all_2fps_source_frames(self) -> None:
        source_fps = 119.208
        two_fps = {
            sample_source_index(index, source_fps, 2.0) for index in range(52)
        }
        four_fps = {
            sample_source_index(index, source_fps, 4.0) for index in range(104)
        }
        self.assertTrue(two_fps.issubset(four_fps))

    def test_duplicate_filter_keeps_only_highest_confidence_overlap(self) -> None:
        detections = [
            raw_object("plastic", 0.722, [0, 0, 100, 100], 1),
            raw_object("plastic", 0.551, [10, 10, 95, 95], 1),
            raw_object("plastic", 0.364, [15, 15, 90, 90], 1),
            raw_object("plastic", 0.80, [120, 0, 220, 100], 1),
        ]

        kept, ignored = filter_duplicate_objects(detections, duplicate_iou=0.45)

        self.assertEqual([item["confidence"] for item in kept], [0.8, 0.722])
        self.assertEqual(len(ignored), 2)
        self.assertTrue(all(item["detection_state"] == "background_ignored" for item in ignored))

    def test_unknown_requires_repeated_hits(self) -> None:
        tracker = LightweightTracker(
            known_conf=0.4,
            min_unknown_hits=2,
            unknown_single_frame_conf=0.38,
        )
        first = process_frame_objects(
            [raw_object("stone", 0.30, [100, 100, 200, 200])], tracker, 0.0
        )
        second = process_frame_objects(
            [raw_object("stone", 0.31, [80, 100, 180, 200])], tracker, 0.25
        )

        self.assertEqual(first["objects"], [])
        self.assertEqual(first["unknown_candidates"][0]["detection_state"], "unknown_candidate")
        self.assertEqual(second["objects"][0]["detection_state"], "confirmed_unknown")
        self.assertEqual(
            first["unknown_candidates"][0]["track_id"], second["objects"][0]["track_id"]
        )

    def test_single_unknown_candidate_does_not_bridge_events(self) -> None:
        first_tracker = LightweightTracker()
        first_objects = process_frame_objects(
            [raw_object("stone", 0.85, [0, 0, 100, 100])], first_tracker, 0.0
        )["objects"]
        candidate_only = process_frame_objects(
            [raw_object("plastic", 0.20, [400, 0, 500, 100], 1)],
            first_tracker,
            0.5,
        )
        second_objects = process_frame_objects(
            [raw_object("plastic", 0.90, [300, 0, 400, 100], 1)],
            first_tracker,
            2.0,
        )["objects"]
        positive_frames = [
            event_frame(0.0, "first.jpg", first_objects),
            event_frame(2.0, "second.jpg", second_objects),
        ]

        events = merge_detection_events(
            positive_frames,
            datetime(2026, 7, 15, 8, 0, 0),
            sample_fps=4.0,
            duration=3.0,
            event_silence_seconds=1.0,
        )

        self.assertEqual(candidate_only["objects"], [])
        self.assertEqual(len(events), 2)

    def test_track_survives_short_detection_gap_without_fake_box(self) -> None:
        tracker = LightweightTracker(track_max_age_seconds=1.0)
        first = process_frame_objects(
            [raw_object("stone", 0.80, [100, 100, 200, 200])], tracker, 0.0
        )
        missing = process_frame_objects([], tracker, 0.25)
        third = process_frame_objects(
            [raw_object("stone", 0.82, [50, 100, 150, 200])], tracker, 0.5
        )

        self.assertEqual(missing["objects"], [])
        self.assertEqual(first["objects"][0]["track_id"], third["objects"][0]["track_id"])

    def test_event_statistics_use_tracks_not_only_peak_frame(self) -> None:
        tracker = LightweightTracker()
        first = process_frame_objects(
            [raw_object("stone", 0.80, [300, 0, 400, 100])], tracker, 0.0
        )["objects"]
        second = process_frame_objects(
            [raw_object("plastic", 0.90, [0, 0, 100, 100], 1)], tracker, 0.5
        )["objects"]
        events = merge_detection_events(
            [event_frame(0.0, "stone.jpg", first), event_frame(0.5, "plastic.jpg", second)],
            datetime(2026, 7, 15, 8, 0, 0),
            sample_fps=2.0,
            duration=2.0,
            event_silence_seconds=1.0,
        )

        self.assertEqual(events[0]["max_simultaneous_objects"], 1)
        self.assertEqual(events[0]["unique_object_count"], 2)
        self.assertEqual(events[0]["class_counts"], {"石块异物": 1, "塑料异物": 1})
        self.assertEqual(len(events[0]["key_frames"]), 2)

    def test_roi_parser(self) -> None:
        self.assertIsNone(parse_roi(""))
        self.assertEqual(parse_roi("100,50,1180,700"), (100, 50, 1180, 700))
        with self.assertRaises(ValueError):
            parse_roi("100,50,20,700")

    def test_existing_4fps_regression_splits_two_confirmed_waves(self) -> None:
        path = Path(
            "outputs/video_detections/"
            "20260715_085631_628913_boardcamera_iron_ore_canga/"
            "detection_results.json"
        )
        if not path.is_file():
            self.skipTest("历史4 FPS回归JSON不存在")
        data = json.loads(path.read_text(encoding="utf-8"))
        tracker = LightweightTracker()
        positive_frames = []
        processed_by_offset = {}
        for raw_frame in data["detection_frames"]:
            processed = process_frame_objects(
                raw_frame["objects"], tracker, float(raw_frame["offset_seconds"])
            )
            processed_by_offset[float(raw_frame["offset_seconds"])] = processed
            if processed["objects"]:
                positive_frames.append(
                    event_frame(
                        float(raw_frame["offset_seconds"]),
                        str(raw_frame["image"]),
                        processed["objects"],
                    )
                )
        events = merge_detection_events(
            positive_frames,
            datetime.fromisoformat(data["video_start_time"]),
            float(data["sample_fps"]),
            float(data["duration_seconds"]),
            event_silence_seconds=1.0,
        )

        self.assertEqual(len(events), 2)
        self.assertLess(events[0]["end_offset_seconds"], events[1]["start_offset_seconds"])
        self.assertEqual(processed_by_offset[14.252]["objects"], [])
        plastic_at_16006 = [
            item
            for item in processed_by_offset[16.006]["objects"]
            if item["predicted_class"] == "plastic"
        ]
        self.assertEqual(len(plastic_at_16006), 1)


if __name__ == "__main__":
    unittest.main()
