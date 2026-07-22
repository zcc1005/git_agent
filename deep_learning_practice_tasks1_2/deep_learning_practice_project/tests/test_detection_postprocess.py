from __future__ import annotations

import unittest

import cv2
import numpy as np

from task2_yolo.detect_yolo import apply_stone_wood_arbitration
from task2_yolo.postprocess import (
    filter_duplicate_objects,
    filter_implausible_geometry,
)


def detection(class_name: str, confidence: float, bbox: list[float]) -> dict:
    return {
        "predicted_class_id": 0,
        "predicted_class": class_name,
        "predicted_class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": bbox,
    }


class DetectionPostprocessTests(unittest.TestCase):
    def test_contained_same_class_box_is_removed(self) -> None:
        kept, ignored = filter_duplicate_objects(
            [
                detection("wood", 0.55, [20, 20, 180, 100]),
                detection("wood", 0.42, [0, 0, 300, 200]),
            ]
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["confidence"], 0.55)
        self.assertEqual(len(ignored), 1)
        self.assertIn("containment", ignored[0]["filter_reason"])

    def test_adjacent_different_classes_are_retained(self) -> None:
        kept, ignored = filter_duplicate_objects(
            [
                detection("wood", 0.70, [0, 0, 100, 100]),
                detection("stone", 0.65, [90, 0, 190, 100]),
            ]
        )

        self.assertEqual(len(kept), 2)
        self.assertEqual(ignored, [])

    def test_huge_full_frame_box_is_filtered(self) -> None:
        kept, ignored = filter_implausible_geometry(
            [detection("wood", 0.45, [0, 0, 1000, 700])],
            image_width=1000,
            image_height=700,
        )

        self.assertEqual(kept, [])
        self.assertEqual(len(ignored), 1)
        self.assertIn("box_area_ratio", ignored[0]["filter_reason"])

    def test_large_object_near_one_edge_is_retained(self) -> None:
        kept, ignored = filter_implausible_geometry(
            [detection("wood", 0.70, [0, 150, 500, 500])],
            image_width=1000,
            image_height=700,
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(ignored, [])

    def test_elongated_brown_evidence_resolves_stone_wood_conflict(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(image, (30, 80), (270, 120), (30, 100, 180), -1)
        objects = [
            detection("stone", 0.70, [10, 30, 290, 170]),
            detection("wood", 0.60, [10, 30, 290, 170]),
        ]

        apply_stone_wood_arbitration(image, objects)
        kept, ignored = filter_duplicate_objects(objects)

        self.assertEqual(kept[0]["predicted_class"], "wood")
        self.assertEqual(kept[0]["class_arbitration"], "elongated_brown_component")
        self.assertEqual(ignored[0]["predicted_class"], "stone")


if __name__ == "__main__":
    unittest.main()
