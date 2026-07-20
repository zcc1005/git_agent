from __future__ import annotations

import unittest
from datetime import datetime, timezone

from agent.temporal import resolve_temporal_expression


FIXED_NOW = datetime(2026, 7, 20, 2, 30, tzinfo=timezone.utc)


class TemporalResolutionTests(unittest.TestCase):
    def resolve(self, message: str):
        return resolve_temporal_expression(
            message,
            now=FIXED_NOW,
            timezone_name="Asia/Shanghai",
        )

    def test_today_and_yesterday_expand_to_local_calendar_days(self) -> None:
        today = self.resolve("查询今天的高风险记录")
        yesterday = self.resolve("生成昨天的风险报告")

        self.assertEqual(today["start_time"], "2026-07-20T00:00:00+08:00")
        self.assertEqual(today["end_time"], "2026-07-20T23:59:59+08:00")
        self.assertEqual(yesterday["start_time"], "2026-07-19T00:00:00+08:00")
        self.assertEqual(yesterday["end_time"], "2026-07-19T23:59:59+08:00")

    def test_today_morning_uses_the_declared_period_boundary(self) -> None:
        result = self.resolve("查询今天上午的记录")

        self.assertEqual(result["start_time"], "2026-07-20T00:00:00+08:00")
        self.assertEqual(result["end_time"], "2026-07-20T11:59:59+08:00")

    def test_clock_range_inherits_period_and_current_date(self) -> None:
        result = self.resolve("查询上午8点到9点的高风险记录")

        self.assertEqual(result["start_time"], "2026-07-20T08:00:00+08:00")
        self.assertEqual(result["end_time"], "2026-07-20T09:00:00+08:00")

    def test_recent_hour_is_a_rolling_interval(self) -> None:
        result = self.resolve("查看最近一小时的报警")

        self.assertEqual(result["start_time"], "2026-07-20T09:30:00+08:00")
        self.assertEqual(result["end_time"], "2026-07-20T10:30:00+08:00")

    def test_minute_ordinals_become_video_offsets(self) -> None:
        result = self.resolve("检测视频第10分钟到第20分钟")

        self.assertEqual(result["kind"], "offset")
        self.assertEqual(result["start_offset_seconds"], 600)
        self.assertEqual(result["end_offset_seconds"], 1200)

    def test_detection_clock_becomes_video_offsets(self) -> None:
        result = self.resolve("从00:30检测到01:20")

        self.assertEqual(result["kind"], "offset")
        self.assertEqual(result["start_offset_seconds"], 30)
        self.assertEqual(result["end_offset_seconds"], 80)


if __name__ == "__main__":
    unittest.main()
