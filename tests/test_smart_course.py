import unittest

import numpy as np
import cv2

from smart_course import (
    TinyVisionRouter,
    build_speed_plan,
    interpolate_sequence_position,
    parse_course_stamp,
    parse_times,
    rank_course_cards,
)


class SmartCourseTests(unittest.TestCase):
    def test_parse_common_time_formats(self):
        self.assertEqual(parse_times("10:09-10:55"), [609, 655])
        self.assertEqual(parse_times("下午 1405"), [845])
        self.assertEqual(parse_times("03-13 10:09-10:55"), [609, 655])

    def test_parse_wechat_card_stamp_with_touching_or_spaced_tokens(self):
        self.assertEqual(parse_course_stamp("09-1610:09-10:55")["start"], "10:09")
        parsed = parse_course_stamp("09-1411:04-11 50")
        self.assertEqual((parsed["date"], parsed["start"], parsed["end"]), ("09-14", "11:04", "11:50"))

    def test_parse_course_stamp_keeps_two_digit_day(self):
        """日期分组必须优先匹配两位，不能把 09-16 只吃成 09-1。"""
        for raw, date, start, end in (
            ("09-16 07:59-08:45", "09-16", "07:59", "08:45"),
            ("09-14 08:54-09:40", "09-14", "08:54", "09:40"),
            ("09-23 08:54-09:40", "09-23", "08:54", "09:40"),
            ("【第2周】 09-16 07:59-08:45", "09-16", "07:59", "08:45"),
            ("09-1610:09-10:55", "09-16", "10:09", "10:55"),
        ):
            parsed = parse_course_stamp(raw)
            self.assertEqual(
                (parsed["date"], parsed["start"], parsed["end"]),
                (date, start, end),
                msg=f"raw={raw!r}",
            )

    def test_cards_are_sorted_by_time_not_screen_position(self):
        items = [
            {"raw": "11:04", "text": "11:04", "cx": 200, "cy": 300, "w": 40, "h": 18, "conf": 91},
            {"raw": "08:30", "text": "08:30", "cx": 700, "cy": 300, "w": 40, "h": 18, "conf": 90},
            {"raw": "14:10", "text": "14:10", "cx": 450, "cy": 500, "w": 40, "h": 18, "conf": 93},
        ]
        cards = rank_course_cards(items, (0, 0, 1000, 800))
        self.assertEqual([card["time"] for card in cards], ["08:30", "11:04", "14:10"])
        self.assertEqual([card["lesson"] for card in cards], [1, 2, 3])

    def test_start_and_end_time_on_one_card_are_merged(self):
        items = [
            {"raw": "08:30", "text": "08:30", "cx": 210, "cy": 280, "w": 45, "h": 18, "conf": 92},
            {"raw": "09:15", "text": "09:15", "cx": 300, "cy": 280, "w": 45, "h": 18, "conf": 91},
            {"raw": "10:20", "text": "10:20", "cx": 650, "cy": 280, "w": 45, "h": 18, "conf": 90},
        ]
        cards = rank_course_cards(items, (0, 0, 1000, 600))
        self.assertEqual([card["time"] for card in cards], ["08:30", "10:20"])

    def test_date_like_compact_number_does_not_sort_course(self):
        items = [
            {"raw": "0313", "text": "0313", "cx": 210, "cy": 280, "w": 38, "h": 18, "conf": 90},
            {"raw": "10:09", "text": "10:09", "cx": 260, "cy": 280, "w": 45, "h": 18, "conf": 94},
            {"raw": "11:04", "text": "11:04", "cx": 650, "cy": 280, "w": 45, "h": 18, "conf": 93},
        ]
        cards = rank_course_cards(items, (0, 0, 1000, 600))
        self.assertEqual([card["time"] for card in cards], ["10:09", "11:04"])

    def test_auto_speed_returns_to_normal_near_end(self):
        self.assertEqual(build_speed_plan(2400, "auto", 120), [(0.0, 2.0), (2280.0, 1.0)])
        self.assertEqual(build_speed_plan(600, "1.5x"), [(0.0, 1.5)])

    def test_missing_week_position_is_interpolated(self):
        visible = {
            2: {"cx": 1731, "cy": 597},
            6: {"cx": 2426, "cy": 597},
        }
        x, y = interpolate_sequence_position(3, visible)
        self.assertTrue(1885 <= x <= 1920)
        self.assertEqual(y, 597)

    def test_router_returns_supported_route(self):
        image = np.full((180, 320, 3), 255, dtype=np.uint8)
        decision = TinyVisionRouter().predict(image)
        self.assertIn(decision.route, {"ocr", "multimodal"})
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)

    def test_router_sends_visual_noise_to_multimodal(self):
        image = np.random.default_rng(7).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        self.assertEqual(TinyVisionRouter().predict(image).route, "multimodal")

    def test_router_separates_diagram_from_text_slide(self):
        diagram = np.full((400, 700, 3), 255, dtype=np.uint8)
        cv2.rectangle(diagram, (50, 60), (340, 350), (50, 80, 120), -1)
        cv2.circle(diagram, (500, 200), 110, (20, 140, 200), -1)
        text_slide = np.full((400, 700, 3), 255, dtype=np.uint8)
        for index in range(7):
            cv2.putText(
                text_slide,
                f"This is text line {index}",
                (40, 50 + index * 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (20, 20, 20),
                2,
            )
        router = TinyVisionRouter()
        self.assertEqual(router.predict(diagram).route, "multimodal")
        self.assertEqual(router.predict(text_slide).route, "ocr")

    def test_text_rich_slide_is_not_overridden_by_frame_edges(self):
        image = np.full((400, 700, 3), 255, dtype=np.uint8)
        boxes = []
        for row in range(8):
            for column in range(6):
                x, y = 20 + column * 110, 30 + row * 44
                cv2.putText(image, "DNA", (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
                boxes.append((x, y, 65, 24))
        cv2.rectangle(image, (0, 0), (699, 399), (70, 70, 70), 5)
        router = TinyVisionRouter()
        decision = router.refine_with_text_regions(image, router.predict(image), boxes, 70)
        self.assertEqual(decision.route, "ocr")


if __name__ == "__main__":
    unittest.main()
