import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import ppt_extractor as app


class AutomaticWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.saved = {
            key: app.state[key]
            for key in ("playing", "playing_all", "monitoring", "region")
        }
        app.state["playing"] = False
        app.state["playing_all"] = False
        app.state["monitoring"] = False
        app.state["region"] = (10, 20, 800, 450)

    def tearDown(self):
        app.state.update(self.saved)

    def test_more_than_two_courses_are_dispatched_in_ranked_order(self):
        cards = [
            {"lesson": 1, "time": "08:30", "cx": 100, "cy": 200},
            {"lesson": 2, "time": "10:10", "cx": 350, "cy": 200},
            {"lesson": 3, "time": "14:00", "cx": 600, "cy": 200},
        ]
        with (
            patch.object(app, "log"),
            patch.object(app, "_stop_monitoring"),
            patch.object(app, "_scan_completed_week_lessons", return_value=(set(), set())),
            patch.object(app, "_discover_week_course_cards", return_value=cards),
            patch.object(app, "_run_simple_course", return_value=True) as run_course,
        ):
            app.run_simple_auto_courses([5], 1200, skip_completed=False)

        self.assertEqual([call.args[1] for call in run_course.call_args_list], [1, 2, 3])
        self.assertTrue(run_course.call_args_list[0].kwargs["dialog_ready"])
        self.assertFalse(run_course.call_args_list[1].kwargs["dialog_ready"])
        self.assertTrue(all(call.kwargs["week_selected_hint"] for call in run_course.call_args_list))
        self.assertFalse(app.state["playing"])
        self.assertFalse(app.state["playing_all"])

    def test_card_duration_toggle_is_passed_through(self):
        """关闭“按卡片时长监测”时，必须把开关传给每一节课程。"""
        cards = [{"lesson": 1, "time": "08:30", "cx": 100, "cy": 200}]
        with (
            patch.object(app, "log"),
            patch.object(app, "_stop_monitoring"),
            patch.object(app, "_scan_completed_week_lessons", return_value=(set(), set())),
            patch.object(app, "_discover_week_course_cards", return_value=cards),
            patch.object(app, "_run_simple_course", return_value=True) as run_course,
        ):
            app.run_simple_auto_courses([5], 1200, skip_completed=True, use_card_duration=False)
        self.assertFalse(run_course.call_args_list[0].kwargs["use_card_duration"])

    def test_change_detector_uses_stable_frames_and_detects_full_page_jump(self):
        detector = app.ChangeDetector()
        first = np.full((1080, 1920, 3), 245, dtype=np.uint8)
        second = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        with patch.object(app, "log"):
            detector.detect(first)
            changed, score, status = detector.detect(second)
        self.assertTrue(changed)
        self.assertGreater(score, 90)
        self.assertEqual(status, "page_jump")

    def test_wechat_speed_labels_are_parsed(self):
        self.assertEqual(app._parse_speed_value("1.0x"), 1.0)
        self.assertEqual(app._parse_speed_value("2.0×"), 2.0)
        self.assertEqual(app._parse_speed_value("1.5倍"), 1.5)
        with patch.object(
            app,
            "_ocr_items",
            return_value=[{"raw": "1.0x", "conf": 91, "cx": 815, "cy": 725}],
        ):
            self.assertEqual(app._find_speed_value(region=(500, 600, 400, 250)), (1.0, (815, 725)))

    def test_week_buttons_include_selected_and_unselected_tabs(self):
        width, height = 940, 1540
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        for index, week in enumerate((1, 2)):
            left = 48 + index * 173
            cv2.rectangle(image, (left, 555), (left + 150, 633), (100, 100, 100), 2)
            cv2.putText(image, str(week), (left + 66, 605), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        def grab(region):
            x, y, w, h = region
            crop = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            return Image.fromarray(crop), x, y

        with patch.object(app, "_grab_ocr_image", side_effect=grab):
            buttons = app._visible_week_buttons((0, 0, width, height))
        self.assertEqual(sorted(buttons), [1, 2])
        self.assertTrue(270 <= buttons[2][0] <= 320)

    def test_week_buttons_parse_full_label_of_selected_tab(self):
        """选中态蓝色按钮整块 OCR 为“第1周”，不能只取数字而误读成 51。"""
        width, height = 940, 1540
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        for index in range(2):
            left = 48 + index * 173
            cv2.rectangle(image, (left, 555), (left + 150, 633), (100, 100, 100), 2)

        def grab(region):
            x, y, w, h = region
            crop = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            return Image.fromarray(crop), x, y

        with (
            patch.object(app, "_grab_ocr_image", side_effect=grab),
            patch.object(app.pytesseract, "image_to_string", side_effect=["第1周", "第2周"]),
        ):
            buttons = app._visible_week_buttons((0, 0, width, height))
        self.assertEqual(sorted(buttons), [1, 2])

    def test_week_selection_confirms_by_highlight_when_title_is_stale(self):
        """周标签切换后弹窗标题不刷新，应以按钮高亮确认选中。"""
        with (
            patch.object(app, "_locked_window_region", return_value=(0, 0, 940, 1540)),
            patch.object(app, "_visible_week_buttons", return_value={1: (100, 600), 2: (280, 600)}),
            patch.object(app, "_check_blue_at", side_effect=[False, True]),
            patch.object(app, "_find_week_tab", return_value=(280, 600)),
            patch.object(app, "_current_switch_week", return_value=3),
            patch.object(app.time, "sleep"),
            patch.object(app, "log"),
        ):
            self.assertTrue(app._ensure_week_selected(2))

    def test_week_selection_skips_click_when_target_already_highlighted(self):
        with (
            patch.object(app, "_locked_window_region", return_value=(0, 0, 940, 1540)),
            patch.object(app, "_visible_week_buttons", return_value={1: (100, 600), 2: (280, 600)}),
            patch.object(app, "_check_blue_at", return_value=True),
            patch.object(app, "_find_week_tab") as find,
            patch.object(app.time, "sleep"),
            patch.object(app, "log"),
        ):
            self.assertTrue(app._ensure_week_selected(1))
            find.assert_not_called()

    def test_open_switch_dialog_reclicks_when_title_seen_but_week_tabs_missing(self):
        """弹窗关闭动画期间标题仍可被 OCR 到，不能因此跳过“切换节次”点击。"""
        with (
            patch.object(app, "_switch_dialog_open", return_value=True),
            patch.object(app, "_switch_dialog_have_week_tabs", return_value={}),
            patch.object(app, "_uia_invoke_by_names", return_value=False),
            patch.object(app, "_switch_button_region", return_value=(0, 0, 200, 100)),
            patch.object(app, "_find_text_target", return_value=(10, 20)) as find,
            patch.object(app, "_strong_click", return_value=True) as click,
            patch.object(app.time, "sleep"),
            patch.object(app, "log"),
        ):
            self.assertTrue(app._open_switch_dialog())
        find.assert_called_once()
        self.assertEqual(click.call_args_list[0].args, (10, 20))

    def test_open_switch_dialog_skips_click_when_week_tabs_visible(self):
        with (
            patch.object(app, "_switch_dialog_open", return_value=True),
            patch.object(app, "_switch_dialog_have_week_tabs", return_value={1: (100, 600)}),
            patch.object(app, "_find_text_target") as find,
            patch.object(app, "_strong_click") as click,
            patch.object(app, "log"),
        ):
            self.assertTrue(app._open_switch_dialog())
        find.assert_not_called()
        click.assert_not_called()

    def test_week_completion_keeps_dialog_open_for_next_week(self):
        """一周全部跳过时应保留弹窗复用，而不是关闭→重开（重开容易被吞点击）。"""
        cards = [{"lesson": 1, "time": "08:30", "cx": 100, "cy": 200}]
        with (
            patch.object(app, "log"),
            patch.object(app, "_stop_monitoring"),
            patch.object(app, "_scan_completed_week_lessons", return_value=({(1, 1)}, set())),
            patch.object(app, "_discover_week_course_cards", return_value=cards),
            patch.object(app, "_run_simple_course", return_value=True) as run_course,
            patch.object(app, "_switch_dialog_have_week_tabs", return_value={}),
            patch.object(app, "_close_switch_dialog", return_value=True) as close_dialog,
        ):
            app.run_simple_auto_courses([1], 1200, skip_completed=True)
        run_course.assert_not_called()
        close_dialog.assert_not_called()

    def test_switch_dialog_closed_once_after_all_weeks(self):
        cards = [{"lesson": 1, "time": "08:30", "cx": 100, "cy": 200}]
        with (
            patch.object(app, "log"),
            patch.object(app, "_stop_monitoring"),
            patch.object(app, "_scan_completed_week_lessons", return_value=({(1, 1)}, set())),
            patch.object(app, "_discover_week_course_cards", return_value=cards),
            patch.object(app, "_run_simple_course", return_value=True),
            patch.object(app, "_switch_dialog_have_week_tabs", return_value={1: (100, 600)}),
            patch.object(app, "_switch_dialog_open", return_value=True),
            patch.object(app, "_close_switch_dialog", return_value=True) as close_dialog,
        ):
            app.run_simple_auto_courses([1], 1200, skip_completed=True)
        close_dialog.assert_called_once()

    def test_speed_menu_selects_and_verifies_2x(self):
        old_speed = app.state["current_speed"]
        try:
            app.state["current_speed"] = 1.0
            with (
                patch.dict(app._config, {"interaction_mode": "uia"}),
                patch.object(app, "_locked_window_region", return_value=(0, 0, 940, 1540)),
                patch.object(app, "_find_speed_value", side_effect=[
                    (1.0, (840, 735)),
                    (2.0, (820, 1150)),
                    (2.0, (840, 735)),
                ]),
                patch.object(app, "_strong_click", return_value=True) as click,
                patch.object(app.time, "sleep"),
                patch.object(app, "log"),
            ):
                self.assertTrue(app._apply_playback_speed_unpaused(2.0))
            self.assertEqual([call.args for call in click.call_args_list], [(840, 735), (820, 1150)])
            self.assertEqual(app.state["current_speed"], 2.0)
        finally:
            app.state["current_speed"] = old_speed

    def test_window_capture_prefers_graphics_capture_frame(self):
        frame = np.full((120, 200, 3), 160, dtype=np.uint8)
        with patch.object(app._graphics_capture, "grab", return_value=frame):
            result = app.capture_window(12345)
        self.assertTrue(np.array_equal(result, frame))

    def test_graphics_capture_keeps_first_immediate_frame(self):
        frame = np.full((30, 40, 4), 180, dtype=np.uint8)

        class ImmediateCapture:
            def __init__(self, **_kwargs):
                self.handlers = {}

            def event(self, callback):
                self.handlers[callback.__name__] = callback
                return callback

            def start_free_threaded(self):
                self.handlers["on_frame_arrived"](type("Frame", (), {"frame_buffer": frame})(), None)
                return type("Control", (), {"stop": lambda self: None})()

        backend = app._GraphicsCaptureBackend()
        with patch.object(app, "WindowsCapture", ImmediateCapture):
            result = backend.grab(12345, wait_seconds=.01)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (30, 40, 3))
        backend.stop()

    def test_black_video_with_watermarks_is_rejected(self):
        frame = np.zeros((478, 847, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (846, 6), (240, 70, 60), -1)
        cv2.putText(frame, "WATERMARK", (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 70), 2)
        self.assertTrue(app._is_black_frame(frame))
        slide = np.full((478, 847, 3), 245, dtype=np.uint8)
        self.assertFalse(app._is_black_frame(slide))

    def test_course_title_crop_stays_inside_purple_header(self):
        with (
            patch.object(app, "_locked_window_region", return_value=(100, 200, 940, 1540)),
            patch.object(app, "_grab_ocr_image", return_value=(Image.new("RGB", (282, 77), "white"), 0, 0)) as grab,
            patch.object(app.pytesseract, "image_to_string", return_value="分 子 生 物 学\n"),
            patch.object(app, "log"),
        ):
            self.assertEqual(app._detect_course_name(), "分子生物学")
        x, y, w, h = grab.call_args.args[0]
        self.assertEqual((x, y), (100 + int(940 * .35), 200 + int(1540 * .04)))
        self.assertLess(y + h, 200 + int(1540 * .10))


if __name__ == "__main__":
    unittest.main()
