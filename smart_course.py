"""Lightweight visual routing and course-card ordering helpers.

The module intentionally has no GUI or Windows dependencies so its behaviour can
be tested independently from the desktop application.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


_TIME_RE = re.compile(
    r"(?<!\d)(?P<hour>[01]?\d|2[0-3])\s*[:：.·]\s*(?P<minute>[0-5]\d)(?!\d)"
)
_COMPACT_TIME_RE = re.compile(r"(?<!\d)(?P<hour>[01]\d|2[0-3])(?P<minute>[0-5]\d)(?!\d)")


def parse_times(text: str) -> list[int]:
    """Return unique HH:MM values as minutes since midnight."""
    raw = str(text or "")
    matches: list[tuple[int, int]] = []
    for pattern in (_TIME_RE, _COMPACT_TIME_RE):
        for match in pattern.finditer(raw):
            value = int(match.group("hour")) * 60 + int(match.group("minute"))
            matches.append((match.start(), value))
    values: list[int] = []
    for _, value in sorted(matches):
        if value not in values:
            values.append(value)
    return values


def format_minutes(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def parse_course_stamp(text: str) -> dict | None:
    """Extract card date and start/end time from noisy OCR text."""
    raw = str(text or "")
    normalized = raw.replace("—", "-").replace("–", "-")
    # 注意分组顺序：必须让 `1[0-2]`/`[12]\d`/`3[01]` 先匹配。
    # 否则交替式 `0?[1-9]` 会抢先把 “16” 只吃成 “1”，导致 09-16 被解析成 09-01。
    date_match = re.search(r"(?:^|[^\d])(1[0-2]|0?[1-9])\s*-\s*(3[01]|[12]\d|0?[1-9])", normalized)
    time_text = normalized
    if date_match:
        start, end = date_match.span()
        time_text = normalized[:start] + " " + normalized[end:]
    # 时间已有冒号时（如 07:59-08:45），中间的连字符是“起-止”分隔符，
    # 不能再当成“HH-MM”转换，故 lookbehind 同时排除冒号。
    time_text = re.sub(r"(?<![\d:])([0-2]?\d)-([0-5]\d)(?!\d)", r"\1:\2", time_text)
    time_text = re.sub(r"(?<![\d:])([0-2]?\d)\s+([0-5]\d)(?!\d)", r"\1:\2", time_text)
    times = parse_times(time_text)
    if not date_match and not times:
        return None
    month = int(date_match.group(1)) if date_match else 99
    day = int(date_match.group(2)) if date_match else 99
    plausible = [value for value in times if value >= 6 * 60]
    usable = plausible or times
    start = usable[0] if usable else 24 * 60
    end = usable[1] if len(usable) > 1 else None
    return {
        "month": month,
        "day": day,
        "start_minute": start,
        "end_minute": end,
        "date": f"{month:02d}-{day:02d}" if date_match else "",
        "start": format_minutes(start) if start < 24 * 60 else "",
        "end": format_minutes(end) if end is not None else "",
    }


def interpolate_sequence_position(target: int, points: dict[int, dict]) -> tuple[int, int] | None:
    """Infer a missing horizontal item from numbered neighbours.

    Returns ``None`` when the observations do not resemble a stable left-to-right
    sequence, preventing a speculative click on irregular layouts.
    """
    usable = sorted(
        (int(number), float(item["cx"]), float(item["cy"]))
        for number, item in points.items()
        if "cx" in item and "cy" in item
    )
    if len(usable) < 2 or not (usable[0][0] < int(target) < usable[-1][0]):
        return None
    numbers = np.asarray([item[0] for item in usable], dtype=np.float64)
    xs = np.asarray([item[1] for item in usable], dtype=np.float64)
    if np.any(np.diff(xs) <= 0):
        return None
    slope, intercept = np.polyfit(numbers, xs, 1)
    if not 35.0 <= slope <= 280.0:
        return None
    predicted = slope * int(target) + intercept
    if len(usable) >= 3:
        residual = float(np.max(np.abs(xs - (slope * numbers + intercept))))
        if residual > max(24.0, slope * 0.28):
            return None
    if not xs.min() - 8 <= predicted <= xs.max() + 8:
        return None
    return int(round(predicted)), int(round(float(np.median([item[2] for item in usable]))))


def _line_groups(items: Iterable[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for item in sorted(items, key=lambda part: (int(part.get("cy", 0)), int(part.get("cx", 0)))):
        cy = int(item.get("cy", 0))
        height = max(8, int(item.get("h", 12)))
        line = next(
            (
                existing
                for existing in lines
                if abs(cy - int(np.median([int(part.get("cy", 0)) for part in existing])))
                <= max(14, height)
            ),
            None,
        )
        if line is None:
            line = []
            lines.append(line)
        line.append(item)
    for line in lines:
        line.sort(key=lambda part: int(part.get("cx", 0)))
    return lines


def rank_course_cards(items: Iterable[dict], region: tuple[int, int, int, int] | None = None) -> list[dict]:
    """Find visible course times and sort cards chronologically.

    Tesseract may return ``10:09`` as one token or as several adjacent tokens.
    We therefore inspect individual tokens and short same-line token windows,
    then merge the start/end times that belong to the same visual card.
    """
    source = [dict(item) for item in items]
    candidates: list[dict] = []

    for line in _line_groups(source):
        for start in range(len(line)):
            for size in range(1, min(4, len(line) - start) + 1):
                window = line[start : start + size]
                if len(window) > 1:
                    gaps = [
                        int(right.get("cx", 0)) - int(right.get("w", 0)) // 2
                        - (int(left.get("cx", 0)) + int(left.get("w", 0)) // 2)
                        for left, right in zip(window, window[1:])
                    ]
                    if any(gap > 45 for gap in gaps):
                        continue
                text = "".join(str(part.get("raw", part.get("text", ""))) for part in window)
                times = parse_times(text)
                if not times:
                    continue
                confidence = float(np.mean([float(part.get("conf", 0)) for part in window]))
                cx = int(np.mean([int(part.get("cx", 0)) for part in window]))
                cy = int(np.mean([int(part.get("cy", 0)) for part in window]))
                for minute in times:
                    candidates.append(
                        {"minute": minute, "time": format_minutes(minute), "cx": cx, "cy": cy, "conf": confidence}
                    )

    # Prefer high-confidence, compact matches and discard duplicate n-gram hits.
    unique: list[dict] = []
    for item in sorted(candidates, key=lambda part: (-part["conf"], part["minute"])):
        duplicate = next(
            (
                old
                for old in unique
                if old["minute"] == item["minute"]
                and abs(old["cx"] - item["cx"]) <= 80
                and abs(old["cy"] - item["cy"]) <= 35
            ),
            None,
        )
        if duplicate is None:
            unique.append(item)

    # A card often contains both start and end time. Nearby times are one card;
    # its earliest value is the chronological key.
    clusters: list[list[dict]] = []
    for item in sorted(unique, key=lambda part: (part["cy"], part["cx"], part["minute"])):
        cluster = next(
            (
                group
                for group in clusters
                if abs(int(np.mean([entry["cx"] for entry in group])) - item["cx"]) <= 150
                and abs(int(np.mean([entry["cy"] for entry in group])) - item["cy"]) <= 55
            ),
            None,
        )
        if cluster is None:
            cluster = []
            clusters.append(cluster)
        cluster.append(item)

    cards: list[dict] = []
    for group in clusters:
        # 卡片通常同时含日期和起止时间。存在正常上课时间时，排除被 OCR
        # 压成 0313 一类的日期数字，避免把 03-13 当作凌晨 03:13。
        plausible_times = [entry for entry in group if entry["minute"] >= 6 * 60]
        start = min(plausible_times or group, key=lambda entry: entry["minute"])
        cx = int(np.mean([entry["cx"] for entry in group]))
        cy = int(np.mean([entry["cy"] for entry in group]))
        cards.append(
            {
                "minute": start["minute"],
                "time": start["time"],
                "cx": cx,
                "cy": cy,
                "conf": max(entry["conf"] for entry in group),
            }
        )

    # Keep points inside the supplied card region and use visual position only
    # as a deterministic tie-breaker.
    if region:
        x, y, w, h = region
        cards = [card for card in cards if x <= card["cx"] <= x + w and y <= card["cy"] <= y + h]
    cards.sort(key=lambda card: (card["minute"], card["cy"], card["cx"]))
    for index, card in enumerate(cards, start=1):
        card["lesson"] = index
    return cards


@dataclass(frozen=True)
class RouteDecision:
    route: str
    confidence: float
    score: float
    features: dict[str, float]
    reason: str


class TinyVisionRouter:
    """A tiny embedded logistic model choosing OCR or multimodal recognition.

    It uses five inexpensive visual features and fixed, versioned weights.  The
    model only routes work; it deliberately does not pretend to be a multimodal
    recognizer itself.
    """

    VERSION = "tiny-router-v3"
    _BIAS = -0.65
    _WEIGHTS = {
        "edge_density": 2.10,
        "colorfulness": 1.15,
        "entropy": 1.05,
        "text_component_ratio": -2.45,
        "line_structure": -1.35,
        "largest_component_ratio": 3.00,
        "ink_coverage": 1.20,
    }

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-30.0, min(30.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def features(self, image: np.ndarray) -> dict[str, float]:
        if image is None or image.size == 0:
            return {key: 0.0 for key in self._WEIGHTS}
        frame = image
        if frame.shape[1] > 480:
            scale = 480.0 / frame.shape[1]
            frame = cv2.resize(frame, (480, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        edges = cv2.Canny(gray, 70, 160)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        if frame.ndim == 3:
            b, g, r = cv2.split(frame.astype(np.float32))
            rg = np.abs(r - g)
            yb = np.abs(0.5 * (r + g) - b)
            colorfulness = min(1.0, float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)) / 90.0)
        else:
            colorfulness = 0.0

        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
        probabilities = hist / max(1.0, float(hist.sum()))
        entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-9))) / 5.0

        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        plausible = 0
        for idx in range(1, count):
            _, _, width, height, area = stats[idx]
            if 2 <= width <= 55 and 5 <= height <= 55 and 8 <= area <= 1600 and 0.08 <= width / max(height, 1) <= 3.5:
                plausible += 1
        text_component_ratio = min(1.0, plausible / max(12.0, gray.size / 4500.0))
        component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([0])
        largest_component_ratio = min(
            1.0,
            float(component_areas.max(initial=0)) / max(1.0, gray.size * 0.35),
        )
        ink_coverage = min(1.0, float(np.count_nonzero(binary)) / max(1.0, gray.size * 0.45))

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1)))
        line_structure = min(1.0, float(np.count_nonzero(horizontal)) / max(1.0, gray.size * 0.08))
        return {
            "edge_density": min(1.0, edge_density / 0.22),
            "colorfulness": colorfulness,
            "entropy": min(1.0, entropy),
            "text_component_ratio": text_component_ratio,
            "line_structure": line_structure,
            "largest_component_ratio": largest_component_ratio,
            "ink_coverage": ink_coverage,
        }

    def predict(self, image: np.ndarray) -> RouteDecision:
        features = self.features(image)
        score = self._BIAS + sum(self._WEIGHTS[name] * features[name] for name in self._WEIGHTS)
        probability = self._sigmoid(score)
        route = "multimodal" if probability >= 0.55 else "ocr"
        confidence = probability if route == "multimodal" else 1.0 - probability
        if route == "ocr":
            reason = "文字结构明显，优先使用本地 OCR"
        else:
            reason = "图形/颜色信息较多，建议多模态识别并保留 OCR 兜底"
        return RouteDecision(route, confidence, score, features, reason)

    def refine_with_text_regions(
        self,
        image: np.ndarray,
        decision: RouteDecision,
        text_boxes: list[tuple[int, int, int, int]],
        mean_ocr_confidence: float = 0.0,
    ) -> RouteDecision:
        """Use OCR boxes to measure diagram structure outside readable text."""
        if image is None or image.size == 0:
            return decision
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 70, 160)
        text_mask = np.zeros(gray.shape, dtype=np.uint8)
        for x, y, width, height in text_boxes:
            padding = max(2, int(height * 0.35))
            cv2.rectangle(
                text_mask,
                (max(0, x - padding), max(0, y - padding)),
                (min(gray.shape[1] - 1, x + width + padding), min(gray.shape[0] - 1, y + height + padding)),
                255,
                -1,
            )
        edge_count = max(1, int(np.count_nonzero(edges)))
        nontext_edge_ratio = float(np.count_nonzero((edges > 0) & (text_mask == 0))) / edge_count
        text_coverage = float(np.count_nonzero(text_mask)) / max(1, text_mask.size)
        features = dict(decision.features)
        features.update({
            "nontext_edge_ratio": nontext_edge_ratio,
            "ocr_text_coverage": text_coverage,
            "ocr_token_count": float(len(text_boxes)),
            "ocr_mean_confidence": float(mean_ocr_confidence) / 100.0,
        })

        # 课件边框、缩略图和中文笔画常被计作“文字框外边缘”。足量、可信的
        # OCR 内容应优先于这个间接指标，否则纯文字 PPT 也会全部进多模态。
        text_dominant = (
            len(text_boxes) >= 35
            and text_coverage >= 0.10
            and mean_ocr_confidence >= 45
            and features.get("text_component_ratio", 0) >= 0.45
        )
        if text_dominant:
            return RouteDecision(
                "ocr",
                min(0.97, 0.62 + text_coverage + mean_ocr_confidence / 500.0),
                decision.score,
                features,
                "可读取文字占主导，优先本地 OCR",
            )

        diagram_evidence = (
            nontext_edge_ratio >= 0.30
            or (
                nontext_edge_ratio >= 0.27
                and (
                    features.get("largest_component_ratio", 0) >= 0.25
                    or features.get("line_structure", 0) <= 0.20
                    or features.get("ink_coverage", 0) >= 0.48
                )
            )
            or (
                features.get("edge_density", 0) >= 0.42
                and features.get("ink_coverage", 0) >= 0.20
                and features.get("entropy", 0) >= 0.32
            )
        )
        sparse_text_visual = len(text_boxes) < 12 and nontext_edge_ratio >= 0.38
        if diagram_evidence or sparse_text_visual or decision.route == "multimodal":
            confidence = max(
                decision.confidence if decision.route == "multimodal" else 0.0,
                min(0.96, 0.58 + max(0.0, nontext_edge_ratio - 0.27) * 2.2),
            )
            return RouteDecision(
                "multimodal",
                confidence,
                decision.score,
                features,
                "文字区域外仍有明显图形结构，需要多模态理解",
            )
        return RouteDecision(
            "ocr",
            max(decision.confidence, min(0.95, 0.55 + text_coverage)),
            decision.score,
            features,
            "主要信息位于可读取文字区域，本地 OCR 足够",
        )


def build_speed_plan(source_duration: float, mode: str = "auto", final_normal_seconds: float = 90.0) -> list[tuple[float, float]]:
    """Return ``(source-second, speed)`` transitions for a course."""
    duration = max(0.0, float(source_duration))
    normalized = str(mode or "auto").lower().replace("×", "x").strip()
    if normalized != "auto":
        match = re.search(r"\d+(?:\.\d+)?", normalized)
        speed = float(match.group()) if match else 1.0
        return [(0.0, max(0.5, min(3.0, speed)))]

    if duration >= 30 * 60:
        speed = 2.0
    elif duration >= 15 * 60:
        speed = 1.5
    elif duration >= 8 * 60:
        speed = 1.25
    else:
        speed = 1.0
    plan = [(0.0, speed)]
    tail = max(0.0, min(float(final_normal_seconds), duration))
    if speed > 1.0 and tail > 0 and duration - tail > 0:
        plan.append((duration - tail, 1.0))
    return plan
