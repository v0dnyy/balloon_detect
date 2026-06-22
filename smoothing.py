"""
EMA (Exponential Moving Average) фильтр для сглаживания координат bbox.

Формула:
filtered = alpha * new_value + (1 - alpha) * prev_filtered
"""

from typing import Optional


class EMAFilter:
    """
    EMA-фильтр для 4 координат bbox: [x1, y1, x2, y2].

    Пример:
        filt = EMAFilter(alpha=0.35)
        smooth_box = filt.update([100, 200, 150, 250])
    """

    def __init__(self, alpha: float = 0.35):
        """
        Args:
            alpha: коэффициент сглаживания ∈ (0, 1].
                   Большее значение = быстрее реагирует, меньше сглаживает.
        """
        assert 0 < alpha <= 1.0, "alpha must be in (0, 1]"
        self.alpha = alpha
        self._state: Optional[list[float]] = None  # [x1, y1, x2, y2]

    def update(self, bbox: list[int | float]) -> list[float]:
        """
        Принимает сырой bbox [x1, y1, x2, y2], возвращает сглаженный.
        При первом вызове инициализирует состояние без сглаживания.
        """
        if self._state is None:
            self._state = [float(v) for v in bbox]
            return list(self._state)

        self._state = [
            self.alpha * new + (1.0 - self.alpha) * prev
            for new, prev in zip(bbox, self._state)
        ]
        return list(self._state)

    def reset(self) -> None:
        """Сбросить состояние фильтра (например, при потере трека)."""
        self._state = None

    @property
    def initialized(self) -> bool:
        return self._state is not None


class MultiTrackEMAFilter:
    """
    EMA-фильтр для нескольких треков одновременно.
    Автоматически создаёт и удаляет фильтры по track_id.

    Используется так:
        filt = MultiTrackEMAFilter(alpha=0.35)

        # каждый кадр:
        smooth_tracks = filt.update(tracks)  # tracks: список dict с "track_id" и "bbox"
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._filters: dict[int, EMAFilter] = {}

    def update(self, tracks: list[dict]) -> list[dict]:
        """
        Принимает список треков, возвращает копию со сглаженными bbox.
        Треки без track_id пропускаются без сглаживания.
        """
        current_ids = {t["track_id"] for t in tracks if "track_id" in t}

        # Удаляем фильтры для исчезнувших треков
        for tid in list(self._filters.keys()):
            if tid not in current_ids:
                del self._filters[tid]

        smoothed = []
        for track in tracks:
            tid = track.get("track_id")
            if tid is None:
                smoothed.append(track)
                continue

            if tid not in self._filters:
                self._filters[tid] = EMAFilter(alpha=self.alpha)

            smooth_bbox = self._filters[tid].update(track["bbox"])
            smoothed.append({**track, "bbox": smooth_bbox})

        return smoothed
