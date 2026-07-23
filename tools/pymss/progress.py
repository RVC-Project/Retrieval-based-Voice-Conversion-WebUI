import math
from time import time

from tqdm.auto import tqdm


def _format_progress_time(value):
    """Format progress seconds as mm:ss or hh:mm:ss."""
    seconds = max(0, int(round(value or 0)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def _is_time_progress_message(message):
    """Return whether a progress message represents audio seconds."""
    return "audio" in str(message).lower()


def _format_progress_rtf(bar):
    """Return real-time factor text for a tqdm audio progress bar."""
    done = float(bar.n or 0)
    if done <= 0:
        return "--"
    elapsed = max(0.0, time() - getattr(bar, "start_t", time()))
    return f"{elapsed / done:.2f}"


def _update_time_progress_bar(bar):
    """Update a tqdm bar with formatted elapsed/total audio time."""
    if bar is None:
        return
    rtf = _format_progress_rtf(bar)
    bar._pymss_audio_progress = f"{_format_progress_time(bar.n)}/{_format_progress_time(bar.total)}, RTF={rtf}"
    bar.refresh()


class _TimeProgressTqdm(tqdm):
    """tqdm subclass with a custom audio progress field."""

    @property
    def format_dict(self):
        data = super().format_dict
        data["audio_progress"] = getattr(self, "_pymss_audio_progress", "")
        return data


class _ProgressContext:
    """Small progress adapter used by demixing helpers."""

    def __init__(
        self,
        pbar=False,
        total=1,
        callback=None,
        done=0,
        message="Processing audio",
        sample_rate=None,
    ):
        """Initialize the progress adapter.

        Args:
            pbar (Any, optional): Whether to show a tqdm progress bar.
                Defaults to False.
            total (Any, optional): Total progress units. Defaults to 1.
            callback (Any, optional): Optional callback receiving
                ``(done, total, message)``. Defaults to None.
            done (Any, optional): Initial completed units. Defaults to 0.
            message (str, optional): Progress message. Defaults to
                ``"Processing audio"``.
            sample_rate (int | None, optional): Sample rate used to expose
                progress in seconds. When omitted, progress values are used as
                already provided.
        """
        self.enabled = bool(pbar or callback)
        self.bar = None
        self.callback = callback
        self.done = done
        self.total = total
        self.message = message
        self.sample_rate = int(sample_rate or 0)
        if not self.enabled:
            return
        self.total = int(self.total or 1)
        self.done = min(max(0, int(self.done or 0)), self.total)
        if pbar:
            bar_kwargs = {"total": self._display_total(), "desc": message, "leave": False}
            if self.sample_rate > 0:
                bar_kwargs.update({"unit": "", "bar_format": "{l_bar}{bar}| {audio_progress}"})
                self.bar = _TimeProgressTqdm(**bar_kwargs)
            else:
                self.bar = tqdm(**bar_kwargs)
            if self.sample_rate > 0:
                _update_time_progress_bar(self.bar)
        if self.bar is not None and self.done:
            self.bar.update(self._display_value(self.done))
            if self.sample_rate > 0:
                _update_time_progress_bar(self.bar)
        self.emit()

    def _display_value(self, value):
        """Return progress value exposed to callbacks and progress bars."""
        if self.sample_rate <= 0:
            return int(value)
        if int(value) >= self.total:
            return self._display_total()
        return min(self._display_total(), int(int(value) // self.sample_rate))

    def _display_total(self):
        """Return total seconds for the current progress unit."""
        if self.sample_rate <= 0:
            return self.total
        return max(1, int(math.ceil(self.total / self.sample_rate)))

    def emit(self, done=None):
        """Emit a progress update."""
        if not self.enabled:
            return
        if done is not None:
            next_done = min(max(0, int(done)), self.total)
            if self.bar is not None:
                self.bar.update(self._display_value(next_done) - self._display_value(self.done))
                if self.sample_rate > 0:
                    _update_time_progress_bar(self.bar)
            self.done = next_done
        if self.callback is None:
            return
        self.callback(self._display_value(self.done), self._display_total(), self.message)

    def update(self, amount):
        """Advance progress by ``amount`` internal units."""
        if not self.enabled:
            return
        amount = int(amount)
        self.emit(self.done + amount)

    def close(self):
        """Close the progress bar when present."""
        if not self.enabled:
            return
        if self.bar:
            self.bar.close()


class _CliInferenceProgress:
    """CLI callback adapter for inference progress updates."""

    def __init__(self):
        self._bar = None
        self._message = None
        self._total = None

    def __call__(self, done, total, message):
        total = max(1, int(total or 1))
        done = max(0, min(int(done), total))
        if self._bar is None or self._message != message or self._total != total or done < self._bar.n:
            self.close()
            self._message = message
            self._total = total
            bar_kwargs = {"total": total, "desc": message, "leave": False, "mininterval": 0, "miniters": 1}
            if _is_time_progress_message(message):
                bar_kwargs.update({"unit": "", "bar_format": "{l_bar}{bar}| {audio_progress}"})
                self._bar = _TimeProgressTqdm(**bar_kwargs)
            else:
                self._bar = tqdm(**bar_kwargs)
            if _is_time_progress_message(message):
                _update_time_progress_bar(self._bar)
        if done != self._bar.n:
            self._bar.update(done - self._bar.n)
            if _is_time_progress_message(message):
                _update_time_progress_bar(self._bar)

    def close(self):
        """Close the active CLI progress bar."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None
