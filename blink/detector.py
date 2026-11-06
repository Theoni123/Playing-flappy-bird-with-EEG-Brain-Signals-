"""Real-time blink and head-nod detection."""

from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

import config


def make_bandpass(fs, band, order=config.FILTER_ORDER):
    return butter(order, band, btype="bandpass", fs=fs, output="sos")


def rolling_p2p(signal, window):
    """Peak-to-peak of the last `window` samples, sample by sample."""
    if signal.size == 0:
        return signal
    if signal.size < window:
        span = float(np.ptp(signal)) if signal.size else 0.0
        return np.full(signal.shape, span)
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(signal, (window - 1, 0), mode="edge")
    view = sliding_window_view(padded, window)
    return view.max(axis=1) - view.min(axis=1)


class BlinkDetector:
    """Band-pass, then trigger on a short-window peak-to-peak envelope.

    Peak-to-peak ignores blink polarity and is much less sensitive to the
    slow electrode drift that drowned the old threshold.
    """

    def __init__(
        self,
        fs=config.SAMPLE_RATE,
        threshold=config.DEFAULT_THRESHOLD_UV,
        polarity=config.POLARITY,
        refractory_s=config.REFRACTORY_S,
        band=config.BANDPASS_HZ,
        rearm_fraction=config.REARM_FRACTION,
        p2p_window_s=config.P2P_WINDOW_S,
    ):
        self.fs = fs
        self.threshold = threshold
        self.sign = 1.0  # envelope is always positive
        self.refractory = int(refractory_s * fs)
        self.rearm_fraction = rearm_fraction
        self._sos = make_bandpass(fs, band)
        self._zi = None
        self._win = deque(maxlen=max(8, int(p2p_window_s * fs)))
        self._armed = True
        self._since_trigger = self.refractory
        self._warmup = int(1.0 * fs)

    def set_polarity(self, polarity):
        del polarity

    def filter(self, signal, extra=None):
        """Band-pass a 1-D signal, or the average of two channel arrays."""
        if extra is not None:
            x = (np.asarray(signal, dtype=float) + np.asarray(extra, dtype=float)) / 2.0
        else:
            x = np.asarray(signal, dtype=float)
            if x.ndim == 2:
                x = x.mean(axis=1)
        if x.size == 0:
            return x
        if self._zi is None:
            self._zi = sosfilt_zi(self._sos) * x[0]
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return y

    def envelope(self, filtered):
        out = np.empty_like(filtered, dtype=float)
        for i, value in enumerate(filtered):
            self._win.append(float(value))
            out[i] = max(self._win) - min(self._win)
        return out

    def detect(self, envelope):
        events = []
        rearm_level = self.threshold * self.rearm_fraction
        for i, value in enumerate(envelope):
            self._since_trigger += 1
            if self._warmup > 0:
                self._warmup -= 1
                continue
            if not self._armed:
                if value < rearm_level and self._since_trigger >= self.refractory:
                    self._armed = True
                continue
            if value >= self.threshold:
                events.append(i)
                self._armed = False
                self._since_trigger = 0
        return events

    def process(self, signal, extra=None):
        filtered = self.filter(signal, extra)
        env = self.envelope(filtered)
        return env, self.detect(env)


class NodDetector:
    """Trigger on each nod peak from the NIC2 accelerometer.

    Rearms as soon as the motion falls off the current peak, so a train of
    nods is counted instead of waiting to return to rest.
    """

    def __init__(self, fs=config.ACC_SAMPLE_RATE, threshold=config.DEFAULT_NOD_THRESHOLD):
        self.fs = fs
        self.threshold = threshold
        self.sign = 1.0
        self.refractory = int(config.NOD_REFRACTORY_S * fs)
        self.peak_drop = config.NOD_PEAK_DROP
        self._sos = make_bandpass(fs, config.ACC_BAND_HZ)
        self._zi = None
        self._armed = True
        self._peak = 0.0
        self._prev = 0.0
        self._since_trigger = self.refractory
        self._warmup = int(0.5 * fs)

    def filter(self, xyz):
        x = np.asarray(xyz, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if self._zi is None:
            self._zi = sosfilt_zi(self._sos)[:, :, None] * x[0][None, None, :]
        y, self._zi = sosfilt(self._sos, x, axis=0, zi=self._zi)
        return np.sqrt((y ** 2).sum(axis=1))

    def detect(self, mag):
        events = []
        for i, value in enumerate(mag):
            value = float(value)
            self._since_trigger += 1
            if self._warmup > 0:
                self._warmup -= 1
                self._prev = value
                continue
            if self._armed:
                rising = value >= self.threshold and value >= self._prev
                if rising:
                    events.append(i)
                    self._armed = False
                    self._peak = value
                    self._since_trigger = 0
            else:
                if value > self._peak:
                    self._peak = value
                dropped = value <= self._peak * self.peak_drop
                if dropped and self._since_trigger >= self.refractory:
                    self._armed = True
            self._prev = value
        return events

    def process(self, xyz):
        mag = self.filter(xyz)
        return mag, self.detect(mag)
