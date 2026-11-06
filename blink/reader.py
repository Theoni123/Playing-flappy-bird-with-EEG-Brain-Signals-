"""Background thread: LSL inlet -> detector -> queue of flap events."""

import collections
import queue
import threading

import numpy as np

import config
from blink.detector import BlinkDetector, NodDetector
from blink.stream import fp_indices, open_inlet

TRACE_SECONDS = 3.0


class BlinkReader(threading.Thread):
    def __init__(self, stream_name=None, threshold=config.DEFAULT_THRESHOLD_UV,
                 polarity=config.POLARITY, channels=None, control="blink"):
        super().__init__(daemon=True)
        self.stream_name = stream_name
        self.threshold = threshold
        self.polarity = polarity
        self.channels = channels
        self.control = control
        self.flaps = queue.Queue()
        self.status = "searching for stream..."
        self.connected = False
        self.fs = config.SAMPLE_RATE if control == "blink" else config.ACC_SAMPLE_RATE
        self._trace = collections.deque(maxlen=int(TRACE_SECONDS * self.fs))
        self._trace_lock = threading.Lock()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def trace(self):
        with self._trace_lock:
            return np.fromiter(self._trace, dtype=float)

    def run(self):
        while not self._stop.is_set():
            try:
                if self.control == "nod":
                    inlet, info = open_inlet(
                        self.stream_name, stream_type=config.ACC_STREAM_TYPE, timeout=5.0
                    )
                    self._acc_loop(inlet, info)
                else:
                    inlet, info = open_inlet(self.stream_name, timeout=5.0)
                    self._eeg_loop(inlet, info)
            except RuntimeError:
                kind = "accelerometer" if self.control == "nod" else "EEG"
                self.status = f"no {kind} stream found, retrying... (space still works)"

    def _eeg_loop(self, inlet, info):
        self.fs = info.nominal_srate() or config.SAMPLE_RATE
        channels = list(self.channels) if self.channels else list(fp_indices(info))
        detector = BlinkDetector(fs=self.fs, threshold=self.threshold)
        with self._trace_lock:
            self._trace = collections.deque(maxlen=int(TRACE_SECONDS * self.fs))
        self.connected = True
        used = " + ".join(f"ch{c}" for c in dict.fromkeys(channels))
        self.status = f"blink  {info.name()}  {used}  ({self.fs:g} Hz)"
        self._pull_loop(inlet, detector, channels)

    def _acc_loop(self, inlet, info):
        self.fs = info.nominal_srate() or config.ACC_SAMPLE_RATE
        detector = NodDetector(fs=self.fs, threshold=self.threshold)
        with self._trace_lock:
            self._trace = collections.deque(maxlen=int(TRACE_SECONDS * self.fs))
        self.connected = True
        self.status = f"nod  {info.name()}  acc XYZ  ({self.fs:g} Hz)"
        self._pull_loop(inlet, detector, None)

    def _pull_loop(self, inlet, detector, channels):
        silent_pulls = 0
        while not self._stop.is_set():
            chunk, _ = inlet.pull_chunk(timeout=0.05, max_samples=256)
            if not chunk:
                silent_pulls += 1
                if silent_pulls > 60:
                    self.connected = False
                    self.status = "stream stopped sending data, reconnecting..."
                    return
                continue
            silent_pulls = 0
            data = np.asarray(chunk, dtype=float)
            if channels is None:
                env, events = detector.process(data)
            else:
                env, events = detector.process(data[:, channels].mean(axis=1))
            with self._trace_lock:
                self._trace.extend(env)
            for _ in events:
                self.flaps.put(self.control)
