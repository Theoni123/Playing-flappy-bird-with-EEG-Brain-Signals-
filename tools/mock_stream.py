"""Fake NIC2 EEG stream for developing without the headset.

Run with:  python -m tools.mock_stream [--auto 2.5]
Press Enter in this terminal to inject a blink. With --auto N, a blink is
also injected roughly every N seconds.
"""

import argparse
import random
import sys
import threading
import time

import numpy as np
import pylsl

import config

LABELS = ["P7", "P4", "Cz", "FP1", "FP2", "P3", "O1", "O2"]
BLINK_UV = 250.0
BLINK_S = 0.3


class MockStream:
    def __init__(self, name=config.MOCK_STREAM_NAME, fs=config.SAMPLE_RATE, auto_interval=0.0):
        self.name = name
        self.fs = fs
        self.auto_interval = auto_interval
        self._shape = self._blink_shape()
        self._fp_idx = [LABELS.index("FP1"), LABELS.index("FP2")]
        self._pending = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._outlet = self._make_outlet()

    def _make_outlet(self):
        info = pylsl.StreamInfo(
            self.name, config.STREAM_TYPE, len(LABELS), self.fs, "float32", f"mock-{self.name}"
        )
        channels = info.desc().append_child("channels")
        for label in LABELS:
            ch = channels.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
        return pylsl.StreamOutlet(info, chunk_size=10)

    def _blink_shape(self):
        t = np.arange(int(BLINK_S * self.fs)) / self.fs
        return BLINK_UV * np.sin(np.pi * t / BLINK_S) ** 2

    def blink(self):
        with self._lock:
            self._pending.append(0)

    def stop(self):
        self._stop.set()

    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread

    def run(self):
        fs = self.fs
        rng = np.random.default_rng()
        drift = np.zeros(len(LABELS))
        n_sent = 0
        start = time.perf_counter()
        next_auto = self._next_auto(start)

        while not self._stop.is_set():
            now = time.perf_counter()
            if next_auto and now >= next_auto:
                self.blink()
                next_auto = self._next_auto(now)

            n = int((now - start) * fs) - n_sent
            if n <= 0:
                time.sleep(0.005)
                continue

            t = (n_sent + np.arange(n)) / fs
            drift = 0.98 * drift + rng.normal(0, 2.0, len(LABELS))
            data = rng.normal(0, 8.0, (n, len(LABELS)))
            data += 10.0 * np.sin(2 * np.pi * 10 * t)[:, None]  # alpha
            data += 15.0 * np.sin(2 * np.pi * 50 * t)[:, None]  # mains hum
            data += drift

            with self._lock:
                for k, pos in enumerate(self._pending):
                    seg = self._shape[pos:pos + n]
                    data[: len(seg), self._fp_idx] += seg[:, None]
                    self._pending[k] = pos + n
                self._pending = [p for p in self._pending if p < len(self._shape)]

            self._outlet.push_chunk(data.astype(np.float32).tolist())
            n_sent += n

    def _next_auto(self, now):
        if not self.auto_interval:
            return None
        return now + self.auto_interval * random.uniform(0.6, 1.4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=config.MOCK_STREAM_NAME)
    parser.add_argument("--auto", type=float, default=0.0, help="auto-blink every ~N seconds")
    args = parser.parse_args()

    mock = MockStream(args.name, auto_interval=args.auto)
    mock.start()
    print(f"Mock stream {args.name!r} running at {mock.fs:g} Hz. Press Enter to blink, Ctrl+C to stop.")
    try:
        for _ in sys.stdin:
            mock.blink()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        mock.stop()


if __name__ == "__main__":
    main()
