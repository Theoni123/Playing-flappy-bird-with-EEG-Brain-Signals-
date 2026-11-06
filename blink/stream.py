"""Finding NIC2 LSL streams and locating usable EEG channels."""

import time

import numpy as np
import pylsl

import config


def find_stream(name=None, stream_type=config.STREAM_TYPE, timeout=10.0):
    """Return the StreamInfo of a stream, or raise if none is found."""
    if name:
        results = pylsl.resolve_byprop("name", name, timeout=timeout)
    else:
        results = pylsl.resolve_byprop("type", stream_type, timeout=timeout)
    if not results:
        what = f"named '{name}'" if name else f"of type '{stream_type}'"
        raise RuntimeError(
            f"No LSL stream {what} found within {timeout:.0f}s. "
            "Is NIC2 running with LSL output enabled and the protocol started?"
        )
    return results[0]


def open_inlet(name=None, stream_type=config.STREAM_TYPE, timeout=10.0):
    info = find_stream(name, stream_type, timeout)
    inlet = pylsl.StreamInlet(info, max_buflen=10, processing_flags=pylsl.proc_clocksync)
    return inlet, inlet.info(timeout=timeout)


def channel_labels(info):
    labels = []
    channel = info.desc().child("channels").child("channel")
    while not channel.empty():
        labels.append(channel.child_value("label"))
        channel = channel.next_sibling()
    if labels:
        return labels
    n = info.channel_count()
    if n == len(config.ENOBIO20_LABELS):
        return list(config.ENOBIO20_LABELS)
    return [f"ch{i}" for i in range(n)]


def fp_indices(info):
    """Indices of FP1 and FP2 in each sample, by label if available."""
    upper = [label.upper() for label in channel_labels(info)]
    try:
        return upper.index("FP1"), upper.index("FP2")
    except ValueError:
        return config.FP1_INDEX_FALLBACK, config.FP2_INDEX_FALLBACK


def latest_quality(timeout=6.0):
    """Contact quality, merged over a full NIC2 impedance scan.

    NIC2 only measures about 5 channels per second and sends -1 for the
    others that tick. A single sample therefore looks like electrodes
    flipping on and off. We keep the best non-negative value seen for
    each channel over a few seconds.
    """
    results = pylsl.resolve_byprop("type", "Quality", timeout=min(timeout, 2.0))
    if not results:
        return None
    inlet = pylsl.StreamInlet(results[0], max_buflen=10)
    best = None
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        chunk, _ = inlet.pull_chunk(timeout=0.4, max_samples=16)
        if not chunk:
            continue
        for sample in chunk:
            q = np.asarray(sample, dtype=float)
            if best is None:
                best = np.full(q.shape, np.nan)
            measured = q >= 0
            best[measured] = np.maximum(np.nan_to_num(best[measured], nan=-1), q[measured])
    if best is None:
        return None
    best[np.isnan(best)] = -1.0
    return best


def live_channel_indices(quality, n_channels):
    if quality is None:
        return list(range(n_channels))
    live = [i for i, q in enumerate(quality[:n_channels]) if q >= config.QUALITY_MIN]
    return live
