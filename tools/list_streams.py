"""List LSL streams on the network and print live FP1/FP2 values.

Run with:  python -m tools.list_streams [--name STREAM_NAME]
"""

import argparse
import time

import numpy as np
import pylsl

from blink.stream import channel_labels, fp_indices, latest_quality, open_inlet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="LSL stream name (default: first EEG stream)")
    parser.add_argument("--wait", type=float, default=3.0, help="seconds to search")
    args = parser.parse_args()

    print(f"Searching for LSL streams for {args.wait:.0f}s...")
    streams = pylsl.resolve_streams(wait_time=args.wait)
    if not streams:
        print("No streams found. Check that NIC2 has LSL enabled and the protocol is running.")
        return
    for s in streams:
        print(
            f"  name={s.name()!r} type={s.type()!r} channels={s.channel_count()} "
            f"rate={s.nominal_srate():g}Hz host={s.hostname()!r}"
        )

    inlet, info = open_inlet(args.name)
    labels = channel_labels(info)
    i1, i2 = fp_indices(info)
    print(f"\nConnected to {info.name()!r}. Channel labels: {labels or '(none published)'}")
    quality = latest_quality()
    if quality is not None:
        print("Contact quality (-1 = off, 1 = good):")
        for i, q in enumerate(quality):
            lab = labels[i] if i < len(labels) else ""
            print(f"  ch{i:<2d} {lab:<4s} {float(q):5.2f}")
    print(f"Using FP1 = channel {i1}, FP2 = channel {i2}. Blink to see the values jump.")
    print("Printing mean and peak-to-peak (uV) every 0.5s. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.5)
            chunk, _ = inlet.pull_chunk(max_samples=1000)
            if not chunk:
                print("  (no data)")
                continue
            data = np.asarray(chunk)
            fp1, fp2 = data[:, i1], data[:, i2]
            print(
                f"  n={len(data):4d}  FP1 mean={fp1.mean():9.1f} p2p={np.ptp(fp1):7.1f}   "
                f"FP2 mean={fp2.mean():9.1f} p2p={np.ptp(fp2):7.1f}"
            )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
