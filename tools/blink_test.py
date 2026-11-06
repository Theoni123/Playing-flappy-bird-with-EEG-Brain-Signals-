"""Live blink test. Keep NIC2 running, then:

    python -m tools.blink_test

Blink hard each time it prints BLINK. It reports whether Fp1/Fp2 jumped.
"""

import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

import config
from blink.stream import fp_indices, latest_quality, open_inlet, channel_labels


def main():
    inlet, info = open_inlet(timeout=10.0)
    fs = info.nominal_srate() or config.SAMPLE_RATE
    labels = channel_labels(info)
    i1, i2 = fp_indices(info)
    print(f"Connected to {info.name()!r}. Fp1=ch{i1} ({labels[i1]}), Fp2=ch{i2} ({labels[i2]})")
    print("Reading a full quality scan (NIC2 only measures 5 electrodes per second)...")
    quality = latest_quality(timeout=6.0)
    if quality is not None:
        for i in (i1, i2):
            q = float(quality[i])
            state = "GOOD" if q >= 0.5 else ("weak" if q >= 0 else "not measured")
            print(f"  {labels[i]} quality {q:.2f}  {state}")
    print("\nSit still. Blink HARD every time you see BLINK.\n")

    sos = butter(2, (1, 15), btype="bandpass", fs=fs, output="sos")
    zi = None
    rest_p2p = []
    # 4 s baseline
    print("REST 4s — try not to blink")
    t_end = time.monotonic() + 4
    while time.monotonic() < t_end:
        chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=512)
        if not chunk:
            continue
        x = np.asarray(chunk, dtype=float)[:, [i1, i2]].mean(axis=1)
        if zi is None:
            zi = sosfilt_zi(sos) * x[0]
        y, zi = sosfilt(sos, x, zi=zi)
        rest_p2p.append(np.ptp(y) if y.size else 0.0)
    noise = float(np.median(rest_p2p) * 3) if rest_p2p else 30.0
    print(f"  resting pulse ~ {np.median(rest_p2p):.0f} uV. A blink should be several times bigger.\n")

    hits = 0
    for n in range(1, 7):
        print(f"  BLINK!  ({n}/6)", flush=True)
        t_end = time.monotonic() + 1.6
        peak = 0.0
        while time.monotonic() < t_end:
            chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=512)
            if not chunk:
                continue
            x = np.asarray(chunk, dtype=float)[:, [i1, i2]].mean(axis=1)
            y, zi = sosfilt(sos, x, zi=zi)
            peak = max(peak, float(np.ptp(y)) if y.size else 0.0)
        ok = peak >= max(60.0, noise)
        hits += int(ok)
        print(f"     peak-to-peak {peak:.0f} uV  {'YES — that looked like a blink' if ok else 'no blink seen'}")
        time.sleep(0.4)

    print(f"\n{hits}/6 blinks detected.")
    if hits >= 4:
        print("Fp1/Fp2 are picking up blinks. Run:  python main.py --calibrate --speed 0.7")
    else:
        print(
            "Still no blink pulse. Keep CMS/DRL on the EAR, Fp1/Fp2 on forehead SKIN,\n"
            "and ignore the red/green lights flipping — that is NIC2 scanning, not unplugging."
        )


if __name__ == "__main__":
    main()
