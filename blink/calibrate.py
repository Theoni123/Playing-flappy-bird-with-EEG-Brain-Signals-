"""Calibrate blink (EEG) or nod (accelerometer) thresholds.

Run with:  python main.py --calibrate
           python main.py --calibrate --control nod
"""

import json
import subprocess
import threading
import time

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

import config
from blink.detector import make_bandpass, rolling_p2p
from blink.stream import channel_labels, latest_quality, live_channel_indices

SETTLE_S = 1.0
REST_S = 6.0
ACTION_S = 8.0
COUNTDOWN_S = 3
SECOND_CHANNEL_RATIO = 0.75
MIN_SCORE = 2.0
P2P_WINDOW_S = config.P2P_WINDOW_S


def _beep():
    threading.Thread(
        target=lambda: subprocess.run(
            ["afplay", "/System/Library/Sounds/Tink.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
        daemon=True,
    ).start()


def _say(text):
    print(text, flush=True)


def _countdown(label):
    _say(f"\n  {label}")
    for n in range(COUNTDOWN_S, 0, -1):
        _say(f"     {n}...")
        _beep()
        time.sleep(1.0)
    _beep()


def _record(inlet, seconds, tick=None):
    out = []
    end = time.monotonic() + seconds
    next_tick = time.monotonic() + 1.0
    while time.monotonic() < end:
        chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=1024)
        if chunk:
            out.append(np.asarray(chunk, dtype=float))
        now = time.monotonic()
        if tick and now >= next_tick:
            tick()
            next_tick += 1.0
    if out:
        return np.concatenate(out)
    return np.zeros((0, inlet.channel_count()))


def _bandpass(raw, fs, band):
    sos = make_bandpass(fs, band)
    zi = sosfilt_zi(sos)[:, :, None] * raw[0][None, None, :]
    filtered, _ = sosfilt(sos, raw, axis=0, zi=zi)
    return filtered


def _envelope(filtered, fs):
    window = max(8, int(P2P_WINDOW_S * fs))
    return np.stack([rolling_p2p(filtered[:, ch], window) for ch in range(filtered.shape[1])], axis=1)


def _p90(x):
    return np.percentile(x, 90, axis=0)


def _print_quality(quality, labels, n_channels):
    if quality is None:
        _say("  (NIC2 quality stream not found; using every EEG channel)")
        return
    _say("  Contact quality from NIC2  (-1 = channel is OFF, 1 = good contact):")
    live = []
    for i in range(n_channels):
        q = float(quality[i]) if i < len(quality) else -1.0
        mark = "LIVE" if q >= config.QUALITY_MIN else "off"
        label = labels[i] if i < len(labels) else f"ch{i}"
        _say(f"    ch{i:<2d} {label:<4s}  {q:5.2f}  {mark}")
        if q >= config.QUALITY_MIN:
            live.append(label)
    fp_live = [lab for lab in live if lab.upper() in ("FP1", "FP2")]
    if not fp_live:
        _say(
            "\n  FP1 and FP2 are OFF in this NIC2 protocol. That is why blinks look\n"
            "  weak: the game was reading disconnected channels. Either enable FP1/FP2\n"
            "  in the NIC2 montage, or we will use the LIVE electrodes above.\n"
            "  If blinks stay weak, play with a head nod instead:\n"
            "      python main.py --calibrate --control nod --speed 0.7"
        )


def run_calibration(inlet, info, on_mock_blink=None):
    fs = info.nominal_srate() or config.SAMPLE_RATE
    labels = channel_labels(info)
    n_channels = info.channel_count()
    quality = latest_quality()
    if quality is not None and len(quality) != n_channels:
        quality = None
    _say("\nBlink calibration. Keep your eyes OPEN the whole time.")
    _print_quality(quality, labels, n_channels)
    live = live_channel_indices(quality, n_channels)
    if not live:
        raise RuntimeError(
            "No EEG channel has contact quality above 0.3. In NIC2, enable FP1 and FP2 "
            "in the montage and wait until their quality is green, then calibrate again. "
            "Or run: python main.py --calibrate --control nod"
        )

    _record(inlet, SETTLE_S)
    _countdown(f"1/2  REST — look at the screen for {REST_S:.0f}s. Try not to blink.")
    _say("     recording rest...")
    rest_raw = _record(inlet, REST_S)
    if rest_raw.shape[0] == 0:
        raise RuntimeError("No data received from the stream during calibration.")

    def action_tick():
        _say("     BLINK!")
        _beep()
        if on_mock_blink:
            on_mock_blink()

    _countdown(f"2/2  BLINK — blink HARD every time you see BLINK, for {ACTION_S:.0f}s.")
    _say("     recording blinks...")
    action_raw = _record(inlet, ACTION_S, tick=action_tick)

    rest_env = _envelope(_bandpass(rest_raw, fs, config.BANDPASS_HZ), fs)
    act_env = _envelope(_bandpass(action_raw, fs, config.BANDPASS_HZ), fs)
    rest_p = _p90(rest_env)
    act_p = _p90(act_env)
    score = act_p / np.maximum(rest_p, 1.0)

    ranked = sorted(((float(score[ch]), ch) for ch in live), reverse=True)
    _say("\n  Live channels, blink pulse vs rest (90th percentile of 200 ms peak-to-peak):")
    for value, ch in ranked[:8]:
        label = labels[ch] if ch < len(labels) else ""
        _say(
            f"    ch{ch} {label:<4s}  {value:5.1f}x   blink {act_p[ch]:7.1f} uV  rest {rest_p[ch]:6.1f} uV"
        )

    upper = [lab.upper() for lab in labels]
    fp_channels = [i for i, lab in enumerate(upper) if lab in ("FP1", "FP2") and i in live]
    if fp_channels:
        channels = fp_channels
        best_score = float(np.mean([score[i] for i in channels]))
        _say("  (locking onto Fp1/Fp2 — forehead blinks live here)")
    else:
        best_score, best_ch = ranked[0]
        channels = [best_ch]
        for value, ch in ranked[1:]:
            if value >= SECOND_CHANNEL_RATIO * best_score:
                channels.append(ch)
                break

    rest_c = rest_env[:, channels].mean(axis=1)
    act_c = act_env[:, channels].mean(axis=1)
    noise = float(_p90(rest_c))
    blink = float(_p90(act_c))
    threshold = 0.35 * noise + 0.65 * blink

    names = " + ".join(
        f"ch{c}" + (f" ({labels[c]})" if c < len(labels) else "") for c in channels
    )
    _say(f"\n  using               : {names}")
    _say(f"  rest pulse          : {noise:7.1f} uV")
    _say(f"  blink pulse         : {blink:7.1f} uV")
    _say(f"  -> threshold        : {threshold:7.1f} uV")
    if best_score < MIN_SCORE:
        _say(
            "\n  Still too weak. Enable FP1/FP2 in NIC2 (they are currently off), wet the\n"
            "  electrodes, or play with head nods:\n"
            "      python main.py --calibrate --control nod --speed 0.7"
        )

    settings = {
        "control": "blink",
        "threshold_uv": round(threshold, 2),
        "polarity": "abs",
        "channels": [int(c) for c in channels],
        "noise_uv": round(noise, 2),
        "blink_uv": round(blink, 2),
        "score": round(float(best_score), 2),
        "stream_name": info.name(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.CALIBRATION_FILE.write_text(json.dumps(settings, indent=2))
    _say(f"  saved to {config.CALIBRATION_FILE.name}\n")
    return settings


def run_nod_calibration(inlet, info):
    fs = info.nominal_srate() or config.ACC_SAMPLE_RATE
    _say(
        "\nHead-nod calibration. Sit upright. A small nod (yes gesture) will flap the bird.\n"
        "This uses the headset accelerometer, not the EEG electrodes."
    )
    _record(inlet, SETTLE_S)
    _countdown(f"1/2  REST — hold your head still for {REST_S:.0f}s.")
    _say("     recording rest...")
    rest_raw = _record(inlet, REST_S)
    if rest_raw.shape[0] == 0:
        raise RuntimeError("No accelerometer data received.")

    def tick():
        _say("     NOD!")
        _beep()

    _countdown(f"2/2  NOD — nod your head every time you see NOD, for {ACTION_S:.0f}s.")
    _say("     recording nods...")
    action_raw = _record(inlet, ACTION_S, tick=tick)

    rest = _bandpass(rest_raw, fs, config.ACC_BAND_HZ)
    act = _bandpass(action_raw, fs, config.ACC_BAND_HZ)
    rest_mag = np.sqrt((rest ** 2).sum(axis=1))
    act_mag = np.sqrt((act ** 2).sum(axis=1))
    noise = float(_p90(rest_mag))
    nod = float(_p90(act_mag))
    threshold = 0.35 * noise + 0.65 * nod
    score = nod / max(noise, 1.0)

    _say(f"\n  rest motion         : {noise:7.1f}")
    _say(f"  nod motion          : {nod:7.1f}")
    _say(f"  -> threshold        : {threshold:7.1f}  ({score:.1f}x)")
    if score < MIN_SCORE:
        _say("  Nods were not much bigger than rest. Nod more clearly and try again.")

    settings = {
        "control": "nod",
        "threshold_uv": round(threshold, 2),
        "polarity": "abs",
        "channels": [],
        "noise_uv": round(noise, 2),
        "blink_uv": round(nod, 2),
        "score": round(score, 2),
        "stream_name": info.name(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.CALIBRATION_FILE.write_text(json.dumps(settings, indent=2))
    _say(f"  saved to {config.CALIBRATION_FILE.name}\n")
    return settings


def load_calibration():
    try:
        return json.loads(config.CALIBRATION_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
