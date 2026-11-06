"""Blink Bird: Flappy Bird controlled by eye blinks or head nods.

  python main.py --calibrate --speed 0.7
  python main.py --calibrate --control nod --speed 0.7
  python main.py --mock
  python main.py --keyboard
"""

import argparse
import sys
import time

import config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream-name", default=config.STREAM_NAME,
                        help="NIC2 LSL stream name (default: first matching stream)")
    parser.add_argument("--mock", action="store_true", help="use a built-in fake EEG stream")
    parser.add_argument("--calibrate", action="store_true", help="run calibration first")
    parser.add_argument("--control", choices=["blink", "nod"], default=None,
                        help="blink = EEG (default), nod = headset accelerometer")
    parser.add_argument("--threshold", type=float, help="override the detection threshold")
    parser.add_argument("--polarity", choices=["positive", "negative", "abs"],
                        help="ignored for the current peak-to-peak detector")
    parser.add_argument("--channels", type=int, nargs="+",
                        help="override which EEG channel numbers to use")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="game speed, e.g. 0.7 for slower (default 1.0)")
    parser.add_argument("--keyboard", action="store_true", help="no sensors, spacebar only")
    return parser.parse_args()


def main():
    args = parse_args()

    from game.flappy import FlappyGame

    if args.keyboard:
        FlappyGame(speed=args.speed).run()
        return

    from blink.calibrate import load_calibration, run_calibration, run_nod_calibration
    from blink.reader import BlinkReader
    from blink.stream import open_inlet

    mock = None
    stream_name = args.stream_name
    if args.mock:
        from tools.mock_stream import MockStream
        mock = MockStream()
        mock.start()
        stream_name = mock.name
        time.sleep(0.5)

    calibration = load_calibration()
    control = args.control or (calibration or {}).get("control") or "blink"
    if (
        calibration
        and args.control
        and calibration.get("control") not in (None, args.control)
        and not args.calibrate
    ):
        sys.exit(
            f"Saved calibration is for {calibration.get('control')}, but you asked for "
            f"{args.control}. Recalibrate with:\n  python main.py --calibrate --control {args.control} --speed 0.7"
        )
    if args.mock and control == "nod":
        sys.exit("Mock stream is EEG-only. Use --mock without --control nod, or use the headset.")

    if args.calibrate:
        stream_type = config.ACC_STREAM_TYPE if control == "nod" else config.STREAM_TYPE
        try:
            inlet, info = open_inlet(stream_name, stream_type=stream_type, timeout=10.0)
        except RuntimeError as err:
            sys.exit(str(err))
        while True:
            if control == "nod":
                calibration = run_nod_calibration(inlet, info)
            else:
                calibration = run_calibration(inlet, info, on_mock_blink=mock.blink if mock else None)
            if calibration.get("score", 0) >= 2.0:
                break
            again = input("Calibration is too weak to play. Try again? [Y/n] ").strip().lower()
            if again.startswith("n"):
                break
        inlet.close_stream()
        input("Press Enter to start the game...")

    threshold = config.DEFAULT_NOD_THRESHOLD if control == "nod" else config.DEFAULT_THRESHOLD_UV
    channels = None
    if calibration:
        threshold = calibration["threshold_uv"]
        channels = calibration.get("channels") or None
        control = args.control or calibration.get("control") or control
        print(
            f"Using calibration from {calibration['created']} ({calibration.get('control', 'blink')}): "
            f"channels {channels or 'default'}, threshold {threshold:.0f}, "
            f"score {calibration.get('score', '?')}"
        )
    else:
        print(f"No calibration found, using default threshold {threshold:.0f}. "
              "Run with --calibrate for better detection.")
    if args.threshold is not None:
        threshold = args.threshold
    if args.channels:
        channels = args.channels

    reader = BlinkReader(
        stream_name, threshold=threshold, channels=channels, control=control
    )
    reader.start()
    try:
        speed = args.speed
        if control == "nod" and args.speed == 1.0:
            speed = 0.7
        FlappyGame(reader=reader, mock=mock, speed=speed).run()
    finally:
        reader.stop()
        if mock:
            mock.stop()


if __name__ == "__main__":
    main()
