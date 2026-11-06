"""Settings shared by the game, calibration and tools."""

from pathlib import Path

# LSL stream published by NIC2. Leave STREAM_NAME as None to take the first
# stream of type "EEG" found on the network.
STREAM_NAME = None
STREAM_TYPE = "EEG"
MOCK_STREAM_NAME = "MockEnobio"

# Channel labels to look for in the stream metadata. If NIC2 does not publish
# labels, the fallback indices are used (0-based position in each sample).
FP1_LABEL = "FP1"
FP2_LABEL = "FP2"
FP1_INDEX_FALLBACK = 0
FP2_INDEX_FALLBACK = 1

# Default Enobio 20 order used when NIC2 does not send labels.
ENOBIO20_LABELS = [
    "P7", "P4", "Cz", "Pz", "P3", "P8", "O1", "O2", "T8", "F8",
    "C4", "F4", "Fp2", "Fz", "C3", "F3", "Fp1", "T7", "F7", "Oz",
]

SAMPLE_RATE = 500.0

# Blink detection. 2-12 Hz keeps the blink pulse and drops slow electrode drift.
BANDPASS_HZ = (2.0, 12.0)
FILTER_ORDER = 2
REFRACTORY_S = 0.35
DEFAULT_THRESHOLD_UV = 80.0
P2P_WINDOW_S = 0.20
QUALITY_MIN = 0.3

# Head-nod detection from NIC2's accelerometer stream.
ACC_STREAM_TYPE = "Accelerometer"
ACC_BAND_HZ = (1.0, 8.0)
ACC_SAMPLE_RATE = 100.0
DEFAULT_NOD_THRESHOLD = 400.0
NOD_REFRACTORY_S = 0.28
# Rearm after the nod peak has fallen by this fraction, even if still above threshold.
NOD_PEAK_DROP = 0.55

# Direction of the blink deflection on FP1/FP2. Depends on the reference
# placement; calibration measures it and overrides this default.
POLARITY = "positive"  # "positive" or "negative"
# After a trigger, the signal must fall below this fraction of the threshold
# before another blink can be detected.
REARM_FRACTION = 0.5

CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"
