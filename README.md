# dot11decoder

Python implementation of a Wi‑Fi OFDM decoder that extracts payload bytes from raw complex baseband samples. It understands the short/long training sequences, estimates the channel with both legacy and HT long training fields, and feeds soft bits to a fast Viterbi core.

## Supported PHYs
- 802.11a/g legacy OFDM (all eight MCS values supported end-to-end).
- 802.11n HT packets that reuse the 11a/g preamble. The HT path currently handles single-stream 20 MHz packets with short or long GI. Features such as multiple spatial streams, LDPC, or 40 MHz extensions are not implemented, so not every 802.11n capture will decode.

## API overview
The main entry point is `dot11decoder.Decoder`, which consumes a `numpy.ndarray` of complex64 IQ samples sampled at 20 MHz.

### `Decoder.decode`
- Requires the internal buffer to already point to the legacy short training sequence.
- Use this when the provided IQ slice is already aligned to the start of a frame or is only off by a few samples (for example when you trimmed the capture around a trigger).
- Performs timing synchronization, channel estimation, and returns a `DecodeResult` with the PSDU plus modulation, CSI, CFO estimate, GI flag, and symbol count metadata.

### `Decoder.decode_next`
- Runs the short-training-sequence detector to locate each packet in a long capture, repositions the internal buffer, then calls `decode()` for you.
- Use this when the capture contains many frames or unknown gaps: the detector finds the start of each frame, so `decode_next()` is required in that situation.
- If your IQ samples are already aligned (or nearly aligned) to a single frame, `decode()` is both simpler and faster, so skip `decode_next()` in that case.

### `DecodeResult`
The object returned by either API exposes:
- `psdu`: payload bytes.
- `modulation`: one of `BPSK`, `QPSK`, `16-QAM`, `64-QAM`, derived from the selected MCS.
- `mcs_index`: integer value decoded from the SIGNAL/HT-SIG MCS bits (legacy indexes 0–7, HT indexes 0–7).
- `ht`: `True` for HT packets, `False` for legacy 11a/g.
- `short_gi`: whether the HT-SIG indicated short guard interval (legacy frames always `False`).
- `n_data_symbols`, `rate_mbps`, `length_bytes`: framing stats straight from the signal field.
- `csi`: complex64 array with the per-subcarrier channel estimate.
- `freq_offset_hz`: combined coarse/fine CFO estimate applied during equalization.

## Example
```python
import numpy as np
from dot11decoder import Decoder

# Load 20 MHz complex baseband samples (interleaved float32 IQ) from a file
samples = np.fromfile("capture.cf32", dtype=np.complex64)

decoder = Decoder(samples)

# Option A: already-aligned frame, just decode once
result = decoder.decode()
print(
    "Single frame:",
    result.psdu,
    result.modulation,
    f"{result.rate_mbps:.1f} Mbps",
    f"CFO {result.freq_offset_hz:.0f} Hz",
)

# Option B: long capture with many packets, iterate over detections
for pkt_idx, result in enumerate(decoder.decode_next()):
    print(f"Frame {pkt_idx}: {result.psdu[:16]!r} shortGI={result.short_gi} ht={result.ht}")
```

## Requirements
- Python 3.12+
- `numpy`, `scipy`
- Viterbi decoder:
  - The implementation lives in `viterbi.py`, which runs as plain Python out of the box.
  - To enable the Cython speedups, install Cython and run `cythonize -i viterbi.py`. The pure-Python module and the compiled extension share the same source, so no code changes are required when switching between the two.
