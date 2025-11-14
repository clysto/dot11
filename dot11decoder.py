import math
from collections import deque

import numpy as np
from viterbi import Viterbi as FastViterbi

FS = 20e6
LTS_F = np.concat(
    [
        [0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1],
        [0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ]
)
HT_LTS_F = np.concat(
    [
        [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1],
        [0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 0, 0, 0],
    ]
)
LTS_T = np.fft.ifft(np.fft.fftshift(LTS_F))
LTS_LEN = 64
STS_LEN = 16
HT_STS_LEN = 80
HT_LTS_LEN = 64
CP_LEN = 16


DATA_IND = np.hstack(
    (
        np.arange(6, 11),
        np.arange(12, 25),
        np.arange(26, 32),
        np.arange(33, 39),
        np.arange(40, 53),
        np.arange(54, 59),
    )
)
HT_DATA_IND = np.hstack(
    (
        np.arange(4, 11),
        np.arange(12, 25),
        np.arange(26, 32),
        np.arange(33, 39),
        np.arange(40, 53),
        np.arange(54, 61),
    )
)

PILOT_IND = np.array([11, 25, 39, 53])

PILOT_SEQ = np.concat(
    [
        [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1],
        [1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1],
        [-1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1],
        [-1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
    ]
)


def hamming_distance(l1, l2):
    return sum(1 for a, b in zip(l1, l2) if a != b and a != -1 and b != -1)


class Viterbi:
    def __init__(self, constraint, polynomials, puncpat=None):
        self._constraint = constraint
        self._polynomials = polynomials
        self._puncpat = puncpat
        self._outputs = [[] for _ in range(1 << self._constraint)]
        self._n_parity_bits = len(polynomials)

        for i in range(1 << self._constraint):
            for p in self._polynomials:
                self._outputs[i].append(int.bit_count(i & p) % 2)

    def _puncture(self, bits):
        puncpat_len = len(self._puncpat)
        return [bit for i, bit in enumerate(bits) if self._puncpat[i % puncpat_len]]

    def _depuncture(self, bits):
        depunctured = []
        it = iter(bits)
        while True:
            for flag in self._puncpat:
                if flag == 1:
                    try:
                        depunctured.append(next(it))
                    except StopIteration:
                        return depunctured
                else:
                    depunctured.append(-1)

    def encode(self, bits):
        output = []
        state = 0
        for b in bits:
            state = (state >> 1) | b << (self._constraint - 1)
            output.extend(self._outputs[state])

        if self._puncpat is not None:
            return self._puncture(output)
        else:
            return output

    def decode(self, bits):
        if self._puncpat is not None:
            bits = self._depuncture(bits)

        trellis: list[list] = []
        path_metrics = [0 if i == 0 else math.inf for i in range(1 << (self._constraint - 1))]

        for i in range(math.ceil(len(bits) / self._n_parity_bits)):
            trellis.append([])
            cur_path_metrics = []

            cur_bits = bits[i * self._n_parity_bits : (i + 1) * self._n_parity_bits]
            if len(cur_bits) < self._n_parity_bits:
                # pad -1
                cur_bits += [-1] * (self._n_parity_bits - len(cur_bits))

            for cur in range(1 << (self._constraint - 1)):
                mask = (1 << (self._constraint - 1)) - 1

                prev1 = (cur << 1) | 0
                prev2 = (cur << 1) | 1

                pm1 = hamming_distance(self._outputs[prev1], cur_bits) + path_metrics[prev1 & mask]
                pm2 = hamming_distance(self._outputs[prev2], cur_bits) + path_metrics[prev2 & mask]

                if pm1 < pm2:
                    trellis[i].append(prev1 & mask)
                    cur_path_metrics.append(pm1)
                else:
                    trellis[i].append(prev2 & mask)
                    cur_path_metrics.append(pm2)

            path_metrics = cur_path_metrics

        # traceback
        out = []
        state = path_metrics.index(min(path_metrics))

        for i in reversed(range(len(trellis))):
            out.append(state >> (self._constraint - 2))
            state = trellis[i][state]

        return out[::-1]


class LegacySignal:
    # n_bpsc, n_cbps, n_dbps, puncpat 802.11a/g
    LT_MCS_PARAMETERS = {
        "1101": (1, 48, 24, None),  # 6 Mbps  (BPSK, 1/2)
        "1111": (1, 48, 36, [1, 1, 1, 0, 0, 1]),  # 9 Mbps  (BPSK, 3/4)
        "0101": (2, 96, 48, None),  # 12 Mbps (QPSK, 1/2)
        "0111": (2, 96, 72, [1, 1, 1, 0, 0, 1]),  # 18 Mbps (QPSK, 3/4)
        "1001": (4, 192, 96, None),  # 24 Mbps (16QAM, 1/2)
        "1011": (4, 192, 144, [1, 1, 1, 0, 0, 1]),  # 36 Mbps (16QAM, 3/4)
        "0001": (6, 288, 192, [1, 1, 1, 0]),  # 48 Mbps (64QAM, 2/3)
        "0011": (6, 288, 216, [1, 1, 1, 0, 0, 1]),  # 54 Mbps (64QAM, 3/4)
    }

    def __init__(self, bits):
        assert len(bits) == 24
        str_bits = "".join([str(b) for b in bits])
        self.mcs_bits = str_bits[:4]
        self.rsvd = str_bits[4]
        self.len_bits = str_bits[5:17]
        self.parity_bits = str_bits[17]
        self.tail_bits = str_bits[18:]
        self.length = int(self.len_bits[::-1], 2)
        self.ht = False
        self.mcs = self.LT_MCS_PARAMETERS[self.mcs_bits]

        parity_ok = sum(bits[:18]) % 2 == 0
        if not parity_ok:
            raise ValueError("LegacySignal parity check failed")


class HTSignal:
    # n_bpsc, n_cbps, n_dbps, puncpat 802.11n (single spatial stream ONLY)
    HT_MCS_PARAMETERS = {
        "0000000": (1, 52, 26, None),  # 6.5 Mbps  (BPSK, 1/2)
        "1000000": (2, 104, 52, None),  # 13.0 Mbps (QPSK, 1/2)
        "0100000": (2, 104, 78, [1, 1, 1, 0, 0, 1]),  # 19.5 Mbps (QPSK, 3/4)
        "1100000": (4, 208, 104, None),  # 26.0 Mbps (16QAM, 1/2)
        "0010000": (4, 208, 156, [1, 1, 1, 0, 0, 1]),  # 39.0 Mbps (16QAM, 3/4)
        "1010000": (6, 312, 208, [1, 1, 1, 0]),  # 52.0 Mbps (64QAM, 2/3)
        "0110000": (6, 312, 234, [1, 1, 1, 0, 0, 1]),  # 58.5 Mbps (64QAM, 3/4)
        "1110000": (6, 312, 260, [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]),  # 65.0 Mbps (64QAM, 5/6)
    }

    def __init__(self, bits):
        assert len(bits) == 48
        str_bits = "".join([str(b) for b in bits])
        self.mcs_bits = str_bits[:7]
        self.cbw = str_bits[7]
        self.len_bits = str_bits[8:24]
        self.smoothing = str_bits[24]
        self.not_sounding = str_bits[25]
        self.rsvd = str_bits[26]
        self.aggregation = str_bits[27]
        self.stbc = str_bits[28:30]
        self.fec = str_bits[30]
        self.short_gi = str_bits[31]
        self.num_ext_stream = str_bits[32:34]
        self.crc = str_bits[34:42]
        self.tail_bits = str_bits[42:48]
        self.length = int(self.len_bits[::-1], 2)
        self.ht = True

        expected_crc = "".join(["%d" % c for c in self.calc_crc(bits[:34])])
        if expected_crc != self.crc:
            raise ValueError(f"HTSignal CRC check failed: expected {expected_crc}, got {self.crc}")
        self.check()

        self.mcs = self.HT_MCS_PARAMETERS[self.mcs_bits]

    def calc_crc(self, bits):
        c = [1] * 8

        for b in bits:
            next_c = [0] * 8
            next_c[0] = b ^ c[7]
            next_c[1] = b ^ c[7] ^ c[0]
            next_c[2] = b ^ c[7] ^ c[1]
            next_c[3] = c[2]
            next_c[4] = c[3]
            next_c[5] = c[4]
            next_c[6] = c[5]
            next_c[7] = c[6]
            c = next_c

        return [1 - b for b in c[::-1]]

    def check(self) -> None:
        if self.mcs_bits not in self.HT_MCS_PARAMETERS:
            raise ValueError(f"Unsupported HT MCS: {self.mcs_bits}. Only single-stream MCS entries are supported.")
        if self.cbw != "0":
            raise ValueError("Unsupported channel bandwidth: only 20 MHz (cbw=0) is supported.")
        if self.rsvd != "1":
            raise ValueError(f"Unexpected reserved bit: rsvd={self.rsvd}. Expected '1'.")
        if self.stbc != "00":
            raise ValueError(f"Unsupported STBC setting: stbc={self.stbc}. Expected '00' (disabled).")
        if self.fec != "0":
            raise ValueError(f"Unsupported FEC: fec={self.fec}. Expected '0' (BCC).")
        if self.num_ext_stream != "00":
            raise ValueError(f"Unsupported number of extension spatial streams: {self.num_ext_stream}. Expected '00' (single stream).")
        if not all(b == "0" for b in self.tail_bits):
            raise ValueError("HT-SIG tail bits must be all zeros.")


class Demodulator:
    QAM16_MAPPING = {
        0b00: -3,
        0b01: -1,
        0b11: 1,
        0b10: 3,
    }

    QAM64_MAPPING = {
        0b000: -7,
        0b001: -5,
        0b011: -3,
        0b010: -1,
        0b110: 1,
        0b111: 3,
        0b101: 5,
        0b100: 7,
    }

    def __init__(self, n_bpsc=1):
        # scale is normalization factor (1 / K_mod) 802.11-2020 p2824 table 17-1
        if n_bpsc == 1:
            # BPSK
            self.scale = 1
            self.bits_per_sym = 1
            self.cons_points = np.array([complex(-1, 0), complex(1, 0)])
        elif n_bpsc == 2:
            # QPSK
            self.scale = math.sqrt(2)
            self.bits_per_sym = 2
            self.cons_points = np.array([complex(-1, -1), complex(-1, 1), complex(1, -1), complex(1, 1)])
        elif n_bpsc == 4:
            # 16-QAM
            self.scale = math.sqrt(10)
            self.bits_per_sym = 4
            self.cons_points = np.array([complex(self.QAM16_MAPPING[i >> 2], self.QAM16_MAPPING[i & 0b11]) for i in range(16)])
        elif n_bpsc == 6:
            # 64-QAM
            self.scale = math.sqrt(42)
            self.bits_per_sym = 6
            self.cons_points = np.array([complex(self.QAM64_MAPPING[i >> 3], self.QAM64_MAPPING[i & 0b111]) for i in range(64)])
        else:
            raise ValueError(f"Unsupported n_bpsc={n_bpsc}. Expected one of {1, 2, 4, 6}.")

    def demodulate(self, carriers: np.ndarray) -> list[int]:
        bits = []
        for sym in carriers:
            idx = np.argmin(np.abs(sym * self.scale - self.cons_points))
            bitstr = f"{idx:0{self.bits_per_sym}b}"
            bits.extend(map(int, bitstr))
        return bits


class SampleBuffer:
    def __init__(self, samples: np.ndarray):
        self._samples = samples
        self._pos = 0
        self._cfo = 0

    def read(self, count):
        ret = self._samples[self._pos : self._pos + count].copy()
        ret *= np.exp(2j * np.pi * self._cfo * ((np.arange(len(ret)) + self._pos) / FS))
        if len(ret) == 0:
            raise Exception("Reached end of sample buffer.")
        self._pos += len(ret)
        return ret

    def window(self, offset, length):
        ret = self._samples[self._pos + offset : self._pos + offset + length].copy()
        ret *= np.exp(2j * np.pi * self._cfo * ((np.arange(len(ret)) + self._pos + offset) / FS))
        return ret

    def advance(self, count):
        self._pos += count

    def freq_compensation(self, cfo):
        self._cfo += cfo


class ChannelEstimator:
    def __init__(self, sample_buffer: SampleBuffer):
        self._buffer = sample_buffer
        self._pilot_seq = deque(PILOT_SEQ)
        self._pilot_polarity = deque([1, 1, 1, -1])
        self._ht = False

        # Estimate CFO using STS
        coarse_cfo_est = self._coarse_cfo_estimate()
        self._buffer.freq_compensation(coarse_cfo_est)

        # Skip 10xSTS
        self._buffer.advance(STS_LEN * 10)
        # Skip GI2 = 2 * CP_LEN
        self._buffer.advance(2 * CP_LEN)

        # Estimate CFO using LTS
        fine_cfo_est = self._fine_cfo_estimate()
        self._buffer.freq_compensation(fine_cfo_est)

        # LTS1
        lts1 = self._buffer.read(LTS_LEN)
        # LTS2
        lts2 = self._buffer.read(LTS_LEN)

        lts1_f = np.fft.fftshift(np.fft.fft(lts1))
        lts2_f = np.fft.fftshift(np.fft.fft(lts2))

        # Calculate channel state information (CSI)
        h1 = LTS_F / lts1_f
        h2 = LTS_F / lts2_f
        self._h_est = (h1 + h2) / 2

    def next_symbol(self):
        sym = self._buffer.read(CP_LEN + 64)
        sym = np.fft.fftshift(np.fft.fft(sym[CP_LEN:])) * self._h_est

        # Calculate the CPE (beta) using pilot symbols
        seq = self._pilot_seq[0] * np.array(self._pilot_polarity)
        beta = np.angle(np.sum(np.conj(sym[PILOT_IND] * seq)))
        # amp = np.abs(np.average(sym[PILOT_IND] * seq))

        self._pilot_seq.rotate(-1)
        if self._ht:
            self._pilot_polarity.rotate(-1)
            return sym[HT_DATA_IND] * np.exp(1j * beta)
        else:
            return sym[DATA_IND] * np.exp(1j * beta)

    def switch_ht(self):
        # Skip HT-STS
        self._buffer.advance(HT_STS_LEN)
        # Skip GI
        self._buffer.advance(CP_LEN)
        # HT-LTS
        ht_lts = self._buffer.read(HT_LTS_LEN)
        ht_lts_f = np.fft.fftshift(np.fft.fft(ht_lts))
        self._h_est = HT_LTS_F / ht_lts_f

        self._ht = True

    def _coarse_cfo_estimate(self):
        sts_seq_1 = self._buffer.window(1 * STS_LEN, 7 * STS_LEN)
        sts_seq_2 = self._buffer.window(2 * STS_LEN, 7 * STS_LEN)
        cfo_est_sts = np.angle(np.sum(sts_seq_2 * np.conj(sts_seq_1)))
        cfo_est_sts = -cfo_est_sts / (2 * np.pi * STS_LEN / FS)
        return cfo_est_sts

    def _fine_cfo_estimate(self):
        lts1 = self._buffer.window(0, LTS_LEN)
        lts2 = self._buffer.window(LTS_LEN, LTS_LEN)
        cfo_est_lts = np.angle(np.sum(lts2 * np.conj(lts1)))
        cfo_est_lts = -cfo_est_lts / (2 * np.pi * LTS_LEN / FS)
        return cfo_est_lts


class Decoder:
    def __init__(self, samples):
        self._buffer = SampleBuffer(samples)

    def descramble(self, bits):
        x = [0] * 7
        x[0] = bits[2] ^ bits[6]
        x[1] = bits[1] ^ bits[5]
        x[2] = bits[0] ^ bits[4]
        x[3] = x[0] ^ bits[3]
        x[4] = x[1] ^ bits[2]
        x[5] = x[2] ^ bits[1]
        x[6] = x[3] ^ bits[0]

        out_bits = []
        for _, b in enumerate(bits):
            feedback = x[6] ^ x[3]
            out_bits.append(feedback ^ b)
            x = [feedback] + x[:-1]

        return out_bits

    def time_sync(self):
        samples = self._buffer.window(0, 350)
        corr = np.abs(np.correlate(samples, LTS_T, mode="valid"))
        peaks = np.argsort(corr)[-2:]
        peak1 = min(peaks)
        peak2 = max(peaks)
        if peak2 - peak1 != 64 or peak1 - 32 - 160 < 0:
            raise Exception("Time sync failed.")
        self._buffer.advance(peak1 - 32 - 160)

    def decode(self):
        self.time_sync()

        estimator = ChannelEstimator(self._buffer)

        # Decode one OFDM symbol for Legacy Signal
        demod = Demodulator(n_bpsc=1)
        deintl = Deinterleaver(n_bpsc=1, n_cbps=48)
        dot11_codec = FastViterbi(7, [0o133, 0o171])

        signal_sym = estimator.next_symbol()
        signal_raw_bits = demod.demodulate(signal_sym)
        signal_coded_bits = deintl.deinterleave(signal_raw_bits)
        signal_bits = dot11_codec.decode(signal_coded_bits)

        signal = LegacySignal(signal_bits)

        # Decode next two OFDM symbols (LT Signal) to detect potential 802.11n packets
        ht_signal_raw_bits = []

        for _ in range(2):
            ht_signal_sym = estimator.next_symbol()
            # LT Signal using QBPSK rotate back
            ht_signal_sym *= -1j
            ht_signal_raw_bits.extend(demod.demodulate(ht_signal_sym))

        ht_signal_coded_bits = deintl.deinterleave(ht_signal_raw_bits)
        ht_signal_bits = dot11_codec.decode(ht_signal_coded_bits)

        try:
            signal = HTSignal(ht_signal_bits)
            estimator.switch_ht()
        except Exception:
            # Not a 802.11n packets rollback
            self._buffer.advance(-160)
            estimator._pilot_seq.rotate(2)

        # Start decoding data symbols
        n_bpsc, n_cbps, n_dbps, puncpat = signal.mcs

        demod = Demodulator(n_bpsc=n_bpsc)
        deintl = Deinterleaver(n_bpsc=n_bpsc, n_cbps=n_cbps, ht=signal.ht)
        dot11_codec = FastViterbi(7, [0o133, 0o171], puncpat)

        n_service = 16
        n_bytes = signal.length
        n_sym = math.ceil((n_service + 8 * n_bytes + 6) / n_dbps)

        data_raw_bits = []

        for _ in range(n_sym):
            data_sym = estimator.next_symbol()
            data_raw_bits.extend(demod.demodulate(data_sym))

        data_coded_bits = deintl.deinterleave(data_raw_bits)
        data_bits = dot11_codec.decode(data_coded_bits)
        data_bits = self.descramble(data_bits)
        data_bytes = np.packbits(data_bits[n_service:], bitorder="little")[:n_bytes].tobytes()

        return data_bytes


class Deinterleaver:
    def __init__(self, n_bpsc=1, n_cbps=48, ht=False):
        n_col = 13 if ht else 16
        n_row = (4 * n_bpsc) if ht else (3 * n_bpsc)

        s = max(n_bpsc // 2, 1)

        first_perm = np.zeros(n_cbps, np.int32)
        for j in range(0, n_cbps):
            first_perm[j] = (s * (j // s)) + ((j + (n_col * j) // n_cbps) % s)

        second_perm = np.zeros(n_cbps, np.int32)
        for i in range(0, n_cbps):
            second_perm[i] = n_col * i - (n_cbps - 1) * (i // n_row)

        self.n_cbps = n_cbps
        self._perm = second_perm[first_perm[np.arange(n_cbps)]]

    def deinterleave(self, in_bits):
        perm = np.tile(self._perm, len(in_bits) // self.n_cbps + 1)[: len(in_bits)]
        perm += np.arange(len(in_bits)) // self.n_cbps * self.n_cbps

        out_bits = np.zeros(len(in_bits), np.int32)
        out_bits[perm] = in_bits
        return out_bits
