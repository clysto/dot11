import numpy as np
import pytest
import scipy

from dot11decoder import Decoder


def awgn(samples, snr):
    power_signal = np.mean(np.abs(samples) ** 2)
    snr_linear = 10 ** (snr / 10.0)
    power_noise = power_signal / snr_linear
    noise_std = np.sqrt(power_noise / 2)
    noise = noise_std * (np.random.randn(*samples.shape) + 1j * np.random.randn(*samples.shape))
    return samples + noise


def channel(sig, time_offset, theta, snr, freq_offset):
    # add time offset
    samples = np.concat((np.zeros(time_offset, dtype=np.complex64), sig))
    # add phase rotation
    samples *= np.exp(1j * theta)
    # add noise
    samples = awgn(samples, snr)
    # add frequency offset
    samples *= np.exp(1j * 2 * np.pi * freq_offset * (np.arange(len(samples)) / 20e6))
    return samples


def run_decoder(variant, mcs):
    mat = scipy.io.loadmat(f"data/80211{variant}-mcs{mcs}.mat", squeeze_me=True)
    samples = mat["waveStruct"]["waveform"].item().astype(np.complex64)

    time_offset = np.random.randint(0, 15)
    theta = np.random.uniform(0, 2 * np.pi)
    snr = 30
    freq_offset = 50e3

    samples = channel(samples, time_offset, theta, snr, freq_offset)

    decoder = Decoder(samples)
    return decoder.decode()


@pytest.mark.parametrize("mcs", range(8), ids=lambda m: f"MCS{m}")
@pytest.mark.parametrize("variant", ["a", "n"], ids=lambda x: "802.11a/g" if x == "a" else "802.11n")
def test_decoder(variant, mcs):
    psdu = run_decoder(variant, mcs)
    assert not any(psdu)


def test_continues_decode():
    mat = scipy.io.loadmat("data/80211n-mcs0.mat", squeeze_me=True)
    samples = mat["waveStruct"]["waveform"].item().astype(np.complex64)
    sig = np.concat(
        (
            np.zeros(3000, dtype=np.complex64),
            samples,
            np.zeros(3000, dtype=np.complex64),
        )
    )
    sig = np.tile(sig, 16)

    theta = np.random.uniform(0, 2 * np.pi)
    snr = 15
    freq_offset = 50e3

    sig = channel(sig, 0, theta, snr, freq_offset)

    decoder = Decoder(sig)

    for psdu in decoder.decode_next():
        assert not any(psdu)
