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


def run_decoder(variant, mcs):
    mat = scipy.io.loadmat(f"data/80211{variant}-mcs{mcs}.mat", squeeze_me=True)
    samples = mat["waveStruct"]["waveform"].item().astype(np.complex64)

    samples = awgn(samples, 30)
    samples *= np.exp(1j * 2 * np.pi * 50e3 * (np.arange(len(samples)) / 20e6))

    decoder = Decoder(samples)
    return decoder.decode()


@pytest.mark.parametrize("mcs", range(8), ids=lambda m: f"MCS{m}")
@pytest.mark.parametrize("variant", ["a", "n"], ids=lambda x: "802.11a/g" if x == "a" else "802.11n")
def test_decoder(variant, mcs):
    psdu = run_decoder(variant, mcs)
    assert not any(psdu)
