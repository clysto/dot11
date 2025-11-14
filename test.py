import numpy as np
import scipy

from dot11decoder import Decoder


def awgn(samples, snr):
    power_signal = np.mean(np.abs(samples) ** 2)
    snr_linear = 10 ** (snr / 10.0)
    power_noise = power_signal / snr_linear
    noise_std = np.sqrt(power_noise / 2)
    noise = noise_std * (np.random.randn(*samples.shape) + 1j * np.random.randn(*samples.shape))
    return samples + noise


if __name__ == "__main__":
    for v in ["a", "n"]:
        for n in range(0, 8):
            mat = scipy.io.loadmat(f"data/80211{v}-mcs{n}.mat", squeeze_me=True)
            samples = mat["waveStruct"]["waveform"].item()
            samples = samples.astype(np.complex64)

            samples = awgn(samples, 30)
            samples *= np.exp(1j * 2 * np.pi * 50e3 * (np.arange(len(samples)) / 20e6))

            decoder = Decoder(samples)
            try:
                psdu = decoder.decode()
                try:
                    for b in psdu:
                        assert b == 0
                    print(f"802.11{v} MCS {n} test passed ✓")
                except Exception:
                    print(f"802.11{v} MCS {n} test failed ✗")
            except Exception as e:
                print(e)
                print(f"802.11{v} MCS {n} test failed ✗")
