import numpy as np
import math

N_LINKS = 10
N_ELEMENTS = 64
ALPHA_MAX_DB = -55
ALPHA_MIN_DB = -60
TRANSMIT_POWER_MAX = 10
TRANSMIT_POWER_MIN = 8
K = 10                 # Rician Factor
sigma2 = 1             # Noise Power
ALPHA_MAX = np.power(10, (ALPHA_MAX_DB/10))
ALPHA_MIN = np.power(10, (ALPHA_MIN_DB/10))


def get_channel():
    large_scale_fading_a = np.random.rand(N_LINKS, 1) * (ALPHA_MAX - ALPHA_MIN) + ALPHA_MIN
    large_scale_fading_b = np.random.rand(N_LINKS, 1) * (ALPHA_MAX - ALPHA_MIN) + ALPHA_MIN

    p_t = np.random.rand(N_LINKS, 1) * (TRANSMIT_POWER_MAX - TRANSMIT_POWER_MIN) + TRANSMIT_POWER_MIN  # Transmit Power

    aoa = np.random.rand(N_LINKS, 1) * 2 * math.pi
    aod = np.random.rand(N_LINKS, 1) * 2 * math.pi

    hai_los = np.exp(np.kron(1j*2*math.pi*0.5*np.sin(aod), np.linspace(0, N_ELEMENTS-1, N_ELEMENTS)))
    hbi_los = np.exp(np.kron(1j*2*math.pi*0.5*np.sin(aoa), np.linspace(0, N_ELEMENTS-1, N_ELEMENTS)))

    hai_nlos = np.sqrt(0.5) * np.random.randn(N_LINKS, N_ELEMENTS) + 1j * np.sqrt(0.5) * np.random.randn(N_LINKS, N_ELEMENTS)
    hbi_nlos = np.sqrt(0.5) * np.random.randn(N_LINKS, N_ELEMENTS) + 1j * np.sqrt(0.5) * np.random.randn(N_LINKS, N_ELEMENTS)

    hai = np.sqrt(K/(K+1)) * hai_los + np.sqrt(1/(K+1)) * hai_nlos
    hbi = np.sqrt(K/(K+1)) * hbi_los + np.sqrt(1/(K+1)) * hbi_nlos
    return large_scale_fading_a, large_scale_fading_b, p_t, hai, hbi, aoa, aod

# theta_vector = np.random.rand(1, N_ELEMENTS) * 2 * math.pi


def execute_channel(large_scale_fading_a, large_scale_fading_b, p_t, hai, hbi, theta_vector):
    theta_matrix = np.diag(np.exp(1j * theta_vector))
    snr = np.zeros((N_LINKS, 1))
    for i in range(N_LINKS):
        snr_tmp1 = np.power(np.abs(hbi[i, :].dot(theta_matrix).dot(hai[i, :])), 2)
        snr_tmp2 = 0
        for j in range(N_LINKS):
            if i != j:
                snr_tmp2 = snr_tmp2 + p_t[j, 0] * large_scale_fading_a[j, 0] * large_scale_fading_b[i, 0] * np.power(np.abs(hbi[i, :].dot(theta_matrix).dot(hai[j, :])), 2)
        snr[i, 0] = np.log10(snr_tmp1 / (snr_tmp2 + sigma2))
    # print('current reward: ', snr.sum())
    return snr.sum()

