from signalprocessingbank.signalscharacterisation import SignalsFeatures


def transform_data(data, transformation):
    """
    Short time fourier is one option
    :param data:
    :param transformation:
    :return:
    """
    return None

def characterise(data, fetures_indices):
    settings = {"energy_window_size": 10, "sampling_freq": 100, "spectral_edge_tfreq": 40,
                "spectral_edge_power_coef": 0.5, "corr_type": "pearson", "autocorr_n_lags": 10
        , "hjorth_fd_k_max": 3, "dfa_overlap": False, "dfa_order": 1, "autoreg_lag": 10,
                "max_xcorr_downsample_rate": 1, "max_xcorr_lag": 20, "freq_hramonies_max_freq": 48,
                "dfa_gpu": True}
    res = []
    for x in data:
        res.append(SignalsFeatures.call_features_by_indexes(fetures_indices, x, settings, normalise=1))

    return res
