import numpy as np
from scipy import stats, signal

def calculate_hrv_features(bpm_data, fs):
    # Calculate time-domain HRV metrics
    features = {}
    bpm_data = np.array(bpm_data)
    rr_intervals = 60000 / np.array(bpm_data)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # mean of RR intervals
    features['mean_rr'] = np.mean(rr_intervals)
    print("mean_rr: ", features['mean_rr'])
    # standard deviation of RR intervals (SDNN)
    features['sdnn'] = np.std(rr_intervals)
    print("sdnn: ", features['sdnn'])
    # root mean square of differences of successive RR intervals (RMSSD)
    features['rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    print("rmssd: ", features['rmssd'])
    # standard deviation of differences between all successive RR intervals (SDSD)
    features['sdsd'] = np.std(np.diff(rr_intervals))
    print("sdsd: ", features['sdsd'])
    # standard deviation of RR intervals divided by RMSSD (SDRR_RMSSD)
    features['sdrr_rmssd'] = features['sdnn'] / features['rmssd']
    print("sdrr_rmssd: ", features['sdrr_rmssd'])
    # percentage of successive RR interval differences greater than 25 ms (pNN25)
    features['pnn25'] = np.sum(np.abs(np.diff(rr_intervals)) > 25) / len(rr_intervals) * 100
    print("pnn25: ", features['pnn25'])
    # percentage of successive RR interval differences greater than 50 ms (pNN50)
    features['pnn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100
    print("pnn50: ", features['pnn50'])
    # SD1 and SD2, the standard deviations of points perpendicular/along the line of identity in the Poincare plot
    diff_rr_intervals = np.diff(rr_intervals)
    features['sd1'] = np.sqrt(np.std(diff_rr_intervals, ddof=1) / 2)
    print("sd1: ", features['sd1'])
    features['sd2'] = np.sqrt(2 * features['sdnn'] ** 2 - features['sd1'] ** 2)
    print("sd2: ", features['sd2'])
    # Skewness and Kurtosis
    features['kurt'] = stats.kurtosis(rr_intervals)
    print("kurt: ", features['kurt'])
    features['skew'] = stats.skew(rr_intervals)
    print("skew: ", features['skew'])


    # Calculate relative RR intervals
    rr_rel_intervals = np.diff(rr_intervals) / np.mean(rr_intervals)

    # mean of relative RR intervals
    features['mean_rel_rr'] = np.mean(rr_rel_intervals)
    # standard deviation of relative RR intervals
    features['sdrr_rel_rr'] = np.std(rr_rel_intervals)
    # root mean square of differences of successive relative RR intervals
    features['rmssd_rel_rr'] = np.sqrt(np.mean(np.square(np.diff(rr_rel_intervals))))
    # standard deviation of differences between all successive relative RR intervals
    features['sdsd_rel_rr'] = np.std(np.diff(rr_rel_intervals))
    # standard deviation of relative RR intervals divided by RMSSD of relative RR intervals
    features['sdrr_rmssd_rel_rr'] = features['sdrr_rel_rr'] / features['rmssd_rel_rr']
    # Skewness and Kurtosis of relative RR intervals
    features['kurt_rel_rr'] = stats.kurtosis(rr_rel_intervals)
    features['skew_rel_rr'] = stats.skew(rr_rel_intervals)

    # Calculate frequency-domain HRV metrics
    f, Pxx = signal.welch(rr_intervals, fs=fs)
    vlf = (f <= 0.04)
    lf = (0.04 <= f) & (f <= 0.15)
    hf = (0.15 < f) & (f <= 0.4)
    p_vlf = np.trapz(Pxx[vlf], f[vlf])
    p_lf = np.trapz(Pxx[lf], f[lf])
    p_hf = np.trapz(Pxx[hf], f[hf])
    # power in VLF band (<= 0.04 Hz)
    features['vlf'] = np.trapz(Pxx[vlf], f[vlf])
    # Percentage of power in VLF band
    features['vlf_pct'] = features['vlf'] / (p_vlf + p_lf + p_hf) * 100
    # power in LF band (0.04-0.15 Hz)
    features['lf'] = np.trapz(Pxx[lf], f[lf])
    # Percentage of power in LF band
    features['lf_pct'] = features['lf'] / (p_vlf + p_lf + p_hf) * 100
    # Normalized units of LF
    features['lf_nu'] = features['lf'] / ((p_vlf + p_lf + p_hf) - features['vlf']) * 100
    # power in HF band (0.15-0.4 Hz)
    features['hf'] = np.trapz(Pxx[hf], f[hf])
    # Percentage of power in HF band
    features['hf_pct'] = features['hf'] / (p_vlf + p_lf + p_hf) * 100
    # Normalized units of HF
    features['hf_nu'] = features['hf'] / ((p_vlf + p_lf + p_hf) - features['vlf']) * 100
    # Total power in all bands
    features['total_power'] = features['vlf'] + features['lf'] + features['hf']
    # LF/HF ratio
    features['lf_hf_ratio'] = features['lf'] / features['hf']
    # HF/LF ratio
    features['hf_lf_ratio'] = features['hf'] / features['lf']
    return features


def calculate_gsr_features(gsr_data):
    features = {}

    # Calculate GSR features
    features['mean_gsr'] = np.mean(gsr_data)
    features['std_gsr'] = np.std(gsr_data)
    features['max_mean_gsr_diff'] = np.abs(np.max(gsr_data) - features['mean_gsr'])
    features['min_mean_gsr_diff'] = np.abs(np.min(gsr_data) - features['mean_gsr'])

    # Calculate additional GSR features
    features['gsr_mode'] = stats.mode(gsr_data)[0][0]
    features['gsr_skewness'] = stats.skew(gsr_data)
    features['gsr_kurtosis'] = stats.kurtosis(gsr_data)
    features['gsr_max_mode_diff'] = np.abs(np.max(gsr_data) - features['gsr_mode'])

    # Calculate number of peaks and time between
    peaks, _ = signal.find_peaks(gsr_data)
    features['num_peaks'] = len(peaks)
    features['time_between_peaks'] = np.mean(np.diff(peaks)) if peaks.size > 1 else 0

    # Derivative features
    first_derivative_gsr = np.diff(gsr_data)
    second_derivative_gsr = np.diff(gsr_data, n=2)

    features['mean_first_derivative_gsr'] = np.mean(first_derivative_gsr)
    features['std_first_derivative_gsr'] = np.std(first_derivative_gsr)
    features['mean_second_derivative_gsr'] = np.mean(second_derivative_gsr)
    features['std_second_derivative_gsr'] = np.std(second_derivative_gsr)

    return features


def calculate_features(bpm_data, gsr_data):
    # Calculate HRV features
    hrv_features = calculate_hrv_features(bpm_data, fs=1)
    # Calculate GSR features
    gsr_features = calculate_gsr_features(gsr_data)
    # Combine features into 2D numpy array
    features = np.array([[list(hrv_features.values()) + list(gsr_features.values())]])
    # print number of features
    print('Number of features: {}'.format(features.shape[1]))
    # Return newly calculated features
    return features
