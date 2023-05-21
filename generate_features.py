import numpy as np
import scipy.signal


# "Min_BPM", "Max_BPM", "Mean_BPM", "HRV", "Min_GSR", "Max_GSR", "Mean_GSR", "GSR_Variability",
def calculate_features(gender, bpm_data, gsr_data):
    # Calculate BPM features
    # min_bpm = np.min(bpm_data)  # commented out as per your request
    # max_bpm = np.max(bpm_data)  # commented out as per your request
    mean_bpm = np.mean(bpm_data)
    hrv = np.std(np.diff(bpm_data))
    std_bpm = np.std(bpm_data)
    rate_change_bpm = (bpm_data[-1] - bpm_data[0]) / len(bpm_data)

    # Calculate GSR features
    # min_gsr = np.min(gsr_data)  # commented out as per your request
    # max_gsr = np.max(gsr_data)  # commented out as per your request
    mean_gsr = np.mean(gsr_data)
    std_gsr = np.std(gsr_data)
    rate_change_gsr = (gsr_data[-1] - gsr_data[0]) / len(gsr_data)
    peaks, _ = scipy.signal.find_peaks(gsr_data)
    num_peaks = len(peaks)
    avg_peak_height = np.mean(gsr_data[peaks]) if peaks.size > 0 else 0
    time_between_peaks = np.mean(np.diff(peaks)) if peaks.size > 1 else 0
    # gsr_gradient = np.gradient(gsr_data)  # maybe this feature is not needed

    # Calculate Cross-correlation of GSR and HR
    cross_corr = np.correlate(gsr_data - np.mean(gsr_data), bpm_data - np.mean(bpm_data), mode='same')

    # Return newly calculated features
    return np.array([[gender, mean_bpm, hrv, std_bpm, rate_change_bpm, mean_gsr, std_gsr, rate_change_gsr, num_peaks,
                      avg_peak_height, time_between_peaks, cross_corr]])

