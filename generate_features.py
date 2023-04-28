import numpy as np


# "Min_BPM", "Max_BPM", "Mean_BPM", "HRV", "Min_GSR", "Max_GSR", "Mean_GSR", "GSR_Variability",
def calculate_features(bpm_data, gsr_data):
    # Calculate BPM features
    min_bpm = np.min(bpm_data)
    max_bpm = np.max(bpm_data)
    mean_bpm = np.mean(bpm_data)
    hrv = np.std(np.diff(bpm_data))

    # Calculate GSR features
    min_gsr = np.min(gsr_data)
    max_gsr = np.max(gsr_data)
    mean_gsr = np.mean(gsr_data)
    gsr_variability = np.std(gsr_data)
    # gsr_gradient = np.gradient(gsr_data)  # maybe this feature is not needed

    # Return newly calculated features
    return np.array([[1, min_bpm, max_bpm, mean_bpm, hrv, min_gsr, max_gsr, mean_gsr, gsr_variability]])
