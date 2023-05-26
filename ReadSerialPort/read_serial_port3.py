import csv
import numpy as np
import pandas as pd
from scipy import stats, signal


# Read csv file (there is 2 float numbers in each row seperated by comma) seperate and write them to 2 different lists

def read_csv(file):
    bpm_data = []
    gsr_data = []
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            bpm_data.append(float(row[0]))
            gsr_data.append(float(row[1]))
        return bpm_data, gsr_data


def hrv_features(bpm_data, fs):
    # Calculate time-domain HRV metrics
    features = {}
    bpm_data = np.array(bpm_data)
    rr_intervals = 60000 / np.array(bpm_data)

    # mean of RR intervals
    features['mean_rr'] = np.mean(rr_intervals)
    # standard deviation of RR intervals (SDNN)
    features['sdnn'] = np.std(rr_intervals)
    # root mean square of differences of successive RR intervals (RMSSD)
    features['rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    # standard deviation of differences between all successive RR intervals (SDSD)
    features['sdsd'] = np.std(np.diff(rr_intervals))
    # standard deviation of RR intervals divided by RMSSD (SDRR_RMSSD)
    features['sdrr_rmssd'] = features['sdnn'] / features['rmssd']
    # percentage of successive RR interval differences greater than 50 ms (pNN50)
    features['pnn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100
    # percentage of successive RR interval differences greater than 25 ms (pNN25)
    features['pnn25'] = np.sum(np.abs(np.diff(rr_intervals)) > 25) / len(rr_intervals) * 100
    # SD1 and SD2, the standard deviations of points perpendicular/along the line of identity in the Poincare plot
    diff_rr_intervals = np.diff(rr_intervals)
    features['sd1'] = np.sqrt(np.std(diff_rr_intervals, ddof=1) / 2)
    features['sd2'] = np.sqrt(2 * features['sdnn'] ** 2 - features['sd1'] ** 2)
    # Skewness and Kurtosis
    features['skew'] = stats.skew(rr_intervals)
    features['kurt'] = stats.kurtosis(rr_intervals)
    cols = ["MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD", "HR",
            "pNN25", "pNN50", "SD1", "SD2", "KURT", "SKEW", "MEAN_REL_RR", "MEDIAN_REL_RR",
            "SDRR_REL_RR", "RMSSD_REL_RR", "SDSD_REL_RR", "SDRR_RMSSD_REL_RR",
            "KURT_REL_RR", "SKEW_REL_RR", "VLF", "VLF_PCT", "LF", "LF_PCT", "LF_NU",
            "HF", "HF_PCT", "HF_NU", "TP", "LF_HF", "HF_LF", "GSR_mean_gsr", "GSR_std_gsr",
            "GSR_max_mean_gsr_diff", "GSR_min_mean_gsr_diff", "GSR_gsr_mode", "GSR_gsr_skewness",
            "GSR_gsr_kurtosis", "GSR_gsr_max_mode_diff", "GSR_num_peaks", "GSR_time_between_peaks",
            "GSR_mean_first_derivative_gsr", "GSR_std_first_derivative_gsr", "GSR_mean_second_derivative_gsr",
            "GSR_std_second_derivative_gsr", "sampen", "higuci", "datasetId", "condition"]
    # Calculate frequency-domain HRV metrics
    f, Pxx = signal.welch(rr_intervals, fs=fs)
    vlf = (f <= 0.04)
    lf = (0.04 <= f) & (f <= 0.15)
    hf = (0.15 < f) & (f <= 0.4)

    # power in VLF band (<= 0.04 Hz)
    features['vlf'] = np.trapz(Pxx[vlf], f[vlf])
    # power in LF band (0.04-0.15 Hz)
    features['lf'] = np.trapz(Pxx[lf], f[lf])
    # power in HF band (0.15-0.4 Hz)
    features['hf'] = np.trapz(Pxx[hf], f[hf])
    # Total power in all bands
    features['total_power'] = features['vlf'] + features['lf'] + features['hf']
    # LF/HF ratio
    features['lf_hf_ratio'] = features['lf'] / features['hf']
    # Normalized units of LF and HF
    features['lf_nu'] = features['lf'] / (features['total_power'] - features['vlf']) * 100
    features['hf_nu'] = features['hf'] / (features['total_power'] - features['vlf']) * 100
    # HF/LF ratio
    features['hf_lf_ratio'] = features['hf'] / features['lf']

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
    features['skew_rel_rr'] = stats.skew(rr_rel_intervals)
    features['kurt_rel_rr'] = stats.kurtosis(rr_rel_intervals)

    # Percentage of power in VLF, LF, and HF bands
    features['vlf_pct'] = features['vlf'] / features['total_power'] * 100
    features['lf_pct'] = features['lf'] / features['total_power'] * 100
    features['hf_pct'] = features['hf'] / features['total_power'] * 100

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


def generate_synthetic_gsr(condition, duration, sampling_rate):
    # number of data points
    n_samples = duration * 60 * sampling_rate

    # set the base range and variation scale depending on the condition
    base_range = (80, 600)
    initial_range = (90, 450)
    middle_range = ((initial_range[0] + initial_range[1]) / 2)

    rare_situation = np.random.choice([True, False], p=[1 / 8, 7 / 8])  # 1/8 chance of a rare situation

    if condition == 'no stress' or rare_situation:
        variation_scale = 2  # smaller variations
    else:
        variation_scale = 6  # larger variations for stress

    # initialize the GSR list with a random starting value
    initial_value = np.random.normal(loc=middle_range, scale=variation_scale)
    initial_value = min(max(initial_value, initial_range[0]),
                        initial_range[1])  # ensure the initial value is within the initial range
    gsr = [initial_value]

    # set initial trend direction and duration
    trend_directions = [-1, 0, 1]
    if condition != 'no stress' or rare_situation:  # for stress condition or rare situations only allow upward or no change trends
        trend_directions.remove(-1)

    trend_direction = np.random.choice(trend_directions)
    trend_duration = int(np.random.normal(35, 10))  # Gaussian distribution around 35 with a standard deviation of 10

    # set sub-trend for 'no change' trend direction
    sub_trend_direction = np.random.choice([-1, 1])  # randomly select either slight increase or decrease
    sub_trend_scale = 0.5  # the scale of the sub-trend is much smaller than a full trend

    trend_count = 0  # count of how many data points have been generated with the current trend

    # generate the GSR data
    for _ in range(n_samples - 1):  # -1 because we already have the initial value
        last_value = gsr[-1]

        # determine if a peak should occur
        peak_ratio = np.random.uniform(1 / 15, 1 / 10)
        peak_occurrence = np.random.choice([True, False], p=[peak_ratio, 1 - peak_ratio])

        if peak_occurrence:
            # create a peak by inverting the trend direction for one data point
            peak_direction = -trend_direction
        else:
            peak_direction = trend_direction

        # determine the change for the next value
        if peak_direction == 0:  # no change
            change = sub_trend_direction * np.random.uniform(0, sub_trend_scale)  # apply sub-trend
        elif peak_direction == 1:  # increase
            change = abs(np.random.normal(0, variation_scale))  # always positive
        else:  # decrease
            change = -abs(np.random.normal(0, variation_scale))  # always negative

        # add the change to the last value to get the new value
        new_value = last_value + change

        # check if we are close to thresholds
        if abs(base_range[0] - new_value) < 30 or abs(base_range[1] - new_value) < 30:
            # Slowly increase or decrease
            new_value += np.random.uniform(-2, 2)

        # ensure the new value is within the base range
        new_value = min(max(new_value, base_range[0]), base_range[1])

        gsr.append(new_value)

        trend_count += 1
        if trend_count == trend_duration:  # if the current trend has reached its duration
            # reset the trend count
            trend_count = 0

            # select a new trend direction and duration
            previous_trend_direction = trend_direction
            trend_direction = np.random.choice([d for d in trend_directions if
                                                d != previous_trend_direction])  # select a new trend direction that is different from the previous one

            # if previous trend was 'no change', choose a new sub-trend direction
            if previous_trend_direction == 0:
                sub_trend_direction = np.random.choice([-1, 1])  # randomly select either slight increase or decrease

            trend_duration = int(np.random.normal(35, 10))  # select a new trend duration

    return gsr


def calculate_features(gender):
    # Read BPM and GSR data
    bpm_data, gsr_data = read_csv(
        "/Users/erguncan/BAU-Capstone-Lie-Detector/ReadSerialPort/yousif1684591776.1853743raw_data.csv")

    # Calculate HRV features
    hrv = hrv_features(bpm_data, 1)
    # Calculate additional HRV features
    # hrv_mode = stats.mode(bpm_data)[0][0]  # Most frequent BPM value (Mode)

    # Load both CSV files
    dataset_df = pd.read_csv('dataset.csv')
    test_df = pd.read_csv('test.csv')

    # Concatenate the two dataframes
    full_df = pd.concat([dataset_df, test_df])

    # Prepare to append GSR features
    gsr_features_dataset = []

    for i, row in full_df.iterrows():
        synthetic_gsr = generate_synthetic_gsr(row['condition'], 2, 1)  # 2 minutes, 1 sample per second

        gsr_features = calculate_gsr_features(synthetic_gsr)
        gsr_features_dataset.append(gsr_features)

        # Print progress every 10,000 rows
        if i % 10000 == 0:
            print(f'Progress: {i} rows processed.')

    # Convert GSR features list to dataframe
    gsr_features_df = pd.DataFrame(gsr_features_dataset)

    # Define the new columns
    gsr_features_df.columns = ['GSR_' + str(i) for i in gsr_features_df.columns]

    # Insert the new columns to the original dataframe
    for i, column in enumerate(gsr_features_df.columns):
        full_df.insert(full_df.columns.get_loc("HF_LF") + i + 1, column, gsr_features_df[column])

    # Save the new dataset
    full_df.to_csv('new_dataset.csv', index=False)


if __name__ == '__main__':
    # read_serial()
    calculate_features(1)
