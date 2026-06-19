#!/usr/bin/env python

import dataclasses
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from neurokit2.ecg.ecg_findpeaks import _ecg_findpeaks_neurokit
from scipy import signal


@dataclasses.dataclass
class EcgPeaksOnsetsOffsetsData:
    """Data structure for Peaks, Onsets, Offsets of ECG

    Attributes:
        r_peaks (np.ndarray): indices for R wave position
        p_onsets (np.ndarray): indices for P onset position
        p_offsets (np.ndarray): indices for P offset position
        qrs_onsets (np.ndarray): indices for QRS onset position
        qrs_offsets (np.ndarray): indices for QRS offset position
        t_offsets (np.ndarray): indices for T offset position

    Notes:
        data type of Attributes is float if NaN is included for
            unavailable index, otherwise int
        data type of instance variables is int (NaN is replaced
            with -1)
    """

    r_peaks: np.ndarray
    p_onsets: np.ndarray
    p_offsets: np.ndarray
    qrs_onsets: np.ndarray
    qrs_offsets: np.ndarray
    t_offsets: np.ndarray

    def __post_init__(self) -> None:
        self.align_parameters()

    def align_parameters(self) -> None:
        """align rhythm parameters"""

        length_array = np.array(
            [
                len(self.r_peaks),
                len(self.p_onsets),
                len(self.p_offsets),
                len(self.qrs_onsets),
                len(self.qrs_offsets),
                len(self.t_offsets),
            ]
        )

        if len(np.unique(length_array)) > 1:
            # ID of rhythm parameter of which elements need to be inserted
            n_r_peaks = np.max(length_array)
            parameter_id_insertion = np.ravel(np.where(length_array < n_r_peaks))

            parameter_list = [
                self.p_onsets,
                self.p_offsets,
                self.qrs_onsets,
                self.qrs_offsets,
                self.t_offsets,
            ]
            # flag_list =
            # [ True (p_onsets is prior to r_peaks), True (p_offsets is prior to r_peaks),
            #   True (qrs_onsets is prior to r_peaks),
            #   False (qrs_offsets is posterior to r_peaks), False (t_offsets is posterior to r_peaks) ]
            flag_list = [True, True, True, False, False]

            # initialize output_buffer with rhythm parameters which are NOT specified
            # by parameter_id_insertion
            output_buffer = self.initialize_output_buffer(
                parameter_id_insertion,
                n_r_peaks,
            )

            # insert missing elements of rhythm parameters which are specified by
            # parameter_id_insertion
            for idx in parameter_id_insertion:
                output_buffer[idx - 1] = self.insert_missing_elements(
                    parameter_list[idx - 1],
                    flag_list[idx - 1],
                )

            # move rhythm parameters, which are specified by parameter_id_insertion,
            # from output_buffer
            self.move_rhythm_parameter(parameter_id_insertion, output_buffer)

    def process_nan_elements(self, parameter: np.ndarray) -> np.ndarray:
        """replace NaN elements with -1

        Args:
            parameter (np.ndarray): rhythm parameter
                (r_peaks, p_onsets, p_offsets, qrs_onsets,
                qrs_offsets,t_offsets)

        Returns:
            output (np.ndarray): processed rhythm parameter
        """
        if parameter != np.array([]):
            output = np.where(np.isnan(parameter), -1, parameter)
        else:
            output = parameter
        return output

    def initialize_output_buffer(
        self,
        parameter_id_insertion: np.ndarray,
        n_r_peaks: int,
    ) -> np.ndarray:
        """initialize buffer to store modified rhythm parameters

        Args:
            parameter_id_insertion (np.ndarray): ID of rhythm parameter
                of which elements need to be inserted
                    1: p_onsets, 2: p_offsets, 3: qrs_onsets,
                    4: qrs_offsets, 5: t_offsets
            n_r_peaks (int): the number of r_peaks

        Returns:
            output_buffer (np.ndarray): buffer to store modified rhythm
                parameters

        Notes:
            Indices which are NOT contained in parameter_id_insertion
                are referred
        """
        # shape is (n_param, n_r_peaks)
        output_buffer = np.full((5, n_r_peaks), -1).astype(int)

        # set rhythm parameters if missing elements does not exist
        if 1 not in parameter_id_insertion:
            output_buffer[0] = self.process_nan_elements(self.p_onsets)
        if 2 not in parameter_id_insertion:
            output_buffer[1] = self.process_nan_elements(self.p_offsets)
        if 3 not in parameter_id_insertion:
            output_buffer[2] = self.process_nan_elements(self.qrs_onsets)
        if 4 not in parameter_id_insertion:
            output_buffer[3] = self.process_nan_elements(self.qrs_offsets)
        if 5 not in parameter_id_insertion:
            output_buffer[4] = self.process_nan_elements(self.t_offsets)

        return output_buffer

    def insert_missing_elements(
        self,
        input: np.ndarray,
        prior_posterior_flag: bool,
    ) -> np.ndarray:
        """insert missing elements (=-1) of input ndarray, which are
            supposed to locate prior/posterior to r_peaks ndarray

        Args:
            input (np.ndarray): ndarray for adding missing elements
                p_onsets, p_offsets, qrs_onsets, qrs_offsets
                and t_offsets
            prior_posterior_flag (bool):
                True: input needs to be aligned prior to r_peaks
                    e.g. p_onsets, p_offsets, qrs_onsets
                False: input needs to be aligned posterior to r_peaks
                    e.g. qrs_offsets, t_offsets

        Returns:
            output (np.ndarray): modified rhythm parameter

        Notes:
            Indices which are contained in parameter_id_insertion
                are referred
        """
        # initialize output with -1
        output = np.full(len(self.r_peaks), -1)
        if prior_posterior_flag:
            # delete np.NaN and >= self.r_peaks[-1]
            max_position = float(np.max(self.r_peaks))
            input_valid = input[input < max_position]
            # each element of input needs to locate prior to reference
            for position in input_valid:
                output[np.min(np.where(self.r_peaks > position))] = position
        else:
            # delete np.NaN
            input_valid = input[~np.isnan(input)]
            # delete <= self.r_peaks[0]
            min_position = float(np.min(self.r_peaks))
            input_valid = input_valid[input_valid > min_position]
            # each element of input needs to locate posterior to reference
            for position in input_valid:
                output[np.max(np.where(self.r_peaks < position))] = position

        return output

    def move_rhythm_parameter(
        self,
        parameter_id_insertion: np.ndarray,
        output_buffer: np.ndarray,
    ) -> None:
        """move rhythm parameters from output_buffer

        Args:
            parameter_id_insertion (np.ndarray): ID of rhythm parameter
                of which elements need to be inserted
                    1: p_onsets, 2: p_offsets, 3: qrs_onsets,
                    4: qrs_offsets, 5: t_offsets
            output_buffer (np.ndarray): buffer to store modified rhythm
                parameters
        """
        # modify rhythm parameters if missing elements are inserted
        if 1 in parameter_id_insertion:
            self.p_onsets = output_buffer[0]
        if 2 in parameter_id_insertion:
            self.p_offsets = output_buffer[1]
        if 3 in parameter_id_insertion:
            self.qrs_onsets = output_buffer[2]
        if 4 in parameter_id_insertion:
            self.qrs_offsets = output_buffer[3]
        if 5 in parameter_id_insertion:
            self.t_offsets = output_buffer[4]


def detect_r_peaks_nk(
    ecg_data: np.ndarray,
    sampling_rate: int,
) -> np.ndarray:
    """detect R-peaks using neurokit

    Args:
        ecg_data (np.ndarray): ECG data (seqlen,).
        sampling_frequency (int): sampling frequency [Hz].

    Returns:
        np.ndarray: detected r-peaks (n_peaks,).

    """

    ecg_data = nk.signal.signal_sanitize(ecg_data)

    r_peaks = np.array([])

    kwargs = {
        "sampling_rate": sampling_rate,
        "smoothwindow": 0.1,
        "avgwindow": 0.48,
        "gradthreshweight": 1.5,
        "minlenweight": 0.4,
        "mindelay": 0.2,
        "show": False,
    }

    try:
        r_peaks = _ecg_findpeaks_neurokit(ecg_data, **kwargs)
    except Exception:
        pass

    if len(r_peaks) == 0:
        try:
            r_peaks = _ecg_findpeaks_neurokit(-1.0 * ecg_data, **kwargs)
        except:
            pass

    assert not np.any(np.isnan(r_peaks))
    assert not np.any(r_peaks < 0)
    assert np.all(r_peaks < ecg_data.shape[0])

    return r_peaks


def delineate(
    ecg_data: np.ndarray,
    ecg_fs: int,
    r_peaks: np.ndarray,
    method: str = "dwt",
) -> EcgPeaksOnsetsOffsetsData:
    """detect P onset/offset and R onset/offset

    Note:
        - [ecg_delineate](https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-delineate)

    Args:
        ecg_data (np.ndarray): ECG data
        ecg_fs (int): sampling frequency [Hz]
        r_peaks (np.ndarray): detected R peaks [frames].
            The shape is (n_peaks,).
        method (str): nk2 delineation method. Defaults to "dwt".

    Returns:
        EcgPeaksOnsetsOffsetsData: P, QRS, and T peaks.

    """

    nk_result = nk.ecg_delineate(
        ecg_data,
        rpeaks=r_peaks,
        sampling_rate=ecg_fs,
        method=method,
        show=False,
        show_type="peaks",
    )[1]

    return EcgPeaksOnsetsOffsetsData(
        r_peaks,
        np.array(nk_result["ECG_P_Onsets"]),
        np.array(nk_result["ECG_P_Offsets"]),
        np.array(nk_result["ECG_R_Onsets"]),
        np.array(nk_result["ECG_R_Offsets"]),
        np.array(nk_result["ECG_T_Offsets"]),
    )


def plot_peaks(
    ecg_data: np.ndarray,
    ecg_fs: int,
    onset: EcgPeaksOnsetsOffsetsData,
    output_file: Path,
) -> None:
    """plot detected peaks

    Args:
        ecg_data (np.ndarray): ECG data
        ecg_fs (int): ECG sampling frequency [Hz]
        onset (EcgPeaksOnsetsOffsetsData): detected peaks
        output_file (Path): output fig file.

    """
    fig, axes = plt.subplots(figsize=(24, 6))

    # ECG signal
    duration = ecg_data.shape[0] / ecg_fs
    times = np.linspace(
        0,
        duration,
        ecg_data.shape[0],
        endpoint=False,
    )
    axes.plot(times, ecg_data)

    # peaks
    cmap = plt.get_cmap("tab10")
    for peaks, label, color, marker in [
        (onset.p_onsets, "p_onset", cmap(0), "."),
        (onset.p_offsets, "p_offset", cmap(1), ","),
        (onset.qrs_onsets, "qrs_onset", cmap(2), "^"),
        (onset.r_peaks, "r_peak", cmap(3), "*"),
        (onset.qrs_offsets, "qrs_offset", cmap(4), "1"),
        (onset.t_offsets, "t_offset", cmap(5), "x"),
    ]:
        peaks = peaks[~np.isnan(peaks)]
        axes.scatter(
            peaks / ecg_fs,
            ecg_data[peaks.astype(int)],
            label=label,
            color=color,
            marker=marker,
            s=50,
        )

    axes.minorticks_on()
    axes.grid(which="major", color="black", linestyle="-", linewidth=0.5)
    axes.grid(which="minor", color="gray", linestyle="--", linewidth=0.5)

    plt.legend()
    plt.title(output_file.stem)
    plt.savefig(output_file)
    plt.clf()
    plt.close()


def notch_filter(
    data: np.ndarray,
    sampling_frequency: int,
    notch_freq: float,
) -> np.ndarray:
    """apply notch filter

    Args:
        data (np.ndarray): signal (seqlen,).
        sampling_frequency (int): sampling frequency [Hz].
        notch_freq: float): notch frequency [Hz].

    Returns:
        np.ndarray: denoised signal.

    """
    numer, denom = signal.iirnotch(
        notch_freq,
        30.0,
        sampling_frequency,
    )
    return signal.filtfilt(numer, denom, data, axis=0).astype(np.float32)


def apply_notch_filters(
    ecg_data: np.ndarray,
    ecg_fs: int,
    notch_freqs: list[float],
) -> np.ndarray:
    """apply multiple notch filters

    Args:
        ecg_data (np.ndarray): ECG data (seqlen,).
        sampling_frequency (int): sampling frequency [Hz].
        notch_freqs (list[float]): notch frequencies [Hz].

    Returns:
        np.ndarray: denoised signal

    """
    for notch_freq in notch_freqs:
        ecg_data = notch_filter(ecg_data, ecg_fs, notch_freq)
    return ecg_data


def calculate_diff(
    peaks_1: np.ndarray, peaks_2: np.ndarray, sampling_frequency
) -> float:
    """calcuate difference [sec] between two peaks.

    Args:
        peaks_1 (np.ndarray): preceding peaks (n_peaks,).
        peaks_2 (np.ndarray): succeeding peaks (n_peaks,).
        sampling_frequency (int): sampling frequency [Hz].

    Returns:
        float: averaged time of difference [sec].

    """
    peaks_1 = np.where(peaks_1 == -1, np.nan, peaks_1)
    peaks_2 = np.where(peaks_2 == -1, np.nan, peaks_2)

    diff = peaks_2 - peaks_1
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0:
        return -1.0
    diff = diff / sampling_frequency
    return np.mean(diff)


def process(
    ecg_data: np.ndarray, ecg_fs: int, duration: float
) -> dict[str, Union[float, int]]:
    """detect peaks

    Args:
        ecg_data (np.ndarray): ECG data (seqlen,).
        ecg_fs (int): sampling frequency [Hz]
        duration (float): duration [sec] of ECG data.

    Returns:
        dict: detected parameters

    """

    assert ecg_fs == 500
    assert ecg_data.shape[0] == int(duration * ecg_fs)

    ecg_data = nk.signal_filter(
        signal=ecg_data,
        sampling_rate=ecg_fs,
        lowcut=0.5,
        method="butterworth",
        order=5,
    )
    ecg_data = apply_notch_filters(
        ecg_data,
        ecg_fs,
        [50.0, 60.0, 100.0, 120.0, 200.0, 240.0],
    )

    # 1. detect R-peaks (n_peaks,).
    r_peaks = detect_r_peaks_nk(ecg_data, ecg_fs)
    assert not np.any(np.isnan(r_peaks))
    assert not np.any(r_peaks < 0)
    assert np.all(r_peaks < ecg_data.shape[0])

    result: dict[str, Union[float, int]] = {}
    if len(r_peaks) == 0:
        return result

    try:
        onset = delineate(ecg_data, ecg_fs, r_peaks)
    except:
        onset = EcgPeaksOnsetsOffsetsData(
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
        )

    r_peaks = np.where(r_peaks == -1, np.nan, r_peaks)
    rr_times = np.diff(r_peaks)
    rr_times = rr_times[~np.isnan(rr_times)]
    rr_time = -1.0 if len(rr_times) < 2 else np.mean(rr_times) / ecg_fs

    pr_interval = calculate_diff(onset.p_onsets, onset.qrs_onsets, ecg_fs)
    qrs_width = calculate_diff(onset.qrs_onsets, onset.qrs_offsets, ecg_fs)

    return {
        "pr_interval": pr_interval,
        "qrs_width": qrs_width,
    }


def main() -> None:

    ecg_fs = 500

    for duration in [1.0, 8.0]:
        ecg_data = nk.ecg_simulate(
            length=int(duration * ecg_fs),
            duration=duration,
            sampling_rate=ecg_fs,
            random_state=0,
        )
        print(process(ecg_data, ecg_fs, duration))


if __name__ == "__main__":
    main()
