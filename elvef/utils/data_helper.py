#!/usr/bin/env python

import random
from typing import Optional

import numpy as np


def trim_np_data(
    data: np.ndarray,
    sampling_frequency: int,
    trim_mode: int = 0,
) -> np.ndarray:
    """trim numpy format time series data

    Args:
        data (np.ndarray): time series data (channels, seqlen).
        sampling_frequency (int): sampling frequency [Hz].
        trim_mode (int): trim option. trim_mode=0 returns the
            input data, trim_mode=1 makes 8 [sec] data.
            Defaults to 0.

    Returns:
        np.ndarray: trimmed time series data

    """
    if trim_mode == 0:
        return data

    duration = data.shape[-1] / sampling_frequency

    if duration >= 13:
        data = data[:, int(5 * sampling_frequency) : int(13 * sampling_frequency)]
    elif duration >= 8:
        data = data[:, -int(8 * sampling_frequency) :]
    else:
        raise ValueError()
    return data


def make_segments(
    data: np.ndarray,
    segment_seqlen: int,
    n_expand: int,
    stride_mode: str = "random",
    stride: Optional[int] = None,
) -> np.ndarray:
    """extract multiple time-series segments

    Args:
        data (np.ndarray): input data (channels, seqlen).
        segment_seqlen (int): segment sequence length
        n_expand (int): the number of segments.
        stride_mode (str): extraction mode. Defaults to "random".
        stride (Optional[int]): stride window param.
            Defaults to None (compute a fixed stride size on-the fly).

    Returns:
        np.ndarray: segments (n_expand, channels, seqlen).

    """

    seqlen = data.shape[-1]

    segments = np.zeros((n_expand, data.shape[0], segment_seqlen), dtype=np.float32)

    if stride_mode == "random":

        for idx in range(n_expand):
            start_idx = random.randint(0, seqlen - segment_seqlen)
            segment = data[:, start_idx : start_idx + segment_seqlen]
            segments[idx, :, : segment.shape[-1]] = segment

    elif stride_mode == "equally_spaced_stride":

        if stride is None:
            stride = int((seqlen - segment_seqlen) / (n_expand - 1))

        for idx in range(n_expand):
            start_idx = stride * idx
            segment = data[:, start_idx : start_idx + segment_seqlen]
            segments[idx, :, : segment.shape[-1]] = segment

    else:
        raise NotImplementedError

    return segments
