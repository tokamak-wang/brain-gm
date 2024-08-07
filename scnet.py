import torch
import numpy as np
from random import randrange



def get_minibatch_sc(minibatch_timeseries, sc_matrix):
    sc_list = []
    for timeseries in minibatch_timeseries:
        sc_list.append(sc_matrix)
    return torch.stack(sc_list)


def process_dynamic_sc(minibatch_timeseries, sc_matrix, window_size, window_stride, dynamic_length=None, sampling_init=None):
    # assumes input shape [minibatch x time x node]
    # output shape [minibatch x time x node x node]
    if dynamic_length is None:
        dynamic_length = minibatch_timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert minibatch_timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert minibatch_timeseries.ndim==3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(minibatch_timeseries.shape[1]-dynamic_length+1)
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride))

    minibatch_sc_list = [get_minibatch_sc(minibatch_timeseries, sc_matrix) for sampling_point in sampling_points]
    dynamic_sc = torch.stack(minibatch_sc_list, dim=1)

    return dynamic_sc, sampling_points

