import numpy as np

def normalize(data, new_min=0, new_max=1):
    data_min = np.min(data)
    data_max = np.max(data)

    # Avoid division by zero
    if data_max - data_min == 0:
        normalized_data = np.zeros_like(data) + new_min
    else:
        normalized_data = (new_max - new_min) * (data - data_min) / (data_max - data_min) + new_min

    return normalized_data, data_min, data_max


def unnormalize(normalized_data, data_min, data_max, new_min=0, new_max=1):
    if new_max - new_min == 0:
        raise ValueError("new_max and new_min cannot be the same")

    original_data = (normalized_data - new_min) * (data_max - data_min) / (new_max - new_min) + data_min
    return original_data
