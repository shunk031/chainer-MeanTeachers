import numpy as np


def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0, 0, rampup_length)
        phase = 1.0 - current / rampup_length
        return (np.exp(-0.5 * phase * phase))
