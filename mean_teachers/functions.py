import numpy as np


def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return (np.exp(-0.5 * phase * phase))


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)

    for ema_param, model_param in zip(ema_model.params(), model.params()):
        ema_param.array = alpha * ema_param.array + (1 - alpha) * model_param.array
