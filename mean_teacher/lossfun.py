import chainer.functions as F


def kl_divergence(y, t):
    entropy = - F.sum(t[t.nonzero()] * F.log(t[t.nonzero()]))
    cross_entropy = - F.sum(t * F.log_softmax(y))

    return (cross_entropy - entropy) / y.shape[0]


def symmetric_mse_loss(input1, input2):
    assert input1.shape == input2.shape

    num_classes = input1.shape[1]
    return F.sum((input1 - input2) ** 2) / num_classes


def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.shape == target_logits.shape

    input_softmax = F.softmax(input_logits)
    target_softmax = F.softmax(target_logits)
    num_classes = input_logits.shape[1]

    return F.mean_squared_error(input_softmax, target_softmax) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.shape == target_logits.shape

    input_log_softmax = F.log_softmax(input_logits)
    target_softmax = F.softmax(target_logits)
    return kl_divergence(input_log_softmax, target_softmax)
