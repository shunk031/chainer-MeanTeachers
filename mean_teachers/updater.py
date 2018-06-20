import chainer
import chainer.functions as F
from chainer import training
from chainer.dataset import convert

from mean_teachers.lossfun import (
    symmetric_mse_loss,
    softmax_mse_loss,
)
from mean_teachers.functions import (
    get_current_consistency_weight,
    update_ema_variables,
)


class MeanTeacherUpdater(training.StandardUpdater):

    def __init__(self,
                 labeled_iter,
                 ema_iter,
                 optimizer,
                 ema_model,
                 ema_decay=0.999,
                 distance_const=-1,
                 consistency=None,
                 consistency_lossfun=softmax_mse_loss,
                 converter=convert.concat_examples,
                 device=None,
                 loss_func=F.softmax_cross_entropy):
        super(MeanTeacherUpdater, self).__init__(
            labeled_iter, optimizer, converter, device, loss_func)

        self._iterators['ema'] = ema_iter
        self.ema_model = ema_model
        self.ema_decay = ema_decay
        self.distance_const = distance_const
        self.consistency = consistency
        self.consistency_lossfun = consistency_lossfun

    def update_core(self):
        labeled_batch = self._iterators['main'].next()
        labeled_var, target_var = self.converter(labeled_batch, self.device)

        batch_size = self._iterators['main'].batch_size

        ema_batch = self._iterators['ema'].next()
        ema_var = self.converter(ema_batch, self.device)

        optimizer = self._optimizers['main']
        model = optimizer.target

        model_out = model(labeled_var)
        with chainer.no_backprop_mode():
            ema_model_out = self.ema_model(ema_var)

        logit1, logit2 = model_out
        ema_logit, _ = ema_model_out

        if self.distance_const >= 0:
            class_logit, cons_logit = logit1, logit2
            res_loss = self.distance_const * symmetric_mse_loss(class_logit, cons_logit) / batch_size
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0

        class_loss = self.loss_func(class_logit, target_var) / batch_size
        ema_class_loss = self.loss_func(ema_logit, target_var) / batch_size

        if self.consistency:
            consistency_weight = get_current_consistency_weight(self.epoch)
            consistency_loss = consistency_weight * self.consistency_lossfun(cons_logit, ema_logit) / batch_size
        else:
            consistency_loss = 0

        loss = class_loss + consistency_loss + res_loss
        model.cleargrads()
        loss.backward()
        optimizer.update()

        update_ema_variables(model, self.ema_model, self.ema_decay, self.epoch)

        chainer.report({
            'accuracy': F.accuracy(class_logit, target_var),
            'loss': loss,
            'class_loss': class_loss,
            'ema_loss': ema_class_loss,
            'cons_loss': consistency_loss,
        }, model)
