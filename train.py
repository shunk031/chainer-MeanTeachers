import argparse

import chainer
from chainer import training
from chainer.training import extensions

from models import archs
from mean_teachers.updater import MeanTeacherUpdater
from mean_teachers.lossfun import (
    softmax_mse_loss,
    softmax_kl_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='resnet50',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--distance_cost', type=float, default=-1,
                        help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=None, type=float,
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency_type', default="mse", type=str,
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency_rampup', default=30, type=int,
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)

    return parser.parse_args()


def main():

    args = parse_args()

    model = archs[args.arch]()
    ema_model = archs[args.arch]()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()
        ema_model.to_gpu()

    train, val = chainer.datasets.get_cifar10()
    _, test = chainer.datasets.get_cifar10(withlabel=False)

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.batchsize, repeat=False, n_processes=args.loaderjob)
    ema_iter = chainer.iterators.MultiprocessIterator(
        test, args.batchsize, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    if args.consistency_type == 'mse':
        consistency_lossfun = softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_lossfun = softmax_kl_loss
    updater = MeanTeacherUpdater(train_iter, ema_iter, optimizer, ema_model,
                                 ema_decay=args.ema_decay,
                                 distance_cost=args.distance_cost,
                                 consistency=args.consistency,
                                 consistency_lossfun=consistency_lossfun, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (1 if args.test else 100000), 'iteration'
    log_interval = (1 if args.test else 1000), 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
