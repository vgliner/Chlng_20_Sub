import abc
import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path
from torch.utils.data import Subset
import numpy as np
from train_results import BatchResult, EpochResult, FitResult

EPS = torch.finfo(torch.float32).eps


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device='cuda', classification_threshold=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.classification_threshold = classification_threshold
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            full_path = os.path.realpath(__file__)
            path, filename = os.path.split(full_path)

            if os.path.isfile(path + '//' + checkpoint_filename):
                checkpoint_filename = path + '//' + checkpoint_filename
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                epochs_without_improvement = \
                    saved_state.get('ewi', epochs_without_improvement)
                self.model.load_state_dict(saved_state['model_state'])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)

            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            (loss, acc, TP, TN, FP, FN, out, y) = train_result
            train_loss += loss
            train_acc.append(acc)
            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            (loss, acc, TP, TN, FP, FN, out, y) = test_result
            test_loss += loss

            if checkpoints:
                if not best_acc:
                    best_acc = acc
                if acc > best_acc:
                    best_acc = acc
                    save_checkpoint = True

            if test_acc:
                if acc <= test_acc[-1]:
                    epochs_without_improvement += 1
                else:
                    epochs_without_improvement = 0

            test_acc.append(acc)

            if early_stopping:
                if epochs_without_improvement >= early_stopping:
                    break

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.state_dict())
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch + 1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        y = []
        out = []
        num_correct = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)
                y.append(batch_res.y)
                out.append(batch_res.out)
                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct
                tp += batch_res.num_TP
                tn += batch_res.num_TN
                fp += batch_res.num_FP
                fn += batch_res.num_FN

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy, num_TP=tp, num_TN=tn, num_FP=fp, num_FN=fn, y=y, out=out)


class FfTrainer(Trainer):

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        num_correct = torch.sum((out > 0) == (y == 1)) / 9
        tp = torch.sum((out > 0) * (y == 1))
        tn = torch.sum((out <= 0) * (y == 0))
        fp = torch.sum((out > 0) * (y == 0))
        fn = torch.sum((out <= 0) * (y == 1))

        return BatchResult(loss.item(), num_correct.item(), tp, tn, fp, fn, out, y)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out, y)
            num_correct = torch.sum((out > 0) == (y == 1)) / 9
            out_norm = torch.sigmoid(out)
            if self.classification_threshold is None:
                tp = torch.sum((out > 0) * (y == 1))
                tn = torch.sum((out <= 0) * (y == 0))
                fp = torch.sum((out > 0) * (y == 0))
                fn = torch.sum((out <= 0) * (y == 1))
            else:
                tp = torch.sum((out_norm >= self.classification_threshold) * (y == 1))
                tn = torch.sum((out_norm < self.classification_threshold) * (y == 0))
                fp = torch.sum((out_norm >= self.classification_threshold) * (y == 0))
                fn = torch.sum((out_norm < self.classification_threshold) * (y == 1))
                num_correct = torch.sum((out_norm > self.classification_threshold) == (y == 1))

        return BatchResult(loss.item(), num_correct.item(), tp, tn, fp, fn, out, y)


class Ecg12LeadNetTrainerBinary(Trainer):

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x).flatten()
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        num_correct = torch.sum((out > 0) == (y == 1))
        tp = torch.sum((out > 0) * (y == 1))
        tn = torch.sum((out <= 0) * (y == 0))
        fp = torch.sum((out > 0) * (y == 0))
        fn = torch.sum((out <= 0) * (y == 1))

        return BatchResult(loss.item(), num_correct.item(), tp, tn, fp, fn, out, y)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x).flatten()
            loss = self.loss_fn(out, y)
            num_correct = torch.sum((out > 0) == (y == 1))
            out_norm = torch.sigmoid(out)
            if self.classification_threshold is None:
                tp = torch.sum((out > 0) * (y == 1))
                tn = torch.sum((out <= 0) * (y == 0))
                fp = torch.sum((out > 0) * (y == 0))
                fn = torch.sum((out <= 0) * (y == 1))
            else:
                tp = torch.sum((out_norm >= self.classification_threshold) * (y == 1))
                tn = torch.sum((out_norm < self.classification_threshold) * (y == 0))
                fp = torch.sum((out_norm >= self.classification_threshold) * (y == 0))
                fn = torch.sum((out_norm < self.classification_threshold) * (y == 1))
                num_correct = torch.sum((out_norm > self.classification_threshold) == (y == 1))

        return BatchResult(loss.item(), num_correct.item(), tp, tn, fp, fn, out, y)


class Ecg12LeadNetTrainerMulticlass(Trainer):

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x)  # .flatten()
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        indices = out > 0  # torch.max(out, 1)  #_,
        indices1 = y > 0  # torch.max(y, 1)  #_,

        num_correct = torch.sum(indices == indices1)

        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out.flatten(), y.flatten())
            indices = out > 0  # torch.max(out, 1) _,
            indices1 = y > 0  # torch.max(y, 1) _,

            num_correct = torch.sum(indices == indices1)
        return BatchResult(loss.item(), num_correct.item())


class SimpleConvNetMulticlassTrainer(Trainer):

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            num_correct, tp, tn, fp, fn, fb, gb, g_mean = calc_stats(y, out > 0)

        return BatchResult(loss.item(), num_correct.item(), tp, tn, fp, fn, fb, gb, g_mean, out, y)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out, y)
            num_correct, tp, tn, fp, fn, fb, gb, g_mean = calc_stats(y, out > 0)

        return BatchResult(loss.item(), num_correct.item(), tp, tn, fp, fn, fb, gb, g_mean, out, y)


class PerClassTrainer(Trainer):
    def __init__(self, training_class, *kargs):
        super().__init__(*kargs)
        self.training_class = training_class

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)
        y = y[:, self.training_class:self.training_class+1].to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            num_correct, tp, tn, fp, fn, fb, gb, g_mean = calc_stats(y, out > 0)

        return BatchResult(loss.item(), num_correct.item(), tp, tn, fp, fn, fb, gb, g_mean, out, y)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)
        y = y[:, self.training_class:self.training_class+1].to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out, y)
            num_correct, tp, tn, fp, fn, fb, gb, g_mean = calc_stats(y, out > 0)

        return BatchResult(loss.item(), num_correct.item(), tp, tn, fp, fn, fb, gb, g_mean, out, y)


def calc_stats(y, y_pred, class_weights=None, beta=2.0):
    num_classes = y.shape[1]

    num_correct = torch.sum(torch.sum((y == y_pred).to(dtype=torch.int64), dim=1) == num_classes)
    tp = torch.sum((y == y_pred) * (y == 1), dim=0, keepdim=True)
    tn = torch.sum((y == y_pred) * (y == 0), dim=0, keepdim=True)
    fp = torch.sum((y != y_pred) * (y == 1), dim=0, keepdim=True)
    fn = torch.sum((y != y_pred) * (y == 0), dim=0, keepdim=True)

    fbl = ((1 + beta ** 2) * tp + EPS) / ((1 + beta ** 2) * tp + fp + beta ** 2 * fn + EPS)
    gbl = (tp + EPS) / (tp + fp + beta * fn + EPS)
    if torch.isnan(fbl).any():
        print('fbl')
        print((1 + beta ** 2) * tp + fp + beta ** 2 * fn)
    if torch.isnan(gbl).any():
        print('gbl')
        print(tp + fp + beta * fn)

    if class_weights is None:
        class_weights = [1] * num_classes

    w = torch.tensor(class_weights).reshape(num_classes, 1).to(y.device, dtype=y.dtype) * 1.0

    fb = torch.matmul(fbl, w) / num_classes
    gb = torch.matmul(gbl, w) / num_classes

    g_mean = (fb * gb) ** 0.5

    return num_correct, tp, tn, fp, fn, fb, gb, g_mean


def train_val_test_split(ds, train=0.6, val=0.2, test=0.2, seed=None):
    assert train + val + test == 1

    if seed:
        np.random.seed(seed)

    probs = np.random.rand(len(ds))
    inds = np.arange(len(ds))

    ds_train = Subset(ds, inds[probs <= train])
    ds_val = Subset(ds, inds[(train < probs) & (probs <= train + val)])
    ds_test = Subset(ds, inds[train + val < probs])

    assert len(ds_train) + len(ds_val) + len(ds_test) == len(ds)
    return ds_train, ds_val, ds_test
