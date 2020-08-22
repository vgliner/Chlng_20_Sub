import abc
import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from typing import Callable, Any, MutableSequence
from pathlib import Path
from train_multimodel_results import BatchResult, EpochResult, FitResult
from torch.nn import Module, Sequential, DataParallel


class MultiModelTrainer(abc.ABC):
    """
    A class abstracting the various tasks of training models comprised of several sub-models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, models: MutableSequence[Module],
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizers: MutableSequence[torch.optim.Optimizer], device='cpu',
                 parallel: bool = False):
        """
        Initialize the trainer.
        :param models: List of the models to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizers: List of corresponding optimizers to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.models = models
        self.loss_fn = loss_fn
        self.optimizers = optimizers
        self.device = device
        for model in self.models:
            model.to(self.device)
        self.train_loss, self.train_acc, self.test_loss, self.test_acc = [], [], [], []
        self.train_challenge_score, self.test_challenge_score = [], []
        self.parallel = parallel
        if parallel:
            print("Using", torch.cuda.device_count(), "GPUs")
            self.model = DataParallel(Sequential(*self.models))

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None,
            challenge_score_as_metric=False, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param challenge_score_as_metric: Whether to use the challenge
        score as metric for early stopping and checkpoint saving
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

        best_acc = None
        best_score = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                best_score = saved_state.get('best_score', best_score)
                epochs_without_improvement =\
                    saved_state.get('ewi', epochs_without_improvement)
                for i, model in enumerate(self.models):
                    attribute_str = 'model_' + str(i) + '_state'
                    model.load_state_dict(saved_state[attribute_str])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            #  Train & evaluate for one epoch including:
            #  - saving of losses and accuracies in the lists above.
            #  - early stopping
            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            (loss, acc, score) = train_result
            self.train_loss += loss
            self.train_acc.append(acc)
            self.train_challenge_score.append(score)

            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            (loss, acc, score) = test_result
            self.test_loss += loss

            actual_num_epochs += 1

            if checkpoints and not challenge_score_as_metric:
                if not best_acc:
                    best_acc = acc
                if acc > best_acc:
                    best_acc = acc
                    save_checkpoint = True

            if self.test_acc and not challenge_score_as_metric:
                if acc <= self.test_acc[-1]:
                    epochs_without_improvement += 1
                else:
                    epochs_without_improvement = 0

            self.test_acc.append(acc)

            if checkpoints and challenge_score_as_metric:
                if not best_score:
                    best_score = score
                if score > best_score:
                    best_score = score
                    save_checkpoint = True

            if self.test_challenge_score and challenge_score_as_metric:
                if score <= self.test_challenge_score[-1]:
                    epochs_without_improvement += 1
                else:
                    epochs_without_improvement = 0

            self.test_challenge_score.append(score)

            with open(os.getcwd() + "/Log_of_intermediate_results.txt", "a") as f:
                f.write(f'{self.train_loss[-1]}\t{self.test_loss[-1]}\t{self.train_acc[-1]}\t{self.test_acc[-1]}\t{best_acc}\t{save_checkpoint}\n')

            if early_stopping:
                if epochs_without_improvement >= early_stopping:
                    break

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc,
                                   best_score=best_score,
                                   ewi=epochs_without_improvement)
                for i, model in enumerate(self.models):
                    attribute_str = 'model_' + str(i) + '_state'
                    saved_state[attribute_str] = model.state_dict()
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch+1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs,
                         self.train_loss, self.train_acc, self.test_loss, self.test_acc,
                         self.train_challenge_score, self.test_challenge_score)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        for model in self.models:
            model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        for model in self.models:
            model.train(False)  # set evaluation (test) mode
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
        num_correct = 0
        cum_score = 0
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

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct
                cum_score += batch_res.sum_of_challenge_score

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            mean_challenge_score = cum_score / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f}, '
                                 f'Score {mean_challenge_score:3f})')

        return EpochResult(losses=losses, accuracy=accuracy, mean_challenge_score=mean_challenge_score)


class PreTrainer(MultiModelTrainer, abc.ABC):
    def save_model(self, f: str = None, model_num: int = 0):
        saved_state = dict(state_dict=self.models[model_num].state_dict())
        Path(os.path.dirname(f)).mkdir(exist_ok=True)
        torch.save(saved_state, f)
        print(f'Model saved in path: {f}')


class FineTuningTrainer(MultiModelTrainer, abc.ABC):
    def load_model(self, f: str, model_num: int = 0):
        assert os.path.isfile(f), f'File {f} not found'
        saved_state = torch.load(f, map_location=self.device)
        self.models[model_num].load_state_dict(saved_state['state_dict'])
        print(f'Loaded model from path: {f}')


class ConvTrainer(MultiModelTrainer):
    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        batch_size = x.shape[0]
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        if self.parallel:
            out = self.model(x)
        else:
            out = self.models[0](x).reshape((batch_size, -1))
            out = self.models[1](out)

        loss = self.loss_fn(out, y)
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()

        with torch.no_grad():
            num_classes = y.shape[1]
            num_correct = torch.sum(torch.sum((y == (out > 0)).to(dtype=torch.int64), dim=1) == num_classes)
            sum_of_challenge_score = sum(challenge_score(y, out))

        return BatchResult(loss.item(), num_correct.item(), sum_of_challenge_score)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        batch_size = x.shape[0]
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            num_classes = y.shape[1]
            if self.parallel:
                out = self.model(x)
            else:
                out = self.models[0](x).reshape((batch_size, -1))
                out = self.models[1](out)
            loss = self.loss_fn(out, y)
            num_correct = torch.sum(torch.sum((y == (out > 0)).to(dtype=torch.int64), dim=1) == num_classes)
            sum_of_challenge_score = sum(challenge_score(y, out))
        return BatchResult(loss.item(), num_correct.item(), sum_of_challenge_score)


class ConvPreTrainer(PreTrainer, ConvTrainer):
    pass


class ConvFineTuningTrainer(FineTuningTrainer, ConvTrainer):
    pass


def challenge_score(y, y_pred):
    # placeholder implementation
    assert y.shape == y_pred.shape, "size mismatch"
    return torch.zeros_like(y[:, 0])
