"""Custom callbacks for Keras model training"""

from time import perf_counter

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class LRLogger(Callback):
    """Add the LR to the logs
    Must be before any log writers in the callback list"""

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['lr'] = float(K.get_value(self.model.optimizer.lr))


class EpochTimeLogger(Callback):
    """Adds the epoch time to the logs
    Must be before any log writers in the callback list"""

    def __init__(self):
        super().__init__()
        self.time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.time = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['epoch_time'] = perf_counter() - self.time


class TimeLimitCallback(Callback):
    """Limit the amount of time for training

    WARNING: This callback will not restore the best weights.
    You must track them with an EarlyStopping callback and set
    the best weights manually after a timeout.
    """

    def __init__(self, timeout: float):
        """
        Args:
            timeout: Maximum training time in seconds
        """
        super().__init__()
        self.timeout = timeout
        self.start_time = perf_counter()
        self.timed_out = False

    def on_train_begin(self, logs=None):
        self.start_time = perf_counter()
        self.timed_out = False

    def on_train_batch_end(self, batch, logs=None):
        if perf_counter() - self.start_time > self.timeout:
            self.model.stop_training = True
            self.timed_out = True
