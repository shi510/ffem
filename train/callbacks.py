import evaluate.recall as recall

import tensorflow as tf


class LogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir='./logs'):
        super(LogCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=epoch)
            self.writer.flush()


class RecallCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_ds, top_k, metric_fn, log_dir=None):
        super(RecallCallback, self).__init__()
        self.ds = test_ds
        self.top_k = top_k
        self.metric_fn = metric_fn
        if log_dir is not None:
            self.log_dir = log_dir
            self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        recall_top_k = recall.evaluate(self.model, self.ds, self.metric_fn, self.top_k, 256)
        if hasattr(self, 'log_dir'):
            with self.writer.as_default():
                for name, value in zip(self.top_k, recall_top_k):
                    name = 'recall@' + str(name)
                    value *= 100
                    tf.summary.scalar(str(name), value, step=epoch)
                    logs[str(name)] = tf.identity(value)
                self.writer.flush()
