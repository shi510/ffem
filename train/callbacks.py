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

    def __init__(self, dataset_dict, top_k, metric, log_dir='logs'):
        super(RecallCallback, self).__init__()
        self.ds_dict = dataset_dict
        self.top_k = top_k
        self.metric = metric
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        recall_avgs = {}
        # Init recall average dictionary
        for k in self.top_k:
            recall_avgs['recall@{}'.format(k)] = 0.
        # Evaluate recall over multiple datasets
        for ds_name in self.ds_dict:
            ds = self.ds_dict[ds_name]
            recall_top_k = recall.evaluate(self.model, ds, self.metric, self.top_k, 256)
            with self.writer.as_default():
                for k, value in zip(self.top_k, recall_top_k):
                    recall_str = 'recall@{}'.format(k)
                    scalar_name = ds_name + '_{}'.format(recall_str)
                    value *= 100
                    tf.summary.scalar(scalar_name, value, step=epoch)
                    logs[scalar_name] = tf.identity(value)
                    recall_avgs[recall_str] += value
                self.writer.flush()
        with self.writer.as_default():
            ds_size = len(self.ds_dict)
            for key in recall_avgs:
                recall_avgs[key] /= ds_size
                logs[key] = recall_avgs[key]
            self.writer.flush()
