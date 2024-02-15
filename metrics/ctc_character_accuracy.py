import tensorflow as tf
from tensorflow.keras.metrics import Metric
from data_utils.data_processing import get_labels


class CTCCharacterAccuracy(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Character Accuracy.
    
    Args:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, vocabulary, name="char_accuracy", **kwargs):
        # Initialize the base Metric class
        super(CTCCharacterAccuracy, self).__init__(name=name, **kwargs)
        self.save_type = "increase"

        # Initialize variables to keep track of the cumulative character/word error rates and counter
        self.cer_accumulator = tf.Variable(0.0, name="cer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)
        
        # Store the vocabulary as an attribute
        self.vocabulary = tf.constant(list(get_labels(vocabulary)))

    @staticmethod
    def get_cer(pred_decoded, y_true, vocab, padding=-1):
        """ Calculates the character error rate (CER) between the predicted labels and true labels for a batch of input data.

        Args:
            pred_decoded (tf.Tensor): The predicted labels, with dtype=tf.int32, usually output from tf.keras.backend.ctc_decode
            y_true (tf.Tensor): The true labels, with dtype=tf.int32
            vocab (tf.Tensor): The vocabulary tensor, with dtype=tf.string
            padding (int, optional): The padding token when converting to sparse tensor. Defaults to -1.

        Returns:
            tf.Tensor: The CER between the predicted labels and true labels
        """
        # Keep only valid indices in the predicted labels tensor, replacing invalid indices with padding token
        vocab_length = tf.cast(tf.shape(vocab)[0], tf.int64)
        valid_pred_indices = tf.less(pred_decoded, vocab_length)
        valid_pred = tf.where(valid_pred_indices, pred_decoded, padding)

        # Keep only valid indices in the true labels tensor, replacing invalid indices with padding token
        y_true = tf.cast(y_true, tf.int64)
        valid_true_indices = tf.less(y_true, vocab_length)
        valid_true = tf.where(valid_true_indices, y_true, padding)

        # Convert the valid predicted labels tensor to a sparse tensor
        sparse_pred = tf.RaggedTensor.from_tensor(valid_pred, padding=padding).to_sparse()

        # Convert the valid true labels tensor to a sparse tensor
        sparse_true = tf.RaggedTensor.from_tensor(valid_true, padding=padding).to_sparse()

        # Calculate the normalized edit distance between the sparse predicted labels tensor and sparse true labels tensor
        distance = tf.edit_distance(sparse_pred, sparse_true, normalize=True)

        return distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the state variables of the metric.

        Args:
            y_true: A tensor of true labels with shape (batch_size, sequence_length).
            y_pred: A tensor of predicted labels with shape (batch_size, sequence_length, num_classes).
            sample_weight: (Optional) a tensor of weights with shape (batch_size, sequence_length).
        """
        # Get the input shape and length
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(input_shape[1], "int32")

        # Decode the predicted labels using greedy decoding
        decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = self.get_cer(decode_predicted[0], y_true, self.vocabulary)

        # Add the sum of the distance tensor to the cer_accumulator variable
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        
        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(len(y_true))

    def result(self):
        """ Computes and returns the metric result.

        Returns:
            A TensorFlow float representing the Character Accuracy.
        """
        cer = tf.math.divide_no_nan(self.cer_accumulator, tf.cast(self.batch_counter, tf.float32))
        return 1 - tf.clip_by_value(cer, clip_value_min=0, clip_value_max=1)


# class CWERMetric(tf.keras.metrics.Metric):
#     """A custom TensorFlow metric to compute the Character Error Rate (CER).
    
#     Args:
#         padding_token: An integer representing the padding token in the input data.
#         name: (Optional) string name of the metric instance.
#         **kwargs: Additional keyword arguments.
#     """
#     def __init__(self, padding_token=0, name="CWER", **kwargs):
#         # Initialize the base Metric class
#         super(CWERMetric, self).__init__(name=name, **kwargs)
#         self.save_type = "decrease"
        
#         # Initialize variables to keep track of the cumulative character/word error rates and counter
#         self.cer_accumulator = tf.Variable(0.0, name="cer_accumulator", dtype=tf.float32)
#         self.wer_accumulator = tf.Variable(0.0, name="wer_accumulator", dtype=tf.float32)
#         self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)
        
#         # Store the padding token as an attribute
#         self.padding_token = padding_token

#     def update_state(self, y_true, y_pred, y_true_length=None, sample_weight=None):
#         """Updates the state variables of the metric.

#         Args:
#             y_true: A tensor of true labels with shape (batch_size, sequence_length).
#             y_pred: A tensor of predicted labels with shape (batch_size, sequence_length, num_classes).
#             sample_weight: (Optional) a tensor of weights with shape (batch_size, sequence_length).
#         """
#         # Get the input shape and length
#         input_shape = tf.keras.backend.shape(y_pred)
#         input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(input_shape[1], "int32")

#         y_true_length = tf.cast(y_true_length, dtype="int32")
        
#         # Decode the predicted labels using greedy decoding
#         decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

#         # Convert the dense decode tensor to a sparse tensor
#         predicted_labels_sparse = tf.keras.backend.ctc_label_dense_to_sparse(decode_predicted[0], input_length)
        
#         # Convert the dense true labels tensor to a sparse tensor and cast to int64
#         true_labels_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, y_true_length), "int64")

#         # Retain only the non-padding elements in the predicted labels tensor
#         predicted_labels_sparse = tf.sparse.retain(predicted_labels_sparse, tf.not_equal(predicted_labels_sparse.values, -1))
        
#         # Retain only the non-padding elements in the true labels tensor
#         true_labels_sparse = tf.sparse.retain(true_labels_sparse, tf.not_equal(true_labels_sparse.values, self.padding_token))

#         # Calculate the normalized edit distance between the predicted labels and true labels tensors
#         distance = tf.edit_distance(predicted_labels_sparse, true_labels_sparse, normalize=True)

#         # Add the sum of the distance tensor to the cer_accumulator variable
#         self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        
#         # Increment the batch_counter by the batch size
#         self.batch_counter.assign_add(len(y_true))

#         # Calculate the number of wrong words in batch and add to wer_accumulator variable
#         self.wer_accumulator.assign_add(tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32)))

#     def result(self):
#         """Computes and returns the metric result.

#         Returns:
#             A dictionary containing the harmonic mean of CER and WER.
#         """
#         CER = tf.math.divide_no_nan(self.cer_accumulator, tf.cast(self.batch_counter, tf.float32))
#         WER = tf.math.divide_no_nan(self.wer_accumulator, tf.cast(self.batch_counter, tf.float32))
#         fCWER = (2 * CER * WER) / (CER + WER)
#         return {
#                 "CER": CER,
#                 "WER": WER,
#                 "fCWER": fCWER
#         }