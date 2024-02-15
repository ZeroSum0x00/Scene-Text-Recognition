import tensorflow as tf
from tensorflow.keras.metrics import Metric
from data_utils.data_processing import get_labels


class CTCWordAccuracy(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Word Accuracy.
    
    Attributes:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, vocabulary: str, name="word_accuracy", **kwargs):
        # Initialize the base Metric class
        super(CTCWordAccuracy, self).__init__(name=name, **kwargs)
        self.save_type = "increase"

        # Initialize variables to keep track of the cumulative character/word error rates and counter
        self.wer_accumulator = tf.Variable(0.0, name="wer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)
        
        # Store the vocabulary as an attribute
        self.vocabulary = tf.constant(list(get_labels(vocabulary)))

    @staticmethod
    def preprocess_dense(dense_input: tf.Tensor, vocab: tf.Tensor, padding=-1, separator="") -> tf.SparseTensor:
        """ Preprocess the dense input tensor to a sparse tensor with given vocabulary
        
        Args:
            dense_input (tf.Tensor): The dense input tensor, dtype=tf.int32
            vocab (tf.Tensor): The vocabulary tensor, dtype=tf.string
            padding (int, optional): The padding token when converting to sparse tensor. Defaults to -1.

        Returns:
            tf.SparseTensor: The sparse tensor with given vocabulary
        """
        # Keep only the valid indices of the dense input tensor
        vocab_length = tf.cast(tf.shape(vocab)[0], tf.int64)
        dense_input = tf.cast(dense_input, tf.int64)
        valid_indices = tf.less(dense_input, vocab_length)
        valid_input = tf.where(valid_indices, dense_input, padding)

        # Convert the valid input tensor to a ragged tensor with padding
        input_ragged = tf.RaggedTensor.from_tensor(valid_input, padding=padding)

        # Use the vocabulary tensor to get the strings corresponding to the indices in the ragged tensor
        input_binary_chars = tf.gather(vocab, input_ragged)

        # Join the binary character tensor along the sequence axis to get the input strings
        input_strings = tf.strings.reduce_join(input_binary_chars, axis=1, separator=separator)

        # Convert the input strings tensor to a sparse tensor
        input_sparse_string = tf.strings.split(input_strings, sep=" ").to_sparse()

        return input_sparse_string

    @staticmethod
    def get_wer(pred_decoded, y_true, vocab, padding=-1, separator=""):
        """ Calculate the normalized WER distance between the predicted labels and true labels tensors

        Args:
            pred_decoded (tf.Tensor): The predicted labels tensor, dtype=tf.int32. Usually output from tf.keras.backend.ctc_decode
            y_true (tf.Tensor): The true labels tensor, dtype=tf.int32
            vocab (tf.Tensor): The vocabulary tensor, dtype=tf.string

        Returns:
            tf.Tensor: The normalized WER distance between the predicted labels and true labels tensors
        """
        pred_sparse = CTCWordAccuracy.preprocess_dense(pred_decoded, vocab, padding=padding, separator=separator)
        true_sparse = CTCWordAccuracy.preprocess_dense(y_true, vocab, padding=padding, separator=separator)

        distance = tf.edit_distance(pred_sparse, true_sparse, normalize=True)

        # test with numerical labels not string
        # true_sparse = tf.RaggedTensor.from_tensor(y_true, padding=-1).to_sparse()

        # replace 23 with -1
        # pred_decoded2 = tf.where(tf.equal(pred_decoded, 23), -1, pred_decoded)
        # pred_decoded2_sparse = tf.RaggedTensor.from_tensor(pred_decoded2, padding=-1).to_sparse()

        # distance = tf.edit_distance(pred_decoded2_sparse, true_sparse, normalize=True)

        return distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        """
        # Get the input shape and length
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(input_shape[1], "int32")

        # Decode the predicted labels using greedy decoding
        decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = self.get_wer(decode_predicted[0], y_true, self.vocabulary)

        # Calculate the number of wrong words in batch and add to wer_accumulator variable
        self.wer_accumulator.assign_add(tf.reduce_sum(tf.cast(distance, tf.float32)))

        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(len(y_true))

    def result(self):
        """Computes and returns the metric result.

        Returns:
            A TensorFlow float representing the Word Accuracy.
        """
        wer = tf.math.divide_no_nan(self.wer_accumulator, tf.cast(self.batch_counter, tf.float32))
        return 1 - tf.clip_by_value(wer, clip_value_min=0, clip_value_max=1)

        
# class SequenAccuracy(tf.keras.metrics.Mean):
#     def __init__(self, blank_index=-1, name="SequenAccuracy", dtype=None):
#         super().__init__(name=name, dtype=dtype)
#         self.blank_index = blank_index
#         self.save_type = "increase"
        
#     def update_state(self, y_true, y_pred, y_true_length=None, sample_weight=None):
#         super().update_state(ctc_accuracy(y_true, y_pred, y_true_length, self.blank_index), sample_weight)


# def ctc_accuracy(y_true, y_pred, y_true_length=None, blank_index=-1):
#     y_pred_shape = tf.shape(y_pred)
#     y_true_shape = tf.shape(y_true)
#     y_pred_length = tf.fill((y_pred_shape[0],), y_pred_shape[1])
#     y_pred = tf.keras.backend.ctc_decode(y_pred, y_pred_length)[0][0]
    
#     if blank_index != 0:
#         condition = tf.equal(y_pred, -1)
#         y_pred = tf.where(condition, blank_index, tf.cast(y_pred, tf.int32))

#     y_pred_shape = y_pred_shape[:-1]  # y_pred shape after the decoding loses the last dimension (classes count)

#     # Pad y_true or y_pred so that they have the same max sequence length
#     if y_true_shape[1] < y_pred_shape[1]:
#         y_true = tf.pad(y_true, paddings=[[0, 0], [0, y_pred_shape[1] - y_true_shape[1]]], constant_values=blank_index)
#         y_true_shape = y_pred_shape
#     elif y_pred_shape[1] < y_true_shape[1]:
#         y_pred = tf.pad(y_pred, paddings=[[0, 0], [0, y_true_shape[1] - y_pred_shape[1]]], constant_values=blank_index)
#         y_pred_shape = y_true_shape

#     # ctc_decode() returns y_pred of int64 type and if y_true is int32, that will cause an error in tf.equal(), so
#     # we need to cast y_pred to the same type
#     if y_pred.dtype != y_true.dtype:
#         y_pred = tf.cast(y_pred, y_true.dtype)

#     matching_labels_count = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.int32), axis=1)
#     # A prediction considered correct if all its labels match the corresponding true sequence
#     correct_predictions = tf.equal(matching_labels_count, y_true_shape[1])

#     return tf.cast(correct_predictions, tf.int32)
