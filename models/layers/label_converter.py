import numpy as np
import tensorflow as tf


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i

        self.character = dict_character + ['[CTCblank]']  # dummy '[CTCblank]' token for CTCLoss
        self.N = len(self.character)
        
    def encode(self, text, max_string_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            max_string_length: max length of text label in the batch. 25 by default
        output:
            text: text index for CTCLoss. [batch_size, max_string_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = np.full(shape=(len(text), max_string_length), fill_value=-1)

        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = np.array(text)
        return tf.Variable(batch_text), tf.Variable(length)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        dummy_token_index = len(self.character) - 1
        for index, l in enumerate(length):
            t = text_index[index, :]
            char_list = []
            for i in range(l):
                if t[i] != dummy_token_index and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[int(t[i])])
            text = ''.join(char_list)
            texts.append(text)
        return ''.join(texts)
