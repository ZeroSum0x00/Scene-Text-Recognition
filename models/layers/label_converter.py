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
        
    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: text index for CTCLoss. [batch_size, max_string_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = np.full(shape=(len(text), max(length)), fill_value=-1, dtype=np.int32)

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


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, batch_max_length=25):
        self.batch_max_length = batch_max_length
        list_token = ['[GO]', '[s]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
        self.N = len(self.character)
        
    def encode(self, text):
        length = [len(s) + 1 for s in text]

        self.batch_max_length += 1
        batch_text = np.full(shape=(len(text), self.batch_max_length + 1), fill_value=0, dtype=np.int32)

        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]

            batch_text[i][1: len(text)+1] = np.array(text)
        return tf.Variable(batch_text), tf.Variable(length)

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
