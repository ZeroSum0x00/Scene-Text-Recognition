import numpy as np
import tensorflow as tf
from data_utils.data_processing import get_labels


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, batch_max_length=-1, blank_index=-1):
        character             = get_labels(character)
        dict_character        = list(character)
        self.batch_max_length = batch_max_length
        self.blank_index      = blank_index
        
        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i

        # dummy '[CTCblank]' token for CTCLoss
        if self.blank_index == -1:
            self.character = dict_character + ['[CTCblank]']
        else:
            self.character = ['[CTCblank]'] + dict_character
            
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
        max_length = self.batch_max_length if self.batch_max_length != -1 else max(length)
        # The index used for padding (=0) would not affect the CTC loss calculation.
        if self.blank_index == 0:
            fill_value = -1
        else:
            fill_value = self.N - 1
            
        batch_text = np.full(shape=(len(text), max_length), fill_value=fill_value, dtype=np.int32)

        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = np.array(text)
        return np.array(batch_text, dtype=np.int32), np.array(length, dtype=np.int32)
        
    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        if self.blank_index == 0:
            dummy_token_index = -1
        else:
            dummy_token_index = len(self.character) - 1

        for index, l in enumerate(length):
            t = text_index[index, :]
            char_list = []
            for i in range(l):
                if t[i] != dummy_token_index and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    if self.blank_index == 0:
                        char_list.append(self.character[int(t[i] + 1)])
                    else:
                        char_list.append(self.character[int(t[i])])
            text = ''.join(char_list)
            texts.append(text)
        return texts


class OnehotLabelConverter(object):

    def __init__(self, character, batch_max_length=25, blank_index=0, blank_character=u'\u2764'):
        character             = get_labels(character)
        dict_character        = list(character)
        self.batch_max_length = batch_max_length + 1
        self.blank_index      = blank_index
        self.blank_character  = blank_character
        
        self.dict = {}
        if self.blank_index == -1:
            self.character = dict_character + [blank_character]
        else:
            self.character = [blank_character] + dict_character
            
        for i, char in enumerate(self.character):
            self.dict[char] = i

        self.N = len(self.character)

    def encode(self, text):
        length = [len(s) for s in text]
        batch_text = []
        for i, t in enumerate(text):
            text = list(t + self.blank_character * (self.batch_max_length - len(t)))
            text = [self.dict[char] for char in text]
            onehot = tf.one_hot(np.array(text), depth=len(self.character))
            batch_text.append(onehot)
        return np.array(batch_text, dtype=np.int32), np.array(length, dtype=np.int32)
        
    def decode(self, text_index, length):
        texts = []
        if self.blank_index == 0:
            dummy_token_index = 0
        else:
            dummy_token_index = len(self.character) - 1

        for index in text_index:
            t = np.argmax(index, axis=-1)
            char_list = []
            for i in range(len(t)):
                if t[i] != dummy_token_index and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[int(t[i])])
            text = ''.join(char_list)
            texts.append(text)
        return texts


class SparseOnehotLabelConverter(object):

    def __init__(self, character, batch_max_length=25, blank_character=u'\u2764', blank_index=0):
        character             = get_labels(character)
        dict_character        = list(character)
        self.batch_max_length = batch_max_length + 1
        self.blank_character  = blank_character
        self.blank_index      = blank_index
        
        self.dict = {}
        if self.blank_index == -1:
            self.character = dict_character + [blank_character]
        else:
            self.character = [blank_character] + dict_character
            
        for i, char in enumerate(self.character):
            self.dict[char] = i

        self.N = len(self.character)

    def encode(self, text):
        length = [len(s) for s in text]
        batch_text = []
        for i, t in enumerate(text):
            text = list(t + self.blank_character * (self.batch_max_length - len(t)))
            text = [self.dict[char] for char in text]
            batch_text.append(text)
        return np.array(batch_text, dtype=np.int32), np.array(length, dtype=np.int32)
        
    def decode(self, text_index, length):
        texts = []
        if self.blank_index == 0:
            dummy_token_index = 0
        else:
            dummy_token_index = len(self.character) - 1

        for t in text_index:
            char_list = []
            for i in range(len(t)):
                if t[i] != dummy_token_index and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[int(t[i])])
            
            text = ''.join(char_list)
            texts.append(text)
        return texts


# class BaiduWarpCTCLabelConverter(object):
#     """ Convert between text-label and text-index """

#     def __init__(self, character, batch_max_length=-1):
#         # character (str): set of the possible characters.
#         dict_character = list(character)

#         self.dict = {}
#         for i, char in enumerate(dict_character):
#             # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
#             self.dict[char] = i + 1

#         self.character = dict_character + ['[CTCblank]']  # dummy '[CTCblank]' token for CTCLoss
#         self.N = len(self.character)

#     def encode(self, text):
#         """convert text-label into text-index.
#         input:
#             text: text labels of each image. [batch_size]
#         output:
#             text: concatenated text index for CTCLoss.
#                     [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
#             length: length of each text. [batch_size]
#         """
#         length = [len(s) for s in text]
#         text = ''.join(text)
#         batch_text = [self.dict[char] for char in text]
#         return np.array(batch_text), np.array(length)
        
#     def decode(self, text_index, length):
#         """ convert text-index into text-label. """
#         texts = []
#         index = 0
#         dummy_token_index = len(self.character) - 1
#         for l in length:
#             t = text_index[index:index + l]
#             char_list = []
#             for i in range(l):
#                 if t[i] != dummy_token_index and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
#                     char_list.append(self.character[int(t[i] - 1)])
#             text = ''.join(char_list)
#             texts.append(text)
#             index += l
#         return texts

        
class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, batch_max_length=-1):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']
        list_character = list(character)
        self.character = list_token + list_character
        self.N = len(self.character)
        self.batch_max_length = batch_max_length + 1 if batch_max_length != -1 else -1
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]
        max_length = self.batch_max_length if self.batch_max_length != -1 else max(length)
        batch_text = np.full(shape=(len(text), max_length + 1), fill_value=1, dtype=np.int32)

        for i, t in enumerate(text):
            text = list(t)
            text.insert(0, '[GO]')            
            text = [self.dict[char] for char in text]
            batch_text[i][0:len(text)] = np.array(text)
        return np.array(batch_text, dtype=np.int32), np.array(length, dtype=np.int32)

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')]
            text = text.replace('[GO]', '')
            texts.append(text)
        return texts


# class TokenLabelConverter(object):
#     """ Convert between text-label and text-index """

#     def __init__(self, character, batch_max_length=-1):
#         # character (str): set of the possible characters.
#         # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
#         self.SPACE = '[s]'
#         self.GO = '[GO]'
#         self.list_token = [self.GO, self.SPACE]
#         self.character = self.list_token + list(character)
#         self.dict = {word: i for i, word in enumerate(self.character)}
#         self.batch_max_length = batch_max_length + len(self.list_token) if batch_max_length != -1 else -1

#     def encode(self, text):
#         length = [len(s) + len(self.list_token) for s in text]  # +2 for [GO] and [s] at end of sentence.
#         max_length = self.batch_max_length if self.batch_max_length != -1 else max(length)
#         batch_text = np.full(shape=(len(text), self.batch_max_length), fill_value=self.dict[self.GO], dtype=np.int32)
#         # batch_text = np.full(shape=(len(text), max(length)), fill_value=self.dict[self.GO], dtype=np.int32)
        
#         for i, t in enumerate(text):
#             txt = [self.GO] + list(t) + [self.SPACE]
#             txt = [self.dict[char] for char in txt]
#             batch_text[i][:len(txt)] = np.array(txt)  # batch_text[:, 0] = [GO] token
#         return np.array(batch_text), np.array(length)

#     def decode(self, text_index, length):
#         """ convert text-index into text-label. """
#         texts = []
#         for index, l in enumerate(length):
#             text = ''.join([self.character[i] for i in text_index[index, :]])
#             texts.append(text)
#         return texts