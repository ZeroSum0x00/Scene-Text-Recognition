import os
import cv2
import math
import numpy as np

        
def resize_with_pad(image, target_size, interpolation=None):
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    ratio = w / float(h)
    if math.ceil(target_size[0] * ratio) > target_size[1]:               
        resized_w = target_size[1]                                     
    else:
        resized_w = math.ceil(target_size[0] * ratio)

    image = cv2.resize(image, (resized_w, target_size[0]), interpolation=interpolation)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    new_h, new_w = target_size[0], resized_w
    pad_image = np.zeros(target_size)    
    pad_image[:, :new_w, :] = image

    if target_size[1] != new_w:  # add border Pad
        pad_image[:, new_w:, :] = np.expand_dims(image[:, new_w-1, :], axis=1)
    return pad_image

def image_preprocessing(image, normalize='sub_divide', target_size=None, interpolation=None):    
    if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
        # image = resize_with_pad(image, target_size, interpolation)
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if normalize == "sub_divide":
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = np.clip(image, -1, 1)
    elif normalize == "divide":
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.clip(image, 0, 1)
    else:
        image = image.astype(np.uint8)
        image = np.clip(image, 0, 255)
    return image


def edit_distance(prediction_tokens, reference_tokens):
    """ Standard dynamic programming algorithm to compute the Levenshtein Edit Distance Algorithm

    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
    Returns:
        Edit distance between the predicted sentence and the reference sentence
    """
    # Initialize a matrix to store the edit distances
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]

    # Fill the first row and column with the number of insertions needed
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j

    # Iterate through the prediction and reference tokens
    for i, p_tok in enumerate(prediction_tokens):
        for j, r_tok in enumerate(reference_tokens):
            # If the tokens are the same, the edit distance is the same as the previous entry
            if p_tok == r_tok:
                dp[i+1][j+1] = dp[i][j]
            # If the tokens are different, the edit distance is the minimum of the previous entries plus 1
            else:
                dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1

    # Return the final entry in the matrix as the edit distance     
    return dp[-1][-1]


def get_cer(preds, target):
    """ Update the cer score with the current set of references and predictions.

    Args:
        preds (typing.Union[str, typing.List[str]]): list of predicted sentences
        target (typing.Union[str, typing.List[str]]): list of target words

    Returns:
        Character error rate score
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]

    total, errors = 0, 0
    for pred_tokens, tgt_tokens in zip(preds, target):
        errors += edit_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)

    if total == 0:
        return 0.0

    cer = errors / total

    return cer


def get_wer(preds, target):
    """ Update the wer score with the current set of references and predictions.

    Args:
        target (typing.Union[str, typing.List[str]]): string of target sentence or list of target words
        preds (typing.Union[str, typing.List[str]]): string of predicted sentence or list of predicted words

    Returns:
        Word error rate score
    """
    if isinstance(preds, str) and isinstance(target, str):
        preds = [preds]
        target = [target]

    if isinstance(preds, list) and isinstance(target, list):
        errors, total_words = 0, 0
        for _pred, _target in zip(preds, target):
            if isinstance(_pred, str) and isinstance(_target, str):
                errors += edit_distance(_pred.split(), _target.split())
                total_words += len(_target.split())
            else:
                print("Error: preds and target must be either both strings or both lists of strings.")
                return np.inf
            
    else:
        print("Error: preds and target must be either both strings or both lists of strings.")
        return np.inf
    
    wer = errors / total_words
            
    return wer