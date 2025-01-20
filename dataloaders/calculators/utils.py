from collections.abc import Iterable
import string

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, str) or not isinstance(item, Iterable):
            flattened.append(item)
        else:
            flattened.extend(flatten_list(item))
    return flattened

def align_two_type(a, b):
    if type(a) == type(b):
        return a, b
    if isinstance(a, int) and isinstance(b, str):
        return string.ascii_uppercase[a], b
    elif isinstance(a, str) and isinstance(b, int):
        return a, string.ascii_uppercase[b]

def one_hot_encode(item, length, topk=1):
    if isinstance(item, int):
        return [1 if i == item else 0 for i in range(length)]
    elif isinstance(item, list):
        if len(item) == 0:
            return [0 for i in range(length)]
        if isinstance(item[0], str):
            return [1 if string.ascii_uppercase[i] in item else 0 for i in range(length)]
        if isinstance(item[0], int):
            return [1 if i in item else 0 for i in range(length)]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def get_acc_of_multiple_choice(pred, gold):
    # Wrong choice don't get points, and the right part ones get only part of the points.
    if len(pred) != len(gold):
        raise ValueError
    correct = sum([i & j for i, j in zip(pred, gold)])
    # return 1 if correct > 0 else 0
    wrong = sum([i & ~j for i, j in zip(pred, gold)])
    # return correct / sum(gold) * 1.0 if wrong == 0 else 0.0
    correct_abs = max(0, correct - wrong)
    return correct_abs / sum(gold)