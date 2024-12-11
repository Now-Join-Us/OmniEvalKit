import re
import string
from difflib import SequenceMatcher

def fuzzy_match(word, sentence):
    seq_match = SequenceMatcher(None, word.lower(), sentence.lower())
    match = seq_match.find_longest_match(0, len(word), 0, len(sentence))
    similarity = match.size / len(word)
    return similarity

def remove_image_token(instruction, tokens):
    for src in tokens:
        if src in instruction:
            instruction = instruction.replace(src, '')
    return instruction

def retain_only_first_sub_str(s, sub_s):
    first_index = s.find(sub_s)

    if first_index != -1:
        s = s[:first_index + len(sub_s)] + s[first_index + len(sub_s):].replace(sub_s, '')
    return s

def replace_image_token(instruction, source_default_tokens, target_tokens, leaved_token_num=1):
    if isinstance(target_tokens, str):
        target_tokens = [target_tokens] * len(source_default_tokens)
    target_id = 0
    for src in source_default_tokens:
        if src in instruction:
            instruction = instruction.replace(src, target_tokens[target_id])
            instruction = retain_only_first_sub_str(instruction, target_tokens[target_id])
            target_id += 1
        if target_id >= leaved_token_num:
            break
    if target_id == 0 and len(target_tokens) > 0:
        instruction = target_tokens[0] + '\n' + instruction
    else:
        instruction = remove_image_token(instruction, source_default_tokens)
    return instruction

def place_begin_image_token(instruction, source_default_tokens, target_tokens, leaved_token_num=1, sep='\n'):
    if isinstance(target_tokens, str):
        target_tokens = [target_tokens] * len(source_default_tokens)

    for src in source_default_tokens:
        if src in instruction:
            instruction = instruction.replace(src, '')

    begin_image_tokens = sep.join(target_tokens[:leaved_token_num])
    return begin_image_tokens + sep + instruction

def choices_raw_match(resp, choices):
    if len(choices) == 0 or len(resp) == 0:
        return {'value': resp, 'is_filtered': False}
    for i_option, i_choice in zip(string.ascii_uppercase[:len(choices)], choices):
        if resp.lower() in i_choice.lower():
            return {'value': [i_option], 'is_filtered': True}
    return {'value': resp, 'is_filtered': False}

def choices_fuzzy_match(resp, choices, gold):
    matched_choices = []
    similarities = []

    if isinstance(gold, str) or isinstance(gold, int):
        gold_length = 1
    elif isinstance(gold, list):
    	gold_length = 0 if len(gold) == 0 or isinstance(gold[0], str) else sum(gold)

    for i in range(len(choices)):
        similarity = fuzzy_match(choices[i], sentence=resp)
        if similarity > 0.5:
            matched_choices.append(string.ascii_uppercase[i])
            similarities.append(similarity)

    if len(matched_choices) == len(choices) and any([i > 0.95 for i in similarities]) and gold_length != len(choices):
        return {'value': resp, 'is_filtered': False}

    if matched_choices:
        combined = list(zip(matched_choices, similarities))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        sorted_matched_choices, sorted_similarities = zip(*sorted_combined)
        sorted_matched_choices = list(sorted_matched_choices)
        return {'value': list(dict.fromkeys(sorted_matched_choices)), 'is_filtered': True}
    else:
        return {'value': resp, 'is_filtered': False}
