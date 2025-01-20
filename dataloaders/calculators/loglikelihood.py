import numpy as np
from dataloaders.calculators.multiple_choice import multiple_choice

def loglikelihood(filtered_r, is_filtered, gold, choices_length, prompt_choices, **kwargs):
    if not is_filtered:
        return {'acc': 0, 'acc_norm': 0}
    filtered_r = filtered_r[:len(choices_length)]
    lls, is_greedy = zip(*filtered_r)
    completion_len = np.array([float(i) for i in choices_length])
    topk = sum(gold) if isinstance(gold, list) else 1
    pred = sorted(range(len(lls)), key=lambda i: lls[i], reverse=True)[:topk]
    lls_norm = lls / completion_len
    pred_norm = sorted(range(len(lls_norm)), key=lambda i: lls_norm[i], reverse=True)[:topk]

    return {
        'acc': multiple_choice(filtered_r=pred, is_filtered=True, gold=gold)['acc'],
        'acc_norm': multiple_choice(filtered_r=pred_norm, is_filtered=True, gold=gold)['acc']
    }, {
        'filtered_r': [prompt_choices[i] for i in pred],
        'filtered_r_norm': [prompt_choices[i] for i in pred_norm]
    }
