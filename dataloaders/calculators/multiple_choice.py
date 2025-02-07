from dataloaders.calculators.utils import align_two_type, get_acc_of_multiple_choice, one_hot_encode

def multiple_choice(filtered_r, is_filtered, gold, **kwargs):
    # filtered_r: int, list of str, or list of int
    # gold: int, str, or one_hot

    if not is_filtered:
        return {'acc': 0}
    acc = 0.0
    if isinstance(gold, int) or isinstance(gold, str):
        # single choice
        if isinstance(filtered_r, list):
            if len(set(filtered_r)) == 1:
                filtered_r = list(set(filtered_r))[0]
                filtered_r, gold = align_two_type(filtered_r, gold)
                acc = 1.0 if filtered_r == gold else 0.0
            else:
                filtered_r = filtered_r[0]
                filtered_r, gold = align_two_type(filtered_r, gold)
                acc = 1.0 if filtered_r == gold else 0.0
        elif isinstance(filtered_r, int) or isinstance(filtered_r, str):
            filtered_r, gold = align_two_type(filtered_r, gold)
            acc = 1.0 if filtered_r == gold else 0.0
        else:
            raise NotImplementedError(f'Unhandled filtered_r type: {type(filtered_r)}')
    elif isinstance(gold, list):
        # multiple choices & gold is one_hot
        filtered_r = one_hot_encode(filtered_r, length=len(gold))
        acc = get_acc_of_multiple_choice(filtered_r, gold)
    else:
        raise NotImplementedError(f'Unhandled gold type: {type(gold)}')
    return {'acc': acc}
