def mean_square_error(filtered_r, gold, **kwargs):
    if not isinstance(gold, list):
        gold = [gold]
    if not isinstance(filtered_r, list):
        filtered_r = [filtered_r]

    flattened_references = flatten_list(gold)
    total_exact_match_score = sum(
        [
            hf_exact_match.compute(predictions=filtered_r, references=[i_ref] * len(filtered_r), **kwargs)['exact_match'] \
                for i_ref in flattened_references
        ]
    )
    if max_to_0_1:
        return {'acc': 1.0} if total_exact_match_score > 0 else {'acc': 0.0}

    return {'acc': total_exact_match_score}
