from collections import defaultdict

def shrink_corresponding(scores, shrink_pairs, categories, sub_categories):
    shrunk_results = defaultdict(lambda: defaultdict(lambda: 1))

    for i, pair in enumerate(shrink_pairs):
        for metric, score in scores[i].items():
            shrunk_results[pair][metric] *= score

    shrunk_results = {k: dict(v) for k, v in shrunk_results.items()}

    for i in range(len(shrink_pairs)):
        shrunk_results[shrink_pairs[i]]['category'] = categories[i]
        if sub_categories:
            shrunk_results[shrink_pairs[i]]['sub_category'] = sub_categories[i]

    return shrunk_results