def sum_or_avg(scores, categories, sub_categories, considered_categories=None, e_type='sum'):
    category2metric2static = {}
    def _s_or_a(inputs):
        assert len(inputs) != 0
        metric2static = {}
        collected_metrics = sorted(list(set(inputs[0].keys())))
        for metric in collected_metrics:
            raw_scores = [i[metric] for i in inputs]
            metric2static[metric] = sum(raw_scores)
            if e_type == 'avg':
                metric2static[metric] = metric2static[metric] / len(raw_scores)

        return metric2static

    categories = [] if categories is None else categories
    sub_categories = [] if sub_categories is None or set(sub_categories) == {None} else sub_categories
    considered_categories = considered_categories if considered_categories is not None else ['full'] + list(set(categories + sub_categories))
    considered_categories = sorted(considered_categories)

    for cat in considered_categories:
        if cat == 'full':
            masked_scores = scores
        else:
            masked_categories = categories if cat in categories else sub_categories
            masked_scores = [i_score for i_score, m_cat in zip(scores, masked_categories) if cat == m_cat]

        category2metric2static[cat] = _s_or_a(masked_scores)
    return category2metric2static