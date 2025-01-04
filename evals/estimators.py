from collections import defaultdict
import numpy as np

class BaseEstimator:
    @staticmethod
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

    @staticmethod
    def avg_k(scores, **kwargs):
        sum_dict = defaultdict(float)
        count_dict = defaultdict(int)
        
        for item in scores:
            pass_at_k = item.get('pass_at_k', {})
            for k, acc in pass_at_k.items():
                sum_dict[k] += acc
                count_dict[k] += 1
        
        # 计算每个k的平均值
        avg_pass_at_k = {}
        for k in sum_dict:
            avg_pass_at_k[k] = sum_dict[k] / count_dict[k]
        
        # 构建结果字典，格式与每一项相同
        result = {"pass_at_k": avg_pass_at_k}
        return result
    
    @staticmethod
    def sum_or_avg(scores, categories, sub_categories, considered_categories=None, e_type='sum', **kwargs):
        import pdb; pdb.set_trace()
        # scores : 列表，每一项是calculate返回的字典
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
