from collections import defaultdict

def avg_k(scores, **kwargs):
        sum_dict = defaultdict(float)
        count_dict = defaultdict(int)
        
        for item in scores:
            pass_at_k = item.get('pass_at_k', {})
            for k, acc in pass_at_k.items():
                sum_dict[k] += acc
                count_dict[k] += 1
        
        avg_pass_at_k = {}
        for k in sum_dict:
            avg_pass_at_k[k] = sum_dict[k] / count_dict[k]
    
        result = {"pass_at_k": avg_pass_at_k}
        return result