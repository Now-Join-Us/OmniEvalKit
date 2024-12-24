
class ExtractTailFilter(object):
    def __init__(self, *args, **kwargs):
        ...
    def apply(self, response, **kwargs):
        split_point = len(response) // 5 if len(response) > 50 else 0
        return {'filtered_response': response[-split_point:], 'is_filtered': True}

filter_core = ExtractTailFilter
