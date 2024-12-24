
class DirectFilter(object):
    def __init__(self, *args, **kwargs):
        ...
    def apply(self, response, **kwargs):
        return {'filtered_response': response, 'is_filtered': True}

filter_core = DirectFilter
