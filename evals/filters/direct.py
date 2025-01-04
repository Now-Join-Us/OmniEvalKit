class DirectFilter(object):
    def __init__(self, **kwargs) -> None:
        ...

    def apply(self, response, data, question_type=None):
        return {'filtered_response': response, 'is_filtered': True}

filter_core = DirectFilter
