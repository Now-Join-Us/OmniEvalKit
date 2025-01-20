
class DirectFilter(object):
<<<<<<< HEAD
    def __init__(self, **kwargs) -> None:
        ...

    def apply(self, response, **kwargs):
        return response
=======
    def __init__(self, *args, **kwargs):
        ...
    def apply(self, response, **kwargs):
        return {'filtered_response': response, 'is_filtered': True}

filter_core = DirectFilter
>>>>>>> aef4b81d4d7526f3d22d331a4e308ff367100185
