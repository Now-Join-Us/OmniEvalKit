
import re

def get_first_number(text):
    pattern = r'\b\d+(\.\d+)?\b'
    match = re.search(pattern, text)
    if match:
        try:
            number = float(match.group(0))
        except:
            number = text
        return number
    else:
        return text

class NumberAfterLastIsFilter(object):
    def __init__(self, **kwargs) -> None:
        ...

    def apply(self, response, **kwargs):
        return get_first_number(response.split('is')[-1].strip())
