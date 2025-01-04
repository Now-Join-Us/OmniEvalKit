import re
import string

from utils import most_common_length_strings
from evals.utils import choices_fuzzy_match


class RegexFilter(object):
    def __init__(
        self,
        dataset_name,
        custom_patterns=[],
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        """

        self.dataset_name = dataset_name

        def get_multiple_choice_patterns(option_str):
            patterns = [
                rf'[Tt]he answer is:?\s+\(?([{option_str}])\)?',
                rf'[Tt]he answer is option:?\s+\(?([{option_str}])\)?',
                rf'[Tt]he correct answer is:?\s+\(?([{option_str}])\)?',
                rf'[Tt]he correct answer is:? ?([{option_str}])',
                rf'[Tt]he correct answer is option:?\s+\(?([{option_str}])\)?',
                rf'[Tt]he answer to the question is:?\s+\(?([{option_str}])\)?',
                rf'^选项\s?([{option_str}])',
                rf'^([{option_str}])\s-?选?项',
                rf'(\s|^)[{option_str}][\s。，,：:\.$]',
                rf'(\s|^)[{option_str}](\s|$)',
                # rf'1.\s?(.*?)$',
                rf'1.\s?([{option_str}])[.。$]?$',
                rf'[Tt]he answer should be\s?\(?([{option_str}])\)?',
                rf'[Tt]he correct choice is\s?\(?([{option_str}])\)?',
                rf'[Tt]he right answer is\s?\(?([{option_str}])\)?',
                rf'[Tt]he right choice is\s?\(?([{option_str}])\)?',
                rf'[Cc]hoose\s?\(?([{option_str}])\)?',
                rf'[Oo]ption\s?\(?([{option_str}])\)?\s?is correct',
                rf'[Aa]nswer:\s?\(?([{option_str}])\)?',
                rf'[Cc]hoice:\s?\(?([{option_str}])\)?',
                rf'[Ss]elect\s?\(?([{option_str}])\)?',
                rf'[Tt]he selection is\s?\(?([{option_str}])\)?',
                rf'[Cc]orrect answer:\s?\(?([{option_str}])\)?',
                rf'[Tt]he answer given is\s?\(?([{option_str}])\)?',
                rf'([{option_str}])\)?\s?is the answer',
                rf'([{option_str}])\)?\s?is correct',
                rf'([{option_str}])\)?\s?is the correct answer',
                rf'答案是?\s*和?或?者?\s*([{option_str}])',
                rf'答案是?\s*：\s*和?或?者?\s*([{option_str}])',
                rf'答案是?\s*:\s*和?或?者?\s*([{option_str}])',
                rf'答案应该?是\s*和?或?者?\s*([{option_str}])',
                rf'答案应该?选择?\s*和?或?者?\s*([{option_str}])',
                rf'答案为\s*和?或?者?\s*([{option_str}])',
                rf'答案选择?\s*和?或?者?\s*([{option_str}])',
                rf'选择?\s*和?或?者?\s*([{option_str}])',
                rf'故选?择?\s*和?或?者?\s*([{option_str}])',
                rf'只有选?项?\s*和?或?者?\s?([{option_str}])\s?是?对',
                rf'只有选?项?\s*和?或?者?\s?([{option_str}])\s?是?错',
                rf'只有选?项?\s*和?或?者?\s?([{option_str}])\s?不?正确',
                rf'只有选?项?\s*和?或?者?\s?([{option_str}])\s?错误',
                rf'说法不?对选?项?的?是\s*和?或?者?\s?([{option_str}])',
                rf'说法不?正确选?项?的?是\s*和?或?者?\s?([{option_str}])',
                rf'说法错误选?项?的?是\s*和?或?者?\s?([{option_str}])',
                rf'([{option_str}])\s?是正确的',
                rf'([{option_str}])\s?是正确答案',
                rf'选项\s?([{option_str}])\s?正确',
                rf'所以答\s?([{option_str}])',
                rf'所以\s?([{option_str}][.。$]?$)',
                rf'所有\s?([{option_str}][.。$]?$)',
                rf'[\s，：:,]([{option_str}])[。，,\.]?$',
                rf'[\s，,：:][故即]([{option_str}])[。\.]?$',
                rf'[\s，,：:]因此([{option_str}])[。\.]?$',
                rf'[是为。]\s?([{option_str}])[。\.]?$',
                rf'因此(选了)?\s?([{option_str}])',
                rf'显然是?选?\s?([{option_str}])',
                # rf'答案是\s?(\S+)(?:。|$)',
                # rf'答案应该是\s?(\S+)(?:。|$)',
                # rf'答案为\s?(\S+)(?:。|$)',
                rf'正确答案是\s?\(?([{option_str}])\)?',
                rf'应选择\s?\(?([{option_str}])\)?',
                rf'选择的是\s?\(?([{option_str}])\)?',
                rf'应选\s?\(?([{option_str}])\)?',
                rf'应答\s?\(?([{option_str}])\)?',
                rf'选择\s?\(?([{option_str}])\)?\s?是正确的',
                rf'最终答案是\s?\(?([{option_str}])\)?',
                rf'最后选择的选项是\s?\(?([{option_str}])\)?',
                rf'题目的答案是\s?\(?([{option_str}])\)?',
                rf'最终答案为\s?\(?([{option_str}])\)?',
                rf'最终选项是\s?\(?([{option_str}])\)?',
                rf'应该选择\s?\(?([{option_str}])\)?',
                rf'答案解析\s?\(?([{option_str}])\)?',
                rf'解析选（?择）?\s?\(?([{option_str}])\)?',
                rf'应答选\s?\(?([{option_str}])\)?',
                rf'[选择答案为]\s?\(?([{option_str}])\)?',
                rf'[正答选择是]\s?\(?([{option_str}])\)?',
                rf'答案\s?([{option_str}])',
                rf'正确的是\s?\(?([{option_str}])\)?',
                rf'([{option_str}])\s?为答案',
                rf'([{option_str}])\s?答对',
                rf'答案：\s?([{option_str}])',
                rf'题目答案：\s?([{option_str}])',
                rf'答案分析：?\s?\(?([{option_str}])\)?',
                rf'[Tt]he final answer is\s?\(?([{option_str}])\)?',
                rf'[Tt]he chosen answer is\s?\(?([{option_str}])\)?',
                rf'[Tt]orrect option is\s?\(?([{option_str}])\)?',
                rf'[Ff]inal choice:\s?\(?([{option_str}])\)?',
                rf'[S|s]elected option is\s?\(?([{option_str}])\)?',
                rf'[C|c]orrect option:\s?\(?([{option_str}])\)?',
                rf'[T|t]he right answer:\s?\(?([{option_str}])\)?',
                rf'[A|a]nswer choice:\s?\(?([{option_str}])\)?',
                rf'[A|a]nswer selected:\s?\(?([{option_str}])\)?',
                rf'[Y|y]our answer is\s?\(?([{option_str}])\)?',
                rf'^[Tt]he answer:(\s?)([{option_str}])',
                rf'^[Tt]he option:(\s?)([{option_str}])',
                rf'^[Aa]ns:\s?([{option_str}])',
                rf'^[Aa]nswer:\s?([{option_str}])',
                rf'^[Rr]esponse:\s?([{option_str}])',
                rf'^Answer is\s?([{option_str}])',
                rf'^[Cc]hoice:\s?([{option_str}])',
                rf'^\(?([{option_str}])\)?\s?(is)? correct',
                rf'^\(?([{option_str}])\)?\s?selected',
                rf'正确选项是?\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'答案已经是?\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'最终答案为?\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'选择答案是?\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'应当选择?\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'应该回答\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'最后答案是?\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'答案应为\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'你选择的答案是?\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'考生答案是?\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'学者认为\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'调查显示\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'研究表明\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'根据数据\s*和?或?者?\s?\(?([{option_str}])\)?',
                rf'^答案:(\s?)([{option_str}])',
                rf'^选项:(\s?)([{option_str}])',
                rf'^答案是:(\s?)([{option_str}])',
                rf'^选答案是:(\s?)([{option_str}])',
                rf'^\(?([{option_str}])\)?\s?为正确答案',
                rf'^\(?([{option_str}])\)?\s?是选项',
                rf'答案：\s?\(?([{option_str}])\)?',
                rf'选答案：\s?\(?([{option_str}])\)?',
                rf'根据\s+\(?([{option_str}])\)?',
                rf'\(*correct\s*answer\s*is\s*\(*([{option_str}])\)*',
                rf'\(*right\s*answer\s*is\s*\(*([{option_str}])\)*',
                rf'\(*selected\s*answer\s*is\s*\(*([{option_str}])\)*',
                rf'\(*correct\s*option\s*is\s*\(*([{option_str}])\)*',
                rf'\(*right\s*option\s*is\s*\(*([{option_str}])\)*',
                rf'\(*selected\s*option\s*is\s*\(*([{option_str}])\)*',
                rf'\(*final\s*answer\s*is\s*\(*([{option_str}])\)*',
                rf'\(*final\s*option\s*is\s*\(*([{option_str}])\)*',
                rf'\(*choice\s*is\s*\(*([{option_str}])\)*',
                rf'[Oo]ption\s?\)?([{option_str}])\)?\s?is\s?correct',
                rf'answer\s?\)?([{option_str}])\)?\s?is\s?correct',
                rf'selected\s?Option\s?\(?([{option_str}])\)?',
                rf'\(?\s*([{option_str}])\)?\s*is\s*the\s*correct\s*answer',
                rf'^\(*([{option_str}])\)*\s+Answer',
                rf'\s*[选为是：\(]*\s?([{option_str}])[\)\.。，,：:\?$]',
                rf'^\(?([{option_str}])\)?$',
                rf'^\(?([{option_str}])\)?\s*选$',
                rf'^.\(*([{option_str}])\s*。\)*$',
                rf'\s+([{option_str}])\s+[is|为]{0,2}.?$',
                rf"\s*([{option_str}])"
            ]
            # cushion_patterns = [
            #     f'([{option_str}]):',
            #     f'[{option_str}]',
            # ]
            # patterns.extend(cushion_patterns + custom_patterns)
            return patterns
        self.get_multiple_choice_patterns = get_multiple_choice_patterns

        def get_open_patterns():

            patterns = [
                r"答案应?该?是?[:：]?\s*([^。！.!]*)([。！.!])",
                r"所以应?该?为?是?[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) correct answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) right answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"这道题的正确答案应?该?是?[:：]?\s*([^。！.!]*)([。！.!])",
                r"正确答案为[:：]?\s*([^。！.!]*)([。！.!])",
                r"答案是[:：]?\s*([^。！.!]*)([。！.!])",
                r"所以答案是[:：]?\s*([^。！.!]*)([。！.!])",
                r"正确的答案是[:：]?\s*([^。！.!]*)([。！.!])",
                r"回答是[:：]?\s*([^。！.!]*)([。！.!])",
                r"结论是[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) solution (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) response (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) answer should (?:be|is)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) correct response (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) accurate answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"由此可知答案是[:：]?\s*([^。！.!]*)([。！.!])",
                r"最终答案是[:：]?\s*([^。！.!]*)([。！.!])",
                r"答案应该是[:：]?\s*([^。！.!]*)([。！.!])",
                r"可以得到答案是[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) final answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) result (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) answer can (?:be|is)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:We|we) find the answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) conclusion (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) answer to this question (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) correct solution (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:It|it) can be concluded that the answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:In|in) conclusion, the answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:Thus|thus), the answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:From|from) this, we can infer the answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) derived answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) exact answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:To|to) summarize, the answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:Ultimately|ultimately), the answer (?:is|are)\s*[:：]?\s*([^。！.!]*)([。！.!])",
                r"(?:The|the) answer(?: is| are)?\s*[:：]?\s*([^。！.!]*)[\s。！.!]?",
            ]

            if 'gsm8k' in self.dataset_name:
                patterns = ["#### (\\-?[0-9\\.\\,]+)",
                            "(-?[$0-9.,]{2,})|(-?[0-9]+)"]
            # elif 'logieval' in self.dataset_name:
            #     patterns = ["^\\s*([A-D])"] + patterns
            elif 'eq_bench' in self.dataset_name:
                patterns = ["(\w+):\s+(\d+)"]
            elif 'bbh' in self.dataset_name:
                patterns = ["(?<=the answer is )(.*)(?=.)"] + patterns

            return patterns
        self.get_open_patterns = get_open_patterns

        def get_binary_patterns():
            patterns = {
                'yes': [
                    'yes',
                    'true',
                    'right',
                    'correct',
                    'agree',
                    'certain',
                    'indeed',
                    'absolute',
                    'affirmative',
                    'accurate',
                    'factual',
                    'genuine',
                    'approve',
                    'authentic',
                    '对',
                    '真',
                    '正确',
                    '同意',
                    '当然',
                    '的确',
                    '确实',
                    '是',
                    '准确',
                    '无误',
                    '确定'
                ],
                'no': [
                    'no',
                    'false',
                    'incorrect',
                    'disagree',
                    'inaccurate',
                    'unreal',
                    'erroneous',
                    'wrong',
                    'unreliable',
                    '否',
                    '假',
                    '错误',
                    '不同意',
                    '不'
                ],
                'others': [
                    'so',
                    'therefore',
                    'thus',
                    'hence',
                    'consequently',
                    'accordingly',
                    'as a result',
                    '所以',
                    '因此',
                    '总而言之',
                    '因而',
                    '由此可见',
                    '综上所述',
                    '因之',
                    '故此'
                ]
            }

            return patterns

        self.get_binary_patterns = get_binary_patterns

    def choices_preprocess(self, resp, choices):
        option_str = string.ascii_uppercase[:len(choices)]
        if len(choices) == 0:
            option_str = 'ABCDEF'

        patterns = self.get_multiple_choice_patterns(option_str)
        matched_choices = []
        while True:
            found = False
            for pattern in patterns:
                matched = re.search(pattern, resp, re.DOTALL)
                if matched:
                    outputs = matched.group(0)
                    for i in option_str:
                        if i in outputs:
                            matched_choices.append(i)
                            resp = re.sub(i, '', resp, 1)
                            found = True
                            break
                    if found:
                        break
            if not found:
                break
        if len(matched_choices) == 0:
            def filter_only_letter_strings():
                valid_chars_pattern = rf'^[{option_str}{re.escape(string.punctuation)}\s]*$'

                if re.match(valid_chars_pattern, resp):
                    matches = re.findall(rf'[{option_str}]', resp)
                    return matches
                else:
                    return []
            matched_choices = filter_only_letter_strings()

        if len(matched_choices) > 0:
            return {
                'filtered_response': list(dict.fromkeys(matched_choices)),
                'is_filtered': True
            }

        return {'filtered_response': resp, 'is_filtered': False}

    def binary_preprocess(self, resp):
        patterns = self.get_binary_patterns()

        def get_first_appear_index(patterns, s):
            pattern_index = float('inf')
            matched_pattern = None
            for p in patterns:
                if p.lower() in s.lower():
                    matched_index = s.lower().find(p.lower())
                    if matched_index < pattern_index:
                        pattern_index = matched_index
                        matched_pattern = p

            return pattern_index, matched_pattern

        # other_pattern_first_index, other_pattern_matched = get_first_appear_index(patterns['others'], resp)
        # begin_index = 0 if other_pattern_matched is None else other_pattern_first_index + len(other_pattern_matched)
        begin_index = 0
        yes_first_index, yes_pattern_matched = get_first_appear_index(patterns['yes'], resp[begin_index:])
        no_first_index, no_pattern_matched = get_first_appear_index(patterns['no'], resp[begin_index:])

        if yes_pattern_matched is None and no_pattern_matched is None:
            return {'filtered_response': resp, 'is_filtered': False}

        return {'filtered_response': 'Yes', 'is_filtered': True} if yes_first_index <= no_first_index else {'filtered_response': 'No', 'is_filtered': True}

    def open_preprocess(self, resp):
        patterns = self.get_open_patterns()
        matched_open_results = []
        if self.dataset_name == 'eq_bench':
            find_resps = re.findall(patterns[0], resp)
            if find_resps:
                return {'filtered_response': find_resps, 'is_filtered': True}
            else:
                return {'filtered_response': resp, 'is_filtered': False}
        else:
            for pattern in patterns:
                matched = re.search(pattern, resp, re.DOTALL)
                if matched:
                    outputs = matched.group(1)
                    matched_open_results.append(outputs)
            # return max(matched_open_results, key=len) if len(matched_open_results) > 0 else matched_open_results
            if len(matched_open_results) > 0 and matched_open_results[0] is not None:

                matched_open_results = [i for i in matched_open_results if i is not None]

                return {
                    'filtered_response': most_common_length_strings(matched_open_results),
                    'is_filtered': True
                }
            return {'filtered_response': resp, 'is_filtered': False}


    def apply(self, response, data, question_type):
        ## 优先级：question_type > data['question_type']
        entry_question_type = question_type if question_type is not None else data.get('question_type', 'open')
        if isinstance(response, dict):
            if 'content' in response.keys():
                response = response['content']

        if isinstance(response, list):
            matched = {'filtered_response': response, 'is_filtered': True}
        elif isinstance(response, str):
            if entry_question_type == 'multiple_choice':
                matched = self.choices_preprocess(response, data['choices'])
                if not matched['is_filtered']:
                    matched = self.choices_preprocess(self.open_preprocess(response)['filtered_response'], data['choices'])
                if not matched['is_filtered']:
                    matched = choices_fuzzy_match(response, data['prompt_choices'] if 'prompt_choices' in data.keys() else data['choices'], data['gold'])
                if not matched['is_filtered'] and 'prompt_choices' in data.keys() and 'choices' in data.keys():
                    matched = choices_fuzzy_match(response, data['choices'], data['gold'])
            elif entry_question_type == 'open':
                matched = self.open_preprocess(response)
            elif entry_question_type == 'yes_or_no':
                matched = self.binary_preprocess(response)
            else:
                raise NotImplementedError(f'Unhandled question type: {question_type}')
        else:
            raise NotImplementedError(f'Unhandled response type: {type(response)}')

        return matched

filter_core = RegexFilter
