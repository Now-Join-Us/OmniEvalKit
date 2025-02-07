from utils import detect_language
from prompts.base import translate_prompt, FILTER_TYPE2LANGUAGE2PROMPT
from evals.filters import RegexFilter

class ModelBasedFilter(RegexFilter):
    def __init__(
        self,
        dataset_name,
        model,
        custom_patterns=[],
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        """
        super().__init__(dataset_name)
        self.model = model

        def _get_prompt(question_type):
            def core(language, question, resp):
                prompt = FILTER_TYPE2LANGUAGE2PROMPT[question_type][language]
                prompt += translate_prompt('Question: ', language) + question + '\n' + \
                    translate_prompt('Answer: ', language) + resp + '\n' + \
                    translate_prompt('Output: ', language) + '\n'

                return prompt
            return core
        self.get_prompt = {
            question_type: _get_prompt(question_type) for question_type in ['multiple_choice', 'open', 'yes_or_no']
        }

    def preprocess(self, resp, question_type, prompt_instruction, choices=None):
        language = detect_language(resp)
        filter_prompt = self.get_prompt[question_type](
            language=language,
            question=prompt_instruction,
            resp=resp
        )
        filtered = self.model.generate_text_only(filter_prompt)
        if question_type == 'yes_or_no':
            if 'yes' in filtered.lower():
                return {'filtered_response': 'Yes', 'is_filtered': True}
            elif 'no' in filtered.lower():
                return {'filtered_response': 'No', 'is_filtered': True}
            return {'filtered_response': resp, 'is_filtered': False}

        elif question_type == 'multiple_choice':
            return super().choices_preprocess(
                resp=filtered,
                choices=choices
            )

        return {'filtered_response': filtered, 'is_filtered': True}

    def apply(self, response, data, question_type=None):
        if isinstance(response, dict) and 'is_filtered' in response.keys() and response['is_filtered']:
            return response

        if isinstance(response, list):
            matched = {'filtered_response': response, 'is_filtered': True}
        elif isinstance(response, str):
            matched = self.preprocess(
                resp=response,
                question_type=data['question_type'],
                prompt_instruction=data['raw_instruction'],
                choices=data['choices'] if 'choices' in data.keys() else None
            )
        else:
            raise NotImplementedError
        return matched
