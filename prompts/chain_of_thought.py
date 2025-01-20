from utils import detect_language

COT_PROMPT = "{prompt_instruction} {guide_words}"

def cot_prompt(instruction):
    language = detect_language(instruction)
    return COT_PROMPT.format(instruction.rstrip(), TYPE2LANGUAGE2PROMPT['cot'][language])
