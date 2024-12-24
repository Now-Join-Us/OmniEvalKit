# Copyright (C) 2024 AIDC-AI

MODEL_PATH = 'your_models_path/models'
DATA_PATH = './data'
OUTPUT_PATH = 'your_outputs_path/outputs'


import os
_DEFAULT_MAX_LENGTH = 2048

MAX_GEN_TOKS = 1024


datasets = \
    ['hellaswag', 'mmlu', 'arc_challenge', 'arc_easy', 'winogrande', 'truthfulqa_mc1', 'truthfulqa_mc2', 'aclue', 'anli', 'boolq', 'cb', 'cmmlu', 'cola', 'crows_pairs', 'copa', 'glue', 'lambada', 'mathqa', 'mnli', 'mrpc', 'openbookqa', 'piqa'] + \
        ['bbh', 'gsm8k', 'drop', 'logieval', 'eq_bench', 'nq_open', 'realworldqa'] + \
            ['mmmu_val', 'ccbench', 'mme', 'ai2d_test', 'ccbench', 'coco_val', 'hallusionbench', 'mathvista_mini', 'mme', 'mmstar', 'mmvet', 'ocrbench', 'realworldqa', 'scienceqa_test', 'scienceqa_val', 'seedbench_img', 'mmbench_dev_cn', 'mmbench_dev_en', 'mmbench_test_cn', 'mmbench_test_en'] + \
                ['cs_mmlu_kw_3', 'cs_mmlu_kw_5', 'cs_mmlu_kw_10', 'cs_mmlu_kw_20', 'cs_mmlu_st_3', 'cs_mmlu_st_5', 'cs_mmlu_st_10', 'cs_mmlu_st_20'] + \
                    ['agieval', 'arc_multilingual', 'hellaswag_multilingual', 'truthfulqa_multilingual_mc2', 'belebele', 'xcopa', 'translation', 'xstorycloze', 'truthfulqa_multilingual_mc1'] + \
                        ['hellaswag_multilingual_sampled', 'truthfulqa_multilingual_mc2_sampled', 'arc_multilingual_sampled', 'xstorycloze_sampled'] + \
                            ['multimodal_complexity', 'mc']

_MODULE2MODEL = {
    'multimodal_llm.deepseek_ai.deepseek_vl': ['deepseek-vl-1.3b-chat', 'deepseek-vl-7b-chat'],
    'multimodal_llm.echo840.monkey': ['Monkey', 'Monkey-Chat'],
    'multimodal_llm.thudm.cogvlm': ['cogvlm2-llama3-chat-19B', 'cogvlm-chat-hf'],
    'multimodal_llm.thudm.glm_4v': ['glm-4v-9b'],
    'multimodal_llm.qwen.qwen_vl': ['Qwen-VL'],
    'multimodal_llm.qwen.qwen_vl_chat': ['Qwen-VL-Chat'],
    'multimodal_llm.qwen.qwen2_vl_instruct': ['Qwen2-VL-7B-Instruct'],
    'multimodal_llm.01_ai.yi_vl': ['Yi-VL-6B'],
    'multimodal_llm.microsoft.phi3_vision': ['Phi-3-vision-128k-instruct', 'Phi-3.5-vision-instruct'],
    'multimodal_llm.microsoft.e5_v': ['e5-v'],
    'multimodal_llm.openbmb.vision_cair': ['MiniGPT-4'],
    'multimodal_llm.openbmb.minicpm_v': ['MiniCPM-V', 'MiniCPM-V-2'],
    'multimodal_llm.openbmb.minicpm_llama3_v': ['MiniCPM-Llama3-V-2_5'],
    'multimodal_llm.openbmb.minicpm_v_2_6': ['MiniCPM-V-2_6'],
    'multimodal_llm.liuhaotian.llava_v1_5': ['llava-v1.5-7b'],
    'multimodal_llm.baai.bunny_llama3_v': ['Bunny-Llama-3-8B-V'],
    'multimodal_llm.huggingfacem4.idefics2': ['idefics2-8b'],
    'multimodal_llm.aidc_ai.ovis': ['Ovis-Clip-Llama3-8B', 'Ovis-Clip-Qwen1_5-7B'],
    'multimodal_llm.aidc_ai.ovis1_6': ['Ovis1.6-Gemma2-9B'],
    'multimodal_llm.internlm.internlm_xcomposer2': ['internlm-xcomposer2-vl-1_8b', 'internlm-xcomposer2-vl-7b', 'internlm-xcomposer2-7b'],
    'multimodal_llm.lamda_llm.wings': ['Wings'],
    'multimodal_llm.opengvlab.internvl2': ['InternVL2-8B'],
    'llm.qwen.qwen1_5_chat': ['Qwen1.5-0.5B-Chat', 'Qwen1.5-1.8B-Chat', 'Qwen1.5-4B-Chat', 'Qwen1.5-MoE-A2.7B-Chat', 'Qwen1.5-7B-Chat', 'Qwen2-0.5B-Instruct', 'Qwen2-1.5B-Instruct', 'Qwen2-7B-Instruct', 'Qwen2-72B-Instruct'],
    'llm.qwen.qwen_base': ['Qwen-1_8B', 'Qwen-7B'],
    'llm.qwen.qwen_chat': ['Qwen-1_8B-Chat', 'Qwen-7B-Chat'],
    'llm.qwen.qwen2_5_chat': ['Qwen2.5-7B-Instruct'],
    'llm.qwen.qwen_coder2': ['Qwen2.5-Coder-1.5B-Instruct'],
    'llm.01_ai.yi_chat': ['Yi-1.5-6B-Chat', 'Yi-1.5-9B-Chat', 'Yi-6B-Chat'],
    'llm.deepseek_ai.deepseek_llm': ['deepseek-llm-7b-chat'],
    'llm.internlm.internlm_chat': ['internlm2-chat-7b', 'internlm-chat-7b'],
    'llm.thudm.glm': ['glm-10b'],
    'llm.thudm.glm_4_chat': ['glm-4-9b-chat'],
    'llm.thudm.chatglm': ['chatglm-6b', 'chatglm2-6b', 'chatglm3-6b'],
    'llm.microsoft.phi3_5': ['Phi-3.5-mini-instruct'],
    'llm.microsoft.phi_instruct': ['Phi-3-mini-4k-instruct', 'Phi-3-mini-128k-instruct', 'Phi-3-small-8k-instruct', 'Phi-3-small-128k-instruct'],
    'llm.microsoft.ocra2': ['Orca-2-7b'],
    'llm.microsoft.phi2': ['phi-2'],
    'llm.microsoft.phi': ['phi-1', 'phi-1_5'],
    'llm.huggingfaceh4.zephyr': ['zephyr-7b-alpha', 'zephyr-7b-beta', 'mistral-7b-sft-beta'],
    'llm.huggingfaceh4.zephyr_gemma': ['zephyr-7b-gemma-v0.1', 'zephyr-7b-gemma-sft-v0.1'],
    'llm.tiiuae.falcon': ['falcon-7b-instruct'],
    'llm.tinyllama.tinyllama': ['TinyLlama-1.1B-Chat-v1.0'],
    'llm.open_ocra.open_ocra': ['Mistral-7B-OpenOrca'],
    'llm.stabilityai.stablebeluga2': ['StableBeluga2'],
    'llm.meta_llama.opt': ['opt-125m', 'opt-1.3b'],
    'llm.meta_llama.llama3': ['Meta-Llama-3-8B'],
    'llm.meta_llama.llama3_it': ['Meta-Llama-3-8B-Instruct'],
    'llm.meta_llama.llama3_1_it': ['Meta-Llama-3.1-8B-Instruct'],
    'llm.mlabonne.neuraldaredevil': ['NeuralDaredevil-7B'],
    'llm.xenon1.metamodel_moex8': ['MetaModel_moex8'],
    'llm.togethercomputer.redpajama': ['RedPajama-INCITE-Chat-3B-v1'],
    'llm.openai.gpt': ['gpt-4-turbo-128k', 'gpt-4o-0513'],
    'llm.openai.gpt_api': ['gpt-4o-mini'],
    'llm.openai.gpt2': ['gpt2-large', 'gpt2-medium'],
    'llm.google.gemma': ['gemma-2b', 'gemma-2b-it', 'gemma-7b-it', 'gemma-2-9b-it'],
    'llm.bigscience.bloomz': ['bloomz-560m', 'bloom-1b7'],
    'llm.baichuan_inc.baichuan2_chat': ['Baichuan2-7B-Chat'],
    'llm.openxlab.claude': ['claude-3-opus-20240229'],
    'llm.baichuan_inc.baichuan': ['Baichuan-7B'],
    'llm.ensemble.packllm': ['PackLLM'],
    'llm.learnware.xranker': ['xranker'],
    'llm.huggingfacetb.sollm': ['SmolLM-1.7B'],
    'llm.cohereforai.aya_expanse': ['aya-expanse-8b'],
    'llm.facebook.mobilellm': ['MobileLLM-125M'],
    'llm.alibaba_nlp.gte_qwen2_it': ['gte-Qwen2-7B-instruct', 'gte-Qwen2-1.5B-instruct'],
    'llm.test_llm': ['TestLLM']
}

_MODULE2DATASET = {
    'coco': ['coco_val'],
    'ocrbench': ['ocrbench'],
    'eq_bench': ['eq_bench'],
    'mme': ['mme'],
    'hallusionbench': ['hallusionbench'],
    'mm_cc_bench': ['mmbench', 'ccbench'],
    'truthfulqa_mc2': ['truthfulqa_mc2', 'truthfulqa_multilingual_mc2'],
    'bbh': ['bbh'],
    'drop': ['drop']
}

STANDARD_DATASET2SHOTS = {
    'arc_challenge': 25,
    'arc_easy': 25,
    'hellaswag': 10,
    'mmlu': 5,
    'winogrande': 5,
    'gsm8k': 5,
    'bbh': 3
}

MODEL2MODULE = {}
for module_type, models in _MODULE2MODEL.items():
    for model in models:
        MODEL2MODULE[model] = module_type

DATASET2MODULE = {}
for dataset_module, dataset_names in _MODULE2DATASET.items():
    for dataset in dataset_names:
        DATASET2MODULE[dataset] = dataset_module
        if dataset in STANDARD_DATASET2SHOTS:
            DATASET2MODULE[f'{dataset}_{STANDARD_DATASET2SHOTS[dataset]}'] = dataset_module

GEN_DATASET2UNTIL = {
    'bbh': ["</s>", "Q", "\n\n"],
    'gsm8k': ["</s>", "Question", "<|im_end|>", "<|endoftext|>"],
    'drop': ['.'],
    'eq_bench': ['\n\n'],
    'logieval': ['\n\n'],
    'nq_open': ['\n', '.', ','],
}

GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.0

DATASET2FILE = {
    i: f'{i}.json' for i in datasets
}

for d in STANDARD_DATASET2SHOTS.keys():
    d_shot = STANDARD_DATASET2SHOTS[d]
    d_new = f'{d}_{d_shot}'

    if d in GEN_DATASET2UNTIL.keys():
        GEN_DATASET2UNTIL[d_new] = GEN_DATASET2UNTIL[d]

    try:
        DATASET2FILE[d_new] = DATASET2FILE[d].replace(d, d_new)
    except Exception as e:
        pass


TYPE2LANGUAGE2PROMPT = {
    'multiple_choice': {
        'EN': 'Please select the correct answer from the options above.\n',
        'ZH': '请直接选择正确选项的字母。\n',
        'AR': 'الرجاء اختيار الإجابة الصحيحة من الخيارات أعلاه.\n',
        'RU': 'Пожалуйста, выберите букву правильного варианта напрямую.\n'
    },
    'open': {
        'EN': 'Please answer the question directly.\n',
        'ZH': '请直接回答问题。\n',
        'AR': 'يرجى الإجابة على السؤال مباشرة.\n',
        'RU': 'Пожалуйста, ответьте на вопрос прямо.\n'
    },
    'yes_or_no': {
        'EN': 'Please answer Yes or No.\n',
        'ZH': '请回答是或否。\n',
        'AR': 'من فضلك أجب بنعم أو لا.\n',
        'RU': 'Пожалуйста, ответьте Да или Нет.\n',
    },
    'cot': {
        'EN': 'Let\'s think step by step. ',
        'ZH': '让我们一步一步来思考。',
        'AR': 'دعنا نفكر خطوة بخطوة.',
        'RU': 'Давайте думать шаг за шагом.'
    }
}

FILTER_TYPE2LANGUAGE2PROMPT = {
    'multiple_choice': {
        'EN': '''
            Please help me match an answer to the multiple choices of the question.
            The option is ABCDEF.
            You get a question and an answer,
            You need to figure out options from ABCDEF that is most similar to the answer.
            If the meaning of all options is significantly different from the answer, we output Unknown.
            You should output one or more options from the following :ABCDEF.
            Example 1:
            Question: Which of the following numbers are positive? \nA.0\nB.-1\nC.5\nD.102\nE.-56\nF.33
            Answer: The answers are 5,102 and 33. \nYour output:CDF
            Example 2:
            Question: Which of these countries is landlocked? \nA.Mongolia \nB.United States \nC.China \nD.Japan
            Answer: A.Mongolia is A landlocked country. \nYour output:A
            Here are the target questions and answers.\n
        ''',

        'ZH': f'''
            请帮我把一个答案与问题的多个选项相匹配。
            选项有<possibilites>。
            你会得到一个问题和一个答案，
            你需要找出哪几个选项<possibilites>与答案最相似。
            如果所有选项的含义都与答案显著不同，则输出Unknown。
            你应该从以下几个个选项中输出一个或多个选项:<possibilites>。
            示例1:
            问题:下面那几个数字是正数？\nA.0\nB.-1\nC.5\nD.102\nE.-56\nF.33
            答案:答案是5，102和33。\n你的输出:CDF
            示例2:
            问题:下面哪个国家是内陆国家?\nA.蒙古\nB.美国\nC.中国\nD.日本
            答案：A.蒙古国是内陆国家。\n你的输出:A
            下面是目标的问题和答案。\n
        ''',
    },
    'open': {
        'EN': '''
        Please help me extract the answers to the given questions from the answers given.
        You get a question and an answer,
        If the answer and question are not relevant, the output is Unknown.
        Example 1:
        Question: What color is the sky?
        Answer: The sky is blue most of the time.\nYour output: The sky is blue
        Example 2:
        Question: Who invented the electric light?
        Answer: Edison is often credited with inventing the light bulb. In fact, he only improved it. The actual inventor of the light bulb was Henry Goebbels.\nYour output: Henry Goebbels
        Here are the target questions and answers. \n
        ''',
        'ZH': '''
            请帮我从给出的答案中提取出给出问题的回答。
            你会得到一个问题和一个答案，
            如果答案和问题不相关，则输出Unknown。
            示例1:
            问题:天空是什么颜色的？
            答案:天空是在大多数时候是蓝色的。\n你的输出:天空是蓝色的
            示例2:
            问题:是谁发明了电灯？
            答案：人们通常认为是爱迪生发明了电灯泡，实际上不然，他只是改进了电灯泡。电灯泡的实际发明人是亨利·戈培尔。\n你的输出:亨利·戈培尔
            下面是目标的问题和答案。\n
        '''
    },
    'yes_or_no': {
        'ZH': '''
            请帮我把一个答案与问题的两个选项相匹配。
            选项只有“是/否”。
            你会得到一个问题和一个答案，
            你需要找出哪个选项(是/否)与答案最相似。
            如果所有选项的含义都与答案显著不同，则输出Unknown。
            你应该从以下3个选项中输出一个单词:Yes, No, Unknown。
            示例1:
            问题:图像中的单词是“Hello”吗?
            答案:这个图像中的单词是“Hello”。\n你的输出:Yes
            示例2:
            问题:图像中的单词是“Hello”吗?
            答案:这个图像中的单词不是“Hello”。\n你的输出:No
            下面是目标的问题和答案。\n
        ''',

        'EN': '''
            Please help me to match an answer with two options of a question.
            The options are only Yes / No.
            You are provided with a question and an answer,
            and you need to find which option (Yes / No) is most similar to the answer.
            If the meaning of all options are significantly different from the answer, output Unknown.
            Your should output a single word among the following 3 choices: Yes, No, Unknown.
            Example 1:
            Question: Is the word in this image 'Hello'?
            Answer: The word in this image is 'Hello'.\nYour output: Yes
            Example 2:
            Question: Is the word in this image 'Hello'?
            Answer: The word in this image is not 'Hello'.\nYour output: No\n
            Now here are the target's Question and Answer:\n
        '''
    },
}

DATASET2DEFAULT_IMAGE_TOKEN = {
    'mmmu_val': ['<image 1>', '<image 2>', '<image 3>', '<image 4>', '<image 5>', '<image 6>', '<image 7>', '<image 8>'],
}
DATASET2DEFAULT_IMAGE_TOKEN.update(
    {d: ['<image>'] for d in datasets if d not in DATASET2DEFAULT_IMAGE_TOKEN.keys()}
)
