# Copyright (C) 2024 AIDC-AI
import os
_DEFAULT_MAX_LENGTH = 2048

MAX_GEN_TOKS = 1024


MODEL_PATH = 'your_models_path/models'
DATA_PATH = './data'
OUTPUT_PATH = 'your_outputs_path/outputs'

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
    'multimodal_llm.lamda_llm.tabular': ['Tabular'],
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
    'llm.microsoft.gtl_delta': ['LLaMA-2-7b-GTL-Delta'],
    'llm.test_llm': ['TestLLM'],
    'llm.qwen.deepseek_r1_distill_qwen': ['DeepSeek-R1-Distill-Qwen-7B']
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
    'drop': ['drop'],
    'humaneval': ['humaneval-prompt']
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

DATASET2DEFAULT_IMAGE_TOKEN = {
    'mmmu_val': ['<image 1>', '<image 2>', '<image 3>', '<image 4>', '<image 5>', '<image 6>', '<image 7>', '<image 8>'],
}
DATASET2DEFAULT_IMAGE_TOKEN.update(
    {d: ['<image>'] for d in datasets if d not in DATASET2DEFAULT_IMAGE_TOKEN.keys()}
)
