export multimodal_datasets='ai2d_test,ccbench,hallusionbench,mathvista_mini,mmbench_dev_cn,mmbench_dev_en,mme,mmmu_val,mmstar,mmvet,ocrbench,realworldqa,scienceqa_test,scienceqa_val,seedbench_img'
export text_only__datasets='arc_challenge_25,arc_easy_25,boolq,cmmlu,mmlu_5,openbookqa,anli,cola,copa,glue,hellaswag_10,aclue,piqa,truthfulqa_mc2,mathqa,hellaswag_multilingual_sampled,xstorycloze_sampled,eq_bench,bbh_3,logieval,drop'

export text_only__datasets='demo_text_only_generate,demo_text_only_multiple_choice'
export multimodal_datasets='demo_vqa_generate'

CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model Phi-3.5-vision-instruct --model_args device_map=cuda,trust_remote_code=True,torch_dtype=auto,_attn_implementation=flash_attention_2 --tokenizer_args trust_remote_code=True,num_crops=4 --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model MiniCPM-V-2_6 --model_args device_map=cuda,trust_remote_code=True,torch_dtype=torch.float16 --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Meta-Llama-3-8B-Instruct --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Meta-Llama-3.1-8B-Instruct --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Baichuan-7B --model_args device_map=auto,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Baichuan2-7B-Chat --model_args device_map=auto,torch_dtype=torch.bfloat16,trust_remote_code=True --tokenizer_args use_fast=False,trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model Bunny-Llama-3-8B-V --model_args torch_dtype=torch.float16,device_map=auto,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model MetaModel_moex8 --model_args torch_dtype=torch.float16,load_in_4bit=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model MiniCPM-Llama3-V-2_5 --model_args device_map=cuda,trust_remote_code=True,torch_dtype=torch.float16 --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model MiniCPM-V --model_args trust_remote_code=True,torch_dtype=torch.bfloat16 --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model MiniCPM-V-2 --model_args device_map=cuda,trust_remote_code=True,torch_dtype=torch.bfloat16 --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model MiniGPT-4 --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Mistral-7B-OpenOrca --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model OmniLMM-12B --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Orca-2-7b --model_args device_map=auto --tokenizer_args use_fast=False --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Phi-3-mini-128k-instruct --model_args torch_dtype=auto,device_map=cuda,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Phi-3-mini-4k-instruct --model_args torch_dtype=auto,device_map=cuda,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Phi-3-small-8k-instruct --model_args torch_dtype=auto,device_map=cuda,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Phi-3-small-128k-instruct --model_args torch_dtype=auto,device_map=cuda,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model Phi-3-vision-128k-instruct --model_args device_map=cuda,trust_remote_code=True,torch_dtype=auto,_attn_implementation=flash_attention_2 --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen-1_8B --model_args device_map=auto,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen-1_8B-Chat --model_args device_map=auto,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen-7B --model_args device_map=auto,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen-7B-Chat --model_args device_map=auto,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model Qwen-VL --model_args device_map=cuda,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model Qwen-VL-Chat --model_args device_map=cuda,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-0.5B --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-0.5B-Chat --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-1.8B --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-1.8B-Chat --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-4B --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-4B-Chat --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-7B --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-7B-Chat --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-MoE-A2.7B --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen1.5-MoE-A2.7B-Chat --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen2-0.5B --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen2-0.5B-Instruct --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen2-1.5B --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen2-1.5B-Instruct --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen2-7B --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen2-7B-Instruct --model_args device_map=auto,torch_dtype=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model RedPajama-INCITE-Chat-3B-v1 --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Yi-1.5-6B --model_args device_map=auto,torch_dtype=auto --tokenizer_args use_fast=False --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Yi-1.5-6B-Chat --model_args device_map=auto,torch_dtype=auto --tokenizer_args use_fast=False --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Yi-1.5-9B --model_args device_map=auto,torch_dtype=auto --tokenizer_args use_fast=False --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Yi-1.5-9B-Chat --model_args device_map=auto,torch_dtype=auto --tokenizer_args use_fast=False --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Yi-6B --model_args device_map=auto,torch_dtype=auto --tokenizer_args use_fast=False --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Yi-6B-Chat --model_args device_map=auto,torch_dtype=auto --tokenizer_args use_fast=False --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Yi-9B --model_args device_map=auto,torch_dtype=auto --tokenizer_args use_fast=False --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model Yi-VL-6B --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model bloom-3b --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model bloom-7b1 --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model bloomz-3b --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model bloomz-7b1 --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model chatglm-6b --model_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model chatglm2-6b --model_args trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model chatglm3-6b --model_args trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model chatglm3-6b-base --model_args trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model cogvlm-chat-hf --model_args device_map=cuda,torch_dtype=torch.bfloat16,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model cogvlm-chat-hf --model_args torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True --tokenizer_args tokenizer_name=vicuna-7b-v1.5 --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model cogvlm2-llama3-chat-19B --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model deepseek-llm-7b-base --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model deepseek-llm-7b-chat --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model deepseek-moe-16b-chat --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model deepseek-vl-1.3b-base --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model deepseek-vl-1.3b-chat --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model deepseek-vl-7b-base --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model deepseek-vl-7b-chat --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model glm-10b --model_args trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model glm-4-9b --model_args torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model glm-4-9b-chat --model_args torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model glm-4v-9b --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model idefics2-8b --model_args device_map=cuda --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model internlm-chat-7b --model_args torch_dtype=torch.float16,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model internlm-xcomposer2-7b --model_args torch_dtype=torch.float32,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model internlm-xcomposer2-vl-1_8b --model_args trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model internlm-xcomposer2-vl-7b --model_args trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model internlm2-1_8b --model_args torch_dtype=torch.float16,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model internlm2-7b --model_args torch_dtype=torch.float16,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model internlm2-chat-1_8b --model_args torch_dtype=torch.float16,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model internlm2-chat-7b --model_args torch_dtype=torch.float16,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model internlm2_5-7b --model_args torch_dtype=torch.float16,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model internlm2_5-7b-chat --model_args torch_dtype=torch.float16,trust_remote_code=True --tokenizer_args trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model llava-v1.5-7b --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model llava-v1.5-7b-lora --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model llava-v1.6-mistral-7b --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model llava-v1.6-vicuna-7b --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model mistral-7b-sft-alpha --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model mistral-7b-sft-beta --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model phi-1 --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model phi-1_5 --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model vicuna-7b-v1.5 --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model zephyr-7b-alpha --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model zephyr-7b-beta --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model zephyr-7b-gemma-sft-v0.1 --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model zephyr-7b-gemma-v0.1 --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model Ovis-Clip-Qwen1_5-7B --model_args torch_dtype=torch.bfloat16,multimodal_max_length=8192,trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model Ovis-Clip-Llama3-8B --model_args torch_dtype=torch.bfloat16,multimodal_max_length=8192,trust_remote_code=True --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $text_only__datasets --model Qwen2-72B-Instruct --model_args torch_dtype=torch.bfloat16,device_map=auto --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data $multimodal_datasets --model Wings --model_args trust_remote_code=True --model_args json_path=path_of_args.json --time_str 03_26_00_00_00
CUDA_VISIBLE_DEVICES="0" python main.py --data bbh_3 --model TestLLM --infer_type direct --infer_args temp=ok --time_str 03_26_00_00_00
