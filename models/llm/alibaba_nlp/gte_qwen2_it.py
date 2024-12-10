from sentence_transformers import SentenceTransformer

from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base import ModelWrapper
from configs import DATA_PATH

import os

class GTEQwen2(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args):
        super().__init__()
        self.model = SentenceTransformer(model_path, **{k: v for k, v in model_args.items() if k != 'documents_file'})
        self.model.max_seq_length = 8192
        with open(os.path.join(DATA_PATH, model_args['documents_file']), 'r', encoding='utf-8') as f:
            themes = [i.strip() for i in f.readlines()]
        self.themes_embeddings = self.model.encode(themes)

    def generate_text_only(self, conversations, device, **kwargs):
        queries = [i['prompt_instruction'] for i in conversations]

        query_embeddings = self.model.encode(queries, prompt_name="query")

        scores = (query_embeddings @ self.themes_embeddings.T) * 100
        return {
            i['id']: i_score for i, i_score in zip(conversations, scores.tolist())
        }

model_core = GTEQwen2
