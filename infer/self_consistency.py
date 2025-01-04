# This file includes functions adapted from the optillm repository (https://github.com/codelion/optillm).
# Original work by codelion et al., licensed under Apache License 2.0 license.
# Copyright (c) 2004 codelion

import torch
from infer.base import InferCenter
from difflib import SequenceMatcher

from typing import List, Dict
from difflib import SequenceMatcher


class InferSCCenter(InferCenter):
    """self consistency inference

    Args:
        model_wrapper: object of model wrapper
    """
    def __init__(self, model_wrapper, n_generate_sample=5, similarity_threshold=0.8, **kwargs):
        super().__init__(model_wrapper)
        self.n_generate_sample = n_generate_sample
        self.completion_tokens = 0
        self.similarity_threshold = similarity_threshold

    def generate_responses(self, data, device):
        responses = [self.infer_direct(data, device, force_use_generate=True) for _ in range(self.n_generate_sample)] # TODO: 传温度
        self.completion_tokens += sum(self.model_wrapper.get_tokenizer().tokenize(i) for i in responses)

        return responses

    def calculate_similarity(self, a: str, b: str):
        return SequenceMatcher(None, a, b).ratio()

    def cluster_similar_responses(self, responses: List[str]):
        clusters = []
        for response in responses:
            added_to_cluster = False
            for cluster in clusters:
                if self.calculate_similarity(response, cluster[0]) >= self.similarity_threshold:
                    cluster.append(response)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([response])
        return clusters

    def aggregate_results(self, responses: List[str]):
        clusters = self.cluster_similar_responses(responses)
        
        cluster_info = []
        for cluster in clusters:
            cluster_info.append({
                "answer": cluster[0],
                "frequency": len(cluster),
                "variants": cluster
            })
        
        cluster_info.sort(key=lambda x: x['frequency'], reverse=True)
        
        return {
            "clusters": cluster_info,
            "total_responses": len(responses),
            "num_unique_clusters": len(clusters)
        }

    def infer(self, data, device=torch.device('cuda'), **kwargs):
        responses = self.generate_responses(data, device)
        aggregated_result = self.aggregate_results(responses)
        
        return {
            "individual_responses": responses,
            "aggregated_result": aggregated_result
        }

infer_core = InferSCCenter
