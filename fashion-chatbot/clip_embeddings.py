# import torch
from typing import Any, Dict, List, Optional
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast


class ClipEmbeddings():

    def __init__(self, model_name, device):
        # text_config = CLIPTextConfig(max_position_embeddings=512)
        # vision_config = CLIPVisionConfig()
        # configuration = CLIPConfig.from_text_vision_configs(text_config, vision_config)
        self.model_name: str = model_name
        self.device: str = device
        # torch.cuda.empty_cache()
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt").input_ids
        text_emb = self.model.get_text_features(inputs.to(self.device))
        return text_emb.tolist()[0]
