import transformers
from typing import Any, Dict, List, Optional
from torch import cuda, bfloat16, backends

DEFAULT_MODEL_ID = 'meta-llama/Llama-2-13b-chat-hf'
DEFAULT_DEVICE = "cpu"
DEFAULT_HF_AUTH = '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'

class LlamaLLM():

    def __init__(self, **kwargs: Any):
        self.device: str = DEFAULT_DEVICE
        self.model_id: str = DEFAULT_MODEL_ID
        self.hf_auth:str = DEFAULT_HF_AUTH

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        # begin initializing HF items, need auth token for these

        model_config = transformers.AutoConfig.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=self.hf_auth
        )
        model.eval()
        print(f"Model loaded on {self.device}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_auth
        )

        self.pipeline = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )

    def generate(self, text: str):
        return self.pipeline(text)