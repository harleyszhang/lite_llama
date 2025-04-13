import warnings
import string

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


from lite_llama_inference import LiteLlamaInference
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

from utils.eval import *


class EvaluatorAccuracy(object):
    def __init__(self, test_data_path, custom_checkpoints_dir, data_batch=10):
        self.custom_checkpoints_dir = custom_checkpoints_dir
        self.test_data_path = test_data_path
        self.data_batch = data_batch

        # init inference
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lite_llama_inference = LiteLlamaInference(
            temperature=0.7,
            top_p=0.8,
            max_seq_len=2048,
            max_gen_len=1900,
            lite_llama_ckpt_dir=self.custom_checkpoints_dir,
            device=self.device
        )
    def process(self,):

        if "hotpot" in self.test_data_path.lower():
            data_obj = HotpotQA(self.test_data_path, self.data_batch)

        elif "hellaswag" in self.test_data_path.lower():
            data_obj = HellaSwag(self.test_data_path, self.data_batch)

        try:
            assert data_obj is not None, "data_obj has not been created"
        except NameError:
            raise AssertionError("Dataset may not be supported")

        ground_truth, prompts, options = data_obj.parse_data()

        predictions = self.lite_llama_inference.process(prompts)

        if data_obj.data_type == "mcq":
            data_obj.evaluate(predictions, ground_truth, options)
        else:
            data_obj.evaluate(predictions, ground_truth)


if __name__ == '__main__':
    ea = EvaluatorAccuracy("/path_to/hotpot_dev_distractor_v1.json",
                           "/path_to/Llama-3.2-3B-Instruct")
    # ea = EvaluatorAccuracy("/path_to/hellaswag_val.jsonl",
    #                        "/path_to/Llama-3.2-3B-Instruct")
    ea.process()