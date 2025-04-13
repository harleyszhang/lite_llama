import random
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.common import *
from typing import List, Optional, Any
import string, re
import torch
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
class HotpotQA(object):
    r"""
        for testing hotpot wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    """
    def __init__(self, data_path, data_batch=None):
        self.data_type = "qa"
        self.data_path = data_path

        # data_batch=none means testing all the data in the dataset
        self.data_batch = data_batch

    def extract_supporting_context(self, data: dict):
        context_dict = dict(data['context'])
        supporting_facts = data['supporting_facts']

        support_text = []
        for title, sent_idx in supporting_facts:
            if title in context_dict:
                sentences = context_dict[title]
                if sent_idx < len(sentences):
                    support_text.append(sentences[sent_idx])
        return '\n'.join(support_text)

    def build_prompt(self, data: dict) -> str:
        context = self.extract_supporting_context(data)
        question = data['question']
        prompt = f"""
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""
        return prompt


    def parse_data(self) -> tuple[Any, Any, list[Any]]:
        data = read_json(self.data_path)
        test_data = list()
        for hotpot_index, hotpot_content in enumerate(data):
            data_index = hotpot_content["_id"]
            prompt = self.build_prompt(hotpot_content)
            answer = hotpot_content['answer'].strip().lower()

            test_data.append({data_index: {"prompt": prompt, "answer": answer}})

        if self.data_batch is None:
            self.data_batch = len(test_data)

        return unify_data(test_data, self.data_batch, "qa")

    def evaluate(self, predictions, ground_truth):

        assert len(predictions) == len(ground_truth), "Prediction and Ground Truth list must be the same length."

        total_em = 0.0
        total_f1 = 0.0
        total_jaccard = 0.0
        total_embed_sim = 0.0
        n = len(predictions)

        for pred, gt in zip(predictions, ground_truth):
            total_em += exact_match(pred, gt)
            total_f1 += penalized_f1(pred, gt)
            total_jaccard += jaccard_similarity(pred, gt)
            total_embed_sim += embedding_similarity(pred, gt)

        scores = {
            "EM": total_em / n,
            "F1 (penalized)": total_f1 / n,
            "Jaccard": total_jaccard / n,
            "Embedding Sim": total_embed_sim / n
        }

        print(f"The test result of lite_llama inference for {self.data_type} dataset: {scores}")


class HellaSwag(object):
    r"""
        for testing HellaSwag wget https://raw.githubusercontent.com/rowanz/hellaswag/refs/heads/master/data/hellaswag_val.jsonl
    """

    def __init__(self, data_path, data_batch=None):
        self.data_path = data_path
        self.data_type = "mcq"
        self.choices = ['A', 'B', 'C', 'D']

        # data_batch=none means testing all the data in the dataset
        self.data_batch = data_batch

    def format_prompt(self, ctx, endings):
        prompt = f"Context: {ctx}\n\nWhich of the following is the most plausible continuation?\n"
        for letter, end in zip(self.choices, endings):
            prompt += f"{letter}) {end.strip()}\n"
        prompt += "\nAnswer:"
        return prompt

    def extract_choice(self, output_text):
        for letter in ['A', 'B', 'C', 'D']:
            if letter in output_text:
                return ['A', 'B', 'C', 'D'].index(letter)
        return -1

    def convert_answer(self, answer) -> str:
        return self.choices[int(answer)]

    def parse_data(self) -> tuple[Any, Any, list[Any]]:

        data = read_jsonl(self.data_path)
        test_data = list()
        for index, content in enumerate(data):
            prompt = self.format_prompt(content["ctx"], content["endings"])
            answer = self.convert_answer(content["label"])

            option = [
                ("A", content["endings"][0]),
                ("B", content["endings"][1]),
                ("C", content["endings"][2]),
                ("D", content["endings"][3]),
            ]

            test_data.append({index: {"prompt": prompt, "answer": answer, "options": option}})

        if self.data_batch is None:
            self.data_batch = len(test_data)
        return unify_data(test_data, self.data_batch, self.data_type)

    def evaluate(self, predictions, ground_truth, options):

        assert len(predictions) == len(ground_truth), "Prediction and Ground Truth list must be the same length."

        total_em = 0.0
        total_f1 = 0.0
        total_jaccard = 0.0
        total_embed_sim = 0.0
        n = len(predictions)

        for pred, gt, op in zip(predictions, ground_truth, options):
            matched_option = extract_final_choice(pred)
            if not matched_option:
                matched_option, similarities = match_mc_option(pred, op)

            pred_ = str(matched_option)

            total_em += exact_match(pred_, gt)
            total_f1 += penalized_f1(pred_, gt)
            total_jaccard += jaccard_similarity(pred_, gt)
            total_embed_sim += embedding_similarity(pred_, gt)

        scores = {
            "EM": total_em / n,
            "F1 (penalized)": total_f1 / n,
            "Jaccard": total_jaccard / n,
            "Embedding Sim": total_embed_sim / n
        }

        print(f"The test result of lite_llama inference for {self.data_type} dataset: {scores}")


def matched_pairs(list1, list2, n):
    """
    Randomly sample n matched pairs from two lists with aligned indices.

    Args:
        list1 (list): The first list.
        list2 (list): The second list. Must have the same length as list1.
        n (int): The number of samples to draw.

    Returns:
        tuple: Two lists containing the sampled elements from list1 and list2,
               where the elements at each index still match.
    """
    assert len(list1) == len(list2), "Both lists must have the same length"
    assert n <= len(list1), "n must not be greater than the length of the lists"

    indices = random.sample(range(len(list1)), n)
    sampled_list1 = [list1[i] for i in indices]
    sampled_list2 = [list2[i] for i in indices]
    return sampled_list1, sampled_list2


def unify_data(test_data, data_batch, data_type: Optional[str]):
    ground_truth, prompts, options = list(), list(), list()

    for index, data in enumerate(test_data):
        key = next(iter(data))

        ground_truth.append(data[key]['answer'])
        prompts.append(data[key]['prompt'])
        if data_type == "mcq":
            options.append(data[key]['options'])
    ground_truth, prompts = matched_pairs(ground_truth, prompts, data_batch)

    return ground_truth, prompts, options



def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def lower(text):
        return text.lower()

    def remove_consecutive_duplicates(text):
        words = text.split()
        result = [words[0]] if words else []

        for i in range(1, len(words)):
            if words[i] != words[i - 1]:
                result.append(words[i])
        return ' '.join(result)

    return remove_consecutive_duplicates(white_space_fix(remove_articles(remove_punc(lower(s)))))


def exact_match(pred, gt):
    return normalize_answer(pred) == normalize_answer(gt)


def penalized_f1(prediction, ground_truth, max_len_ratio=3, penalty_factor=0.5):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    if len(pred_tokens) > len(gt_tokens) * max_len_ratio:
        f1 *= penalty_factor

    return f1



def jaccard_similarity(prediction, ground_truth):
    pred_tokens = set(normalize_answer(prediction).split())
    gt_tokens = set(normalize_answer(ground_truth).split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    intersection = pred_tokens & gt_tokens
    union = pred_tokens | gt_tokens
    return len(intersection) / len(union)

def embedding_similarity(prediction, ground_truth):
    embeddings = embedding_model.encode([prediction, ground_truth], convert_to_tensor=True)
    sim_score = util.cos_sim(embeddings[0], embeddings[1])
    return sim_score.item()


def extract_final_choice(text: str) -> Any | None:
    """
    Extracts the final multiple-choice answer (a/b/c/d) from long-form LLM output.
    Returns the answer in lowercase (e.g., 'a', 'b', 'c', 'd'), or None if not found.
    """
    # Normalize text for consistent matching
    text = normalize_answer(text)
    # Priority: explicit natural language conclusion
    # Pattern 1: match "answer: a", "correct answer is: b", etc.
    patterns = [
        r'answer\s*[:\-]?\s*([a-dA-D])\b',
        r'option\s*([a-dA-D])\b',
        r'\b([a-dA-D])\b\s+is\s+(correct|the answer)',
        r'\b([a-dA-D])[\).]',
        r'choice\s*[:\-]?\s*([a-dA-D])\b'
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()


    return None


def match_mc_option(prediction, options):
    """
    Match the free-text prediction to the closest option (A, B, C, D) by semantic similarity
    """
    # Convert all options into a list of strings
    prediction = normalize_answer(prediction)
    option_texts = [text for _, text in options]

    # Encode all texts
    pred_emb = embedding_model.encode(prediction, convert_to_tensor=True)
    option_embs = embedding_model.encode(option_texts, convert_to_tensor=True)

    # Compute cosine similarity
    cos_sims = util.cos_sim(pred_emb, option_embs)[0]  # shape: (4,)
    best_idx = int(torch.argmax(cos_sims).item())
    return options[best_idx][0], cos_sims.tolist()  # Returns the matching option ID and all similarities

if __name__ == '__main__':


    hw = HellaSwag("/path_to/hellaswag_val.jsonl")
    hw.process()