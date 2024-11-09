from typing import List, Optional
from generate import GenerateText, Dialog
import torch
from torch.profiler import profile, ProfilerActivity

def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = 64,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator = GenerateText.build(
        checkpoints_dir='/gemini/code/Llama-3.2-1B-Instruct/my_weight/',
        tokenizer_path='/gemini/code/Llama-3.2-1B-Instruct/',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=max_batch_size,
        device=device,
        triton_weight=True,
        compiled_model=True
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        "Roosevelt was the first president of the United States, he has",
    ]
    
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        main()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))