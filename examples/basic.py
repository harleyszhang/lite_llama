"""Basic batch inference — mirrors vllm's ``examples/basic/basic.py``.

The script builds an :class:`~rapid_llm.engine.llm.LLM` over the
checkpoint named at the top, generates for a fixed prompt list with
greedy params, and prints the texts — the smallest honest smoke test.

Usage:
    python examples/basic.py   # set ``checkpoints_dir`` first
"""

from rapid_llm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "The future of artificial intelligence is",
    "In three sentences, explain quantum computing:",
    "Once upon a time, in a quiet village,",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_gen_len=64)


def main():
    llm = LLM(model="my_weight/Qwen2.5-0.5B")
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and finish reason.
    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        print(f"Prompt:    {output.prompt!r}")
        print(f"Output:    {output.text!r}  (finish: {output.outputs[0].finish_reason})")
        print("-" * 60)


if __name__ == "__main__":
    main()
