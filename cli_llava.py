import torch
from typing import Optional
from lite_llama.llava_generate_stream import LlavaGeneratorStream
from lite_llama.utils.image_process import vis_images
from lite_llama.utils.prompt_templates import get_prompter, get_image_token
from rich.console import Console
from rich.prompt import Prompt
import sys,os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

# 模型检查点目录，请根据实际情况修改
checkpoints_dir = "/gemini/code/lite_llama/my_weight/llava-1.5-7b-hf"

def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gpu_num_blocks = None,
    max_gen_len: Optional[int] = 512,
    load_model: bool = True,
    compiled_model: bool = False,
    triton_weight: bool = True
):
    """
    主函数，处理用户输入并生成响应。

    Args:
        temperature (float, optional): 生成文本的温度。默认值为 0.6。
        top_p (float, optional): 生成文本的top-p值。默认值为 0.9。
        max_seq_len (int, optional): 最大序列长度。默认值为 2048。
        max_gpu_num_blocks: 用户自行设置的最大可用 blocks(tokens), 如果设置该值， kv cache 内存管理器的最大可用内存-tokens 由该值决定。
        max_gen_len (Optional[int], optional): 生成文本的最大长度。默认值为 512。
        load_model (bool, optional): 是否加载模型。默认值为True。
        compiled_model (bool, optional): 是否使用编译模型。默认值为True。
        triton_weight (bool, optional): 是否使用Triton权重。默认值为True。
    """
    console = Console()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False

    model_prompter = get_prompter("llama", checkpoints_dir, short_prompt)

    # 初始化多模态模型文本生成器
    try:
        generator = LlavaGeneratorStream(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            load_model=load_model,
            compiled_model=compiled_model,
            triton_weight=triton_weight,
            device=device,
        )
    except Exception as e:
        console.print(f"[red]模型加载失败: {e}[/red]")
        sys.exit(1)

    while True:
        console.print("[bold green]请输入图片路径或URL (输入 'exit' 退出）：[/bold green]") # 获取用户输入的图片路径或URL
        while True: # 循环判断输入图像路径是否成功, 成功则跳出循环
            image_input = Prompt.ask("图片")
            if os.path.isfile(image_input):
                break
            elif image_input.strip().lower() == 'exit':
                break
            else:
                print(f"错误：'{image_input}' 不是有效的文件路径！")
                image_input = Prompt.ask("图片")

        image_input = image_input.strip()
        if image_input.lower() == 'exit':
            break
        
        image_items = [image_input] # 准备image_items列表
        image_num = len(image_items) # 计算输入图片数量
        vis_images(image_items) # 在终端中显示图片

        # console.print("\n[bold blue]请输入提示词（输入 'exit' 退出）：[/bold blue]") # 获取用户的提示词
        input_prompt = Prompt.ask("[bold green]提示词[/bold green]").strip()
        if input_prompt.lower() == 'exit':
            break

        image_token = get_image_token()
        model_prompter.insert_prompt(image_token * image_num + input_prompt)

        # prompts = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        prompts = [model_prompter.model_input] # 准备提示词，替换<image>标记

        # 调用生成器生成文本
        try:
            stream = generator.text_completion_stream(
                prompts,
                image_items,
                temperature=temperature,
                top_p=top_p,
                max_gen_len=max_gen_len,
            )
        except Exception as e:
            console.print(f"[red]文本生成失败: {e}[/red]")
            continue
        
        completion = ''  # 初始化生成结果
        console.print("ASSISTANT: ", end='')
        
        for batch_completions in stream:
            next_text = batch_completions[0]['generation'][len(completion):]
            completion = batch_completions[0]['generation']
            print(f"\033[91m{next_text}\033[0m", end='', flush=True)  # 红色文本
        
        console.print("\n[bold green]==================================[/bold green]\n")
    
if __name__ == "__main__":
    main()