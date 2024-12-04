import torch
from typing import Optional
from lite_llama.llava_generate_stream import LlavaGeneratorStream
from lite_llama.utils.image_process import vis_images
from lite_llama.utils.prompt_templates import get_prompter, get_image_token
from rich.console import Console
from rich.prompt import Prompt
import sys,os

# 模型检查点目录，请根据实际情况修改
checkpoints_dir = "/gemini/code/lite_llama/my_weight/llava-1.5-7b-hf"

def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,  # 每次处理一个 Prompt
    max_gen_len: Optional[int] = 512,
    load_model: bool = True,
    compiled_model: bool = True,
    triton_weight: bool = True
):
    """
    主函数，处理用户输入并生成响应。

    Args:
        temperature (float, optional): 生成文本的温度。默认值为 0.6。
        top_p (float, optional): 生成文本的top-p值。默认值为 0.9。
        max_seq_len (int, optional): 最大序列长度。默认值为 2048。
        max_batch_size (int, optional): 最大批次大小。默认值为 1。
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
            max_batch_size=max_batch_size,
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
        console.print("\n[bold green]请输入图片路径或URL (输入 'exit' 退出）：[/bold green]") # 获取用户输入的图片路径或URL
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

        console.print("\n[bold blue]请输入提示词（输入 'exit' 退出）：[/bold blue]") # 获取用户的提示词
        input_prompt = Prompt.ask("提示词")
        if input_prompt.lower() == 'exit':
            break

        image_token = get_image_token()
        model_prompter.insert_prompt(image_token * image_num + input_prompt)

        # prompts = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        prompts = [model_prompter.model_input] # 准备提示词，替换<image>标记
        print("prompts ", prompts)
        print("image_items ", image_items)

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
        console.print("[bold yellow]助手正在生成响应...[/bold yellow]")
        console.print("ASSISTANT: ", end='')
        # 迭代生成的文本流
        try:
            for batch_completions in stream:
                for completion_dict in batch_completions:
                    next_text = completion_dict['generation'][len(completion):] # [len(completion):] 提取新生成的文本部分（自上次生成以来新增的部分）。
                    completion = completion_dict['generation']
                    console.print(next_text, end='')
                console.print("\n\n[bold green]==================================[/bold green]\n")
        
        except KeyboardInterrupt:
            console.print("\n[red]用户中断操作。[/red]")
            sys.exit(0)
        
        except Exception as e:
            console.print(f"\n[red]生成过程中出错: {e}[/red]")
            continue

if __name__ == "__main__":
    main()