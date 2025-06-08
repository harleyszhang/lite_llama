import torch
from typing import Union, TextIO, Optional
import torch
from transformers import AutoModel, AutoConfig, PreTrainedModel
from accelerate import init_empty_weights

MODEL_ID = "/home/honggao/llm_weights/Qwen3-235B-A22B"


def print_empty_model(model_id):
	"""
	Accelerate 提供 init_empty_weights 上下文管理器，令所有 Parameter 和 Buffer 
	都放在 meta device，尺寸为 0，因此既 不下载权重 也 不占内存。
	"""
	cfg = AutoConfig.from_pretrained(model_id)   # 只拉配置

	with init_empty_weights(): 
		model = AutoModel.from_config(cfg)
		print(model)
	return model

def print_transformers_model_summary(
    model: PreTrainedModel,
    *,
    use_torchinfo: bool = False,
    input_size: Optional[tuple] = None,
    file: Union[str, TextIO, None] = None,
) -> None:
    """
    打印 Hugging Face Transformers 模型结构 + 权重 shape。
    
    Args:
        model (PreTrainedModel): 已加载好的模型实例。
        use_torchinfo (bool): 是否调用 torchinfo.summary() 生成额外摘要。
        input_size (tuple): 当 use_torchinfo=True 时需提供 (seq_len, ) or (bs, seq_len, ...)。
        file: None  -> 输出到 stdout；
              str   -> 输出到指定路径文件；
              TextIO -> 已打开的文件句柄。
    """
    import math

    def _human_readable(num: float, *, base: int = 1000, units=("", "K", "M", "G", "T", "P"), suffix=""):
        """Convert a large number to human‑readable form (e.g. 12.3G)."""
        if num == 0:
            return f"0{suffix}"
        exp = min(int(math.log(num, base)), len(units) - 1)
        value = num / (base ** exp)
        return f"{value:.2f}{units[exp]}{suffix}"

    def _dump(msg: str = ""):
        if fh:
            fh.write(msg + "\n")
        else:
            print(msg)

    # 0) 处理输出目标
    fh = open(file, "w") if isinstance(file, str) else file

    # 1) 模型 __repr__
    _dump("=" * 60)
    _dump("Model architecture (__repr__):")
    _dump("=" * 60)
    _dump(str(model))
    
    # 2) 权重 shape
    _dump("\n" + "=" * 60)
    _dump("Parameter shapes (name -> shape, #elements):")
    _dump("=" * 60)

    # Token count estimation for FLOPs (default = 1 token if unknown)
    tokens = 1
    if input_size is not None:
        # Accept (seq_len,), (bs, seq_len) or any shape where last dim is seq_len
        if len(input_size) == 1:
            tokens = input_size[0]
        else:
            tokens = input_size[0] * input_size[-1]

    total_params = 0
    total_flops = 0
    total_mem_bytes = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        # ---- Estimate per‑parameter FLOPs ----
        if param.dim() == 2:  # typical (out, in) weight matrix
            flops = 2 * param.shape[0] * param.shape[1] * tokens
        elif param.dim() == 1:  # bias / norm weight
            flops = param.shape[0] * tokens
        else:
            flops = numel  # fallback crude estimate
        total_flops += flops

        # ---- Memory access cost (parameter bytes only) ----
        mem_bytes = numel * param.element_size()
        total_mem_bytes += mem_bytes

        # ---- Pretty print ----
        flops_str = _human_readable(flops, suffix="F")
        mem_str = _human_readable(mem_bytes, base=1024, units=("B","KB","MB","GB","TB","PB"))
        _dump(f"{name:<60} {str(tuple(param.shape)):<20} {numel:,}  |  {flops_str:<8}  |  {mem_str}")

    _dump(f"\nTotal parameters: {total_params:,}")
    _dump(f"Estimated forward FLOPs: {_human_readable(total_flops, suffix='F')}")
    _dump(f"Parameter memory: {_human_readable(total_mem_bytes, base=1024, units=('B','KB','MB','GB','TB','PB'))}")

    # 3) 可选 torchinfo 摘要
    if use_torchinfo:
        try:
            from torchinfo import summary  # pip install torchinfo
            assert input_size is not None, "`input_size` must be provided when use_torchinfo=True"
            info = summary(
                model,
                input_size=input_size,
                depth=3,
                col_names=("kernel_size", "output_size", "num_params", "mult_adds"),
                dtypes=[torch.long],  # 对 NLP 模型输入通常是 int64 token id
            )
            _dump("\n" + "=" * 60)
            _dump("torchinfo summary():")
            _dump("=" * 60)
            _dump(str(info))
        except ImportError:
            _dump("torchinfo 未安装，跳过摘要。pip install torchinfo 获取更丰富视图。")
    
    if isinstance(file, str):  # 自动关闭文件
        fh.close()

from torchviz import make_dot  # pip install torchviz graphviz
def save_model_graph(model, input_example: torch.Tensor, file_name: str = "model_graph.svg") -> None:
    """
    利用 torchviz 生成前向图；input_example 必须能直接送入 model。
    """
    model.eval()
    y = model(input_example)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = file_name.split(".")[-1]  # 自动根据后缀决定 svg/png
    dot.render(file_name, cleanup=True)
    print(f"✅ Graph saved to {file_name}")

if __name__ == "__main__":
    # model = AutoModel.from_pretrained(MODEL_ID)
    model =  print_empty_model(MODEL_ID)
    input_example = torch.randint(0, 1000, (2, 2048))  # 随机输入
    print_transformers_model_summary(
        model=model,
        use_torchinfo=True,
        input_size=(2, 2048),
        file="qwen3_8b_structure.txt"
    )