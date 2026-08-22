"""``lite-llama`` command-line entry point.

Built bottom-up in four decoupled layers: a declarative option table
(:class:`CliOption` + ``COMMON_OPTIONS``) shared by every subcommand; a frozen
:class:`EngineOptions` that validates the parsed args and builds the right
``TextGenerator`` / ``VisionGenerator``; a :class:`PrompterResolver` that picks
base-vs-instruct prompting; and :class:`CliCommand` subclasses that wire it up.
Adding a subcommand = one ``CliCommand`` subclass listed in ``COMMANDS``.

Usage:
    lite-llama chat --model-dir my_weight/Qwen2.5-0.5B
    lite-llama vl-chat --model-dir my_weight/llava-1.5-7b-hf
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from .engine import SamplingParams, TextGenerator, VisionGenerator
from .models.config import read_model_type
from .utils.prompt_templates import ChatPrompter, get_prompter


# --------------------------------------------------------------------------- #
# TP worker (module-level so mp can pickle it)
# --------------------------------------------------------------------------- #
def _tp_mirror_worker(
    rank: int,
    world_size: int,
    model_dir: str,
    max_seq_len: int,
    max_gpu_num_blocks: int | None,
    quantization: str | None,
    image_paths: list[str] | None = None,
    prompt_text: str | None = None,
) -> None:
    """Non-rank-0 TP worker: builds model, then mirrors rank 0's forwards via NCCL.

    The worker sits in a loop: it receives a flag (1=forward, 0=exit) and the
    prompt tokens from rank 0 via dist.broadcast, then calls generator.stream()
    which participates in the same all-reduces as rank 0. Output is discarded.

    For multimodal models, ``image_paths`` and ``prompt_text`` let the worker
    run the same vision-tower + language-model forward calls as rank 0.
    """
    import torch
    import torch.distributed as dist

    from .distributed.parallel_state import init_tensor_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(rank)
    init_tensor_parallel(rank=rank, world_size=world_size)

    if image_paths:
        from .engine import VisionGenerator
        generator = VisionGenerator(
            checkpoints_dir=model_dir,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=f"cuda:{rank}",
            quantization=quantization,
            tensor_parallel_size=world_size,
        )
    else:
        generator = TextGenerator(
            checkpoints_dir=model_dir,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=f"cuda:{rank}",
            use_cuda_graph=False,
            quantization=quantization,
            tensor_parallel_size=world_size,
        )
    params = SamplingParams(temperature=0.0, max_gen_len=4096)

    # Mirror loop: wait for rank 0's broadcast, generate, discard output.
    while True:
        flag = torch.zeros(1, dtype=torch.int64, device=f"cuda:{rank}")
        dist.broadcast(flag, src=0)
        if flag.item() == 0:
            break
        length = torch.zeros(1, dtype=torch.int64, device=f"cuda:{rank}")
        dist.broadcast(length, src=0)
        tok_tensor = torch.zeros(length.item(), dtype=torch.int64, device=f"cuda:{rank}")
        dist.broadcast(tok_tensor, src=0)

        prompt = generator.engine.tokenizer.decode(tok_tensor.tolist())
        if image_paths:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            for _ in generator.stream(prompt_text or prompt, images, params):
                pass
        else:
            for _ in generator.stream([prompt], params):
                pass  # participate in all-reduces, discard output

    dist.destroy_process_group()

# ---------------------------------------------------------------------------
# 第一层:声明式 CLI 参数表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliOption:
    """一条 CLI 参数声明;``register`` 将其翻译成一次 ``add_argument`` 调用。"""

    flag: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def register(self, sub: argparse.ArgumentParser) -> None:
        sub.add_argument(self.flag, **self.kwargs)


# 所有子命令共享的参数。repetition_penalty 默认 1.1 而非 1.0:小参数 base
# 模型在 fp16 argmax 平局(~0.02 logit gap)下极易滑入重复死循环,默认
# 开启轻量惩罚是更安全的出厂行为,传 1.0 可显式关闭。
COMMON_OPTIONS: tuple[CliOption, ...] = (
    CliOption("--model-dir", {"help": "Checkpoint directory"}),
    CliOption("--max-seq-len", {"type": int, "default": 2048}),
    CliOption("--max-gpu-num-blocks", {"type": int, "default": None}),
    CliOption("--device", {"default": "cuda"}),
    CliOption("--temperature", {"type": float, "default": 0.6}),
    CliOption("--top-p", {"type": float, "default": 0.9}),
    CliOption("--max-gen-len", {"type": int, "default": None}),
    CliOption(
        "--repetition-penalty",
        {
            "type": float,
            "default": 1.1,
            "help": "Penalise logits of already-generated tokens (1.0 disables it; "
            "1.1 is the default because small base models easily loop on repeats)",
        },
    ),
    CliOption(
        "--quantization",
        {
            "choices": ["int8"],
            "default": None,
            "help": "Runtime weight quantisation for fp16 checkpoints "
            "(fp8 checkpoints are detected automatically from config.json)",
        },
    ),
    CliOption(
        "--tensor-parallel-size",
        {
            "type": int,
            "default": 1,
            "help": "Number of GPUs for tensor parallelism (splits weights across cards)",
        },
    ),
)


# ---------------------------------------------------------------------------
# 第二层:引擎构造参数(配置对象 + Builder)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineOptions:
    """引擎构造参数;统一从 ``argparse.Namespace`` 显式提取、校验并持有。

    充当 TextGenerator / VisionGenerator 的工厂:两条生成路径的构造签名
    差异(如 CUDA Graph 只接入文本 decode 路径)在此消化,命令类只见
    统一的 ``build_*`` 接口。
    """

    model_dir: str
    max_seq_len: int = 2048
    max_gpu_num_blocks: int | None = None
    device: str = "cuda"
    use_cuda_graph: bool = False
    quantization: str | None = None
    tensor_parallel_size: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EngineOptions":
        # --model-dir 优先,其次环境变量 LITE_LLAMA_MODEL_DIR
        model_dir = args.model_dir or os.environ.get("LITE_LLAMA_MODEL_DIR")
        if not model_dir:
            raise SystemExit(
                "Model directory not provided. "
                "Pass --model-dir <path> or set LITE_LLAMA_MODEL_DIR."
            )
        if not Path(model_dir).is_dir():
            raise SystemExit(f"model directory {model_dir!r} does not exist")
        return cls(
            model_dir=model_dir,
            max_seq_len=args.max_seq_len,
            max_gpu_num_blocks=args.max_gpu_num_blocks,
            device=args.device,
            use_cuda_graph=getattr(args, "use_cuda_graph", False),
            quantization=getattr(args, "quantization", None),
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        )

    def build_text_generator(self) -> TextGenerator:
        return TextGenerator(
            checkpoints_dir=self.model_dir,
            max_seq_len=self.max_seq_len,
            max_gpu_num_blocks=self.max_gpu_num_blocks,
            device=self.device,
            use_cuda_graph=self.use_cuda_graph,
            quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size,
        )

    def build_vision_generator(self) -> VisionGenerator:
        # CUDA Graph capture 只接了文本 decode 路径,vl-chat 不传该参数
        return VisionGenerator(
            checkpoints_dir=self.model_dir,
            max_seq_len=self.max_seq_len,
            max_gpu_num_blocks=self.max_gpu_num_blocks,
            device=self.device,
            quantization=self.quantization,
            tensor_parallel_size=self.tensor_parallel_size,
        )


# ---------------------------------------------------------------------------
# 第三层:Prompter 选择策略
# ---------------------------------------------------------------------------


class PrompterResolver:
    """聊天模板选择策略:决定 checkpoint 该套哪个 prompter(或原样直传)。

    背景:给 *base* 模型套 chat 模板是有害的——base Qwen2.5-0.5B 收到
    ``<|im_start|>assistant`` 会回显 "Assistant" 然后退化成重复;反之给
    *chat* 模型喂裸 prompt 同样糟糕。因此必须可靠区分两类 checkpoint。
    """

    # 名称中出现即判定为 instruct/chat 模型的标记
    _INSTRUCT_NAME_HINTS: ClassVar[tuple[str, ...]] = ("instruct", "chat", "-it")
    # 默认即 chat 模型的家族(config.json 的 model_type)
    _CHAT_BY_DEFAULT_TYPES: ClassVar[tuple[str, ...]] = ("qwen3", "qwen3_vl")
    # model_type -> prompter 注册名;Qwen2/Qwen3 同为 ChatML 模板,共用 "qwen2"
    _PROMPTER_BY_TYPE: ClassVar[dict[str, str]] = {"qwen2": "qwen2", "qwen3": "qwen2"}
    # config.json 无法表达 LLaMA 家族的细分模板(vicuna / llama3 / llava),
    # 只能退回按路径名匹配
    _NAME_CANDIDATES: ClassVar[tuple[str, ...]] = ("qwen2", "qwen3", "llama3", "llama")

    @staticmethod
    def read_model_type(model_dir: str) -> str:
        """从 checkpoint 的 config.json 读 ``model_type``;读不到返回 ``""``。

        读取本身委托 :func:`lite_llama.models.config.read_model_type`(config SSOT);
        CLI 侧只对缺失/损坏的配置做容错,降级为按目录名推断模板。
        """
        try:
            return read_model_type(model_dir)
        except (OSError, ValueError):
            return ""

    @classmethod
    def is_instruct(cls, model_dir: str) -> bool:
        """该 checkpoint 是否经指令微调、适配 chat 模板。规则按序生效:

        1. 名称显式带 ``base``(如 Qwen3-0.6B-Base)→ 不是 instruct;
        2. 名称带 ``instruct`` / ``chat`` / ``-it`` → 是(LLaMA、Qwen2.5);
        3. Qwen3 家族默认就是 chat 模型,只有 ``-Base`` 变体是裸补全模型——
           规则 1 已拦截后者,这里只看 model_type。
        """
        name = Path(model_dir).name.lower()
        if "base" in name:
            return False
        if any(hint in name for hint in cls._INSTRUCT_NAME_HINTS):
            return True
        return cls.read_model_type(model_dir) in cls._CHAT_BY_DEFAULT_TYPES

    @classmethod
    def resolve(cls, model_dir: str, explicit: str | None = None) -> str:
        """推断 prompter 注册名;base 模型返回 ``"empty"``(原样直传)。"""
        if explicit:
            return explicit
        if not cls.is_instruct(model_dir):
            return "empty"
        model_type = cls.read_model_type(model_dir)
        if model_type in cls._PROMPTER_BY_TYPE:
            return cls._PROMPTER_BY_TYPE[model_type]
        name = Path(model_dir).name.lower()
        for candidate in cls._NAME_CANDIDATES:
            if candidate in name:
                return candidate
        return "empty"

    @staticmethod
    def build(style: str, tokenizer) -> ChatPrompter | None:
        """构造 prompter;base 模型(``style == "empty"``)或无 chat_template 时返回 ``None``。

        返回 ``None`` 表示"原样直传":base 模型没有聊天模板,套模板反而有害
        (base Qwen2.5 收到 ``<|im_start|>assistant`` 会回显并退化重复)。instruct
        模型则交给 :func:`get_prompter`,用 tokenizer 自带的官方模板(vLLM 式)。
        """
        if style == "empty":
            return None
        return get_prompter(tokenizer)


# ---------------------------------------------------------------------------
# 第四层:子命令(Command + Template Method)
# ---------------------------------------------------------------------------


class CliCommand(ABC):
    """CLI 子命令基类。

    ``register`` 是模板方法:建子 parser → 注册公共参数 → 注册命令特有
    参数(``add_arguments`` 钩子)→ 绑定 handler。子类只补充差异部分。
    """

    name: ClassVar[str]
    help: ClassVar[str]

    def register(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(self.name, help=self.help)
        for opt in COMMON_OPTIONS:
            opt.register(sub)
        self.add_arguments(sub)
        sub.set_defaults(handler=self)
        return sub

    def add_arguments(self, sub: argparse.ArgumentParser) -> None:
        """注册命令特有参数;默认无。"""

    @staticmethod
    def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_gen_len=args.max_gen_len,
            repetition_penalty=args.repetition_penalty,
        )

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        """执行子命令,返回进程退出码。"""


class ChatCommand(CliCommand):
    """``chat``:交互式文本对话(REPL 循环,逐 token 流式输出)。"""

    name = "chat"
    help = "Interactive text chat"

    def add_arguments(self, sub: argparse.ArgumentParser) -> None:
        CliOption(
            "--prompt-style", {"help": "Prompter name (auto-detected by default)"}
        ).register(sub)
        CliOption(
            "--use-cuda-graph",
            {
                "action": "store_true",
                "help": "Capture decode CUDA graphs for faster generation (text models only)",
            },
        ).register(sub)

    def run(self, args: argparse.Namespace) -> int:
        opts = EngineOptions.from_args(args)

        # Tensor parallelism: spawn N processes, each running the same engine.
        # All ranks call forward with the same inputs; the NCCL all-reduce
        # inside RowParallelLinear keeps them in lockstep.
        if opts.tensor_parallel_size > 1:
            return self._run_tp(args, opts)

        generator = opts.build_text_generator()
        style = PrompterResolver.resolve(opts.model_dir, args.prompt_style)
        prompter = PrompterResolver.build(style, generator.tokenizer)
        params = self.build_sampling_params(args)

        self._print_banner(opts.model_dir, style, params)
        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C 退出
                print()
                return 0
            if not user_input:
                continue
            if user_input.lower() == "exit":
                return 0
            self._stream_reply(generator, prompter, params, user_input)

    def _run_tp(self, args: argparse.Namespace, opts: EngineOptions) -> int:
        """Run TP chat: main process is rank 0 (has stdin), workers mirror forwards."""
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp

        from .distributed.parallel_state import init_tensor_parallel

        world_size = opts.tensor_parallel_size

        # Spawn mirror workers for ranks 1..N-1.
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        workers = []
        for rank in range(1, world_size):
            p = mp.Process(
                target=_tp_mirror_worker,
                args=(rank, world_size, opts.model_dir, opts.max_seq_len,
                      opts.max_gpu_num_blocks, opts.quantization),
                daemon=True,
            )
            p.start()
            workers.append(p)

        # Main process = rank 0.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(0)
        init_tensor_parallel(rank=0, world_size=world_size)

        generator = TextGenerator(
            checkpoints_dir=opts.model_dir,
            max_seq_len=opts.max_seq_len,
            max_gpu_num_blocks=opts.max_gpu_num_blocks,
            device="cuda:0",
            use_cuda_graph=False,
            quantization=opts.quantization,
            tensor_parallel_size=world_size,
        )
        style = PrompterResolver.resolve(opts.model_dir, getattr(args, "prompt_style", None))
        prompter = PrompterResolver.build(style, generator.tokenizer)
        params = self.build_sampling_params(args)
        self._print_banner(opts.model_dir, style, params)

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                user_input = "exit"

            # Broadcast flag: 1=generate, 0=exit.
            flag_val = 0 if user_input in ("", "exit") else 1
            flag = torch.tensor([flag_val], dtype=torch.int64, device="cuda:0")
            dist.broadcast(flag, src=0)
            if flag_val == 0:
                break
            if not user_input:
                continue

            # Tokenize and broadcast prompt tokens to workers.
            formatted = prompter.insert_prompt(user_input) if prompter else user_input
            tokens = generator.tokenizer.encode(formatted)
            tok_tensor = torch.tensor(tokens, dtype=torch.int64, device="cuda:0")
            length = torch.tensor([len(tokens)], dtype=torch.int64, device="cuda:0")
            dist.broadcast(length, src=0)
            dist.broadcast(tok_tensor, src=0)

            # All ranks generate together (all-reduces synchronize).
            prompt_text = generator.tokenizer.decode(tokens)
            for step_tokens in generator.stream([prompt_text], params):
                print(step_tokens[0], end="", flush=True)
            print()

        # Shutdown workers.
        for p in workers:
            p.join(timeout=5)
        dist.destroy_process_group()
        return 0

    @staticmethod
    def _print_banner(model_dir: str, style: str, params: SamplingParams) -> None:
        print(f"Loaded {Path(model_dir).name}. Type 'exit' to quit.")
        if style == "empty":
            print("(base model detected: prompts are sent verbatim, no chat template)")
        if params.is_greedy and params.repetition_penalty == 1.0:
            # greedy + base 模型是经典的重复循环组合:fp16 argmax 平局翻转
            # (~0.02 logit gap) 之后循环无法自我纠正,提前给用户提个醒
            print(
                "(greedy decoding without a repetition penalty can loop on repeats; "
                "try --temperature 0.6 or --repetition-penalty 1.1)"
            )
        print()

    @staticmethod
    def _stream_reply(
        generator: TextGenerator,
        prompter: ChatPrompter | None,
        params: SamplingParams,
        user_input: str,
    ) -> None:
        """单轮对话:套模板(若有)→ 流式打印 → 检查重复早停原因。"""
        prompt_style_input = user_input
        if prompter is not None:
            prompter.insert_prompt(user_input)
            prompt_style_input = prompter.model_input

        for step in generator.stream([prompt_style_input], params):
            print(step[0], end="", flush=True)
        reasons = generator.engine.last_stop_reasons
        if reasons and reasons[0] == "repeat":
            print(
                "\n[stopped early: degenerate repetition detected; try a higher "
                "--repetition-penalty or --temperature]",
                file=sys.stderr,
            )
        print("\n")


class VlChatCommand(CliCommand):
    """``vl-chat``:单轮图像条件对话(LLaVA / Qwen3-VL)。"""

    name = "vl-chat"
    help = "Single-turn image-conditioned chat"

    def add_arguments(self, sub: argparse.ArgumentParser) -> None:
        CliOption(
            "--image",
            {"nargs": "+", "required": True, "help": "One or more image paths"},
        ).register(sub)
        CliOption(
            "--prompt",
            {"help": "Prompt text; must contain '<image>' for LLaVA, plain text for Qwen3-VL"},
        ).register(sub)

    def run(self, args: argparse.Namespace) -> int:
        opts = EngineOptions.from_args(args)

        if opts.tensor_parallel_size > 1:
            return self._run_tp(args, opts)

        generator = opts.build_vision_generator()
        params = self.build_sampling_params(args)

        images = [Image.open(p).convert("RGB") for p in args.image]
        # LLaVA 吃裸的 "USER: <image> ... ASSISTANT:" 字符串;Qwen3-VL 只要
        # 一条纯 user 消息,由 VisionGenerator 自己套 chat template
        default_prompt = (
            "Describe this image."
            if generator.is_qwen3_vl
            else "USER: <image>\nDescribe this image. ASSISTANT:"
        )
        for delta in generator.stream(args.prompt or default_prompt, images, params):
            print(delta, end="", flush=True)
        print()
        return 0

    @staticmethod
    def _run_tp(args: argparse.Namespace, opts: EngineOptions) -> int:
        """TP vision chat: all ranks process the image independently (vision tower
        is replicated); the language model's all-reduces keep them in lockstep."""
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp

        from .distributed.parallel_state import init_tensor_parallel
        from .engine import VisionGenerator

        world_size = opts.tensor_parallel_size

        # Determine the prompt before spawning workers so they get it too.
        from .models.config import read_model_type
        is_qwen3_vl = read_model_type(opts.model_dir) == "qwen3_vl"
        default_prompt = (
            "Describe this image."
            if is_qwen3_vl
            else "USER: <image>\nDescribe this image. ASSISTANT:"
        )
        prompt = args.prompt or default_prompt

        # Spawn mirror workers for ranks 1..N-1.
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        workers = []
        for rank in range(1, world_size):
            p = mp.Process(
                target=_tp_mirror_worker,
                args=(rank, world_size, opts.model_dir, opts.max_seq_len,
                      opts.max_gpu_num_blocks, opts.quantization,
                      args.image, prompt),
                daemon=True,
            )
            p.start()
            workers.append(p)

        # Main process = rank 0.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(0)
        init_tensor_parallel(rank=0, world_size=world_size)

        generator = VisionGenerator(
            checkpoints_dir=opts.model_dir,
            max_seq_len=opts.max_seq_len,
            max_gpu_num_blocks=opts.max_gpu_num_blocks,
            device="cuda:0",
            quantization=opts.quantization,
            tensor_parallel_size=world_size,
        )
        params = CliCommand.build_sampling_params(args)

        images = [Image.open(p).convert("RGB") for p in args.image]

        # Broadcast the go-signal and prompt tokens so workers enter the same
        # forward calls as rank 0 (vision tower + language model).
        formatted = prompt
        tokens = generator.engine.tokenizer.encode(formatted)
        tok_tensor = torch.tensor(tokens, dtype=torch.int64, device="cuda:0")
        length = torch.tensor([len(tokens)], dtype=torch.int64, device="cuda:0")
        flag = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        dist.broadcast(flag, src=0)
        dist.broadcast(length, src=0)
        dist.broadcast(tok_tensor, src=0)

        for delta in generator.stream(prompt, images, params):
            print(delta, end="", flush=True)
        print()

        # Shutdown workers.
        flag.zero_()
        dist.broadcast(flag, src=0)
        for p in workers:
            p.join(timeout=5)
        dist.destroy_process_group()
        return 0


# ---------------------------------------------------------------------------
# 装配层:命令注册表 + 入口
# ---------------------------------------------------------------------------

COMMANDS: tuple[CliCommand, ...] = (ChatCommand(), VlChatCommand())
"""已注册子命令;新增命令 = 实现一个 :class:`CliCommand` 子类并加入此表。"""


def build_parser() -> argparse.ArgumentParser:
    # description 只取模块 docstring 首行;完整架构说明留在文档注释里
    parser = argparse.ArgumentParser(
        prog="lite-llama", description=__doc__.splitlines()[0] if __doc__ else None
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Silence a noisy but harmless warning from torch._utils.
    warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
    args = build_parser().parse_args(argv)
    return args.handler.run(args)


# ---------------------------------------------------------------------------
# 向后兼容 facade:tests/test_repeat_detection.py 依赖的旧函数名,
# 实现已迁入 PrompterResolver,此处仅保留委托。
# ---------------------------------------------------------------------------


def _is_instruct_checkpoint(model_dir: str) -> bool:
    return PrompterResolver.is_instruct(model_dir)


def _infer_prompter_type(model_dir: str) -> str:
    return PrompterResolver.resolve(model_dir)


if __name__ == "__main__":
    sys.exit(main())
