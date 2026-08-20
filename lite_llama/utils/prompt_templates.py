# Modified from https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/utils/prompt_templates.py


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 32000
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

LLAVA_DEFAULT_IMAGE_TOKEN = "<image>"


def get_image_token(model=None, model_name=None):
    return LLAVA_DEFAULT_IMAGE_TOKEN + "\\n "


class BasePrompter:
    """用于构建模型的提示词 (Prompt) 模板和管理对话流程"""

    def __init__(
        self,
        system_inst,
        role1,
        role2,
        sen_spliter="\n",
        qa_spliter="\n",
        colon=":",
        decorator: list[str] | None = None,
    ):
        self.system_inst = system_inst  # System Instruction
        self.role1 = role1  # The name of USER
        self.role2 = role2  # The name of AI-Assistant
        self.sen_spliter = sen_spliter  # How to split system/user/assistant outputs
        self.qa_spliter = qa_spliter  # How to split Q&A rounds
        self.decorator = decorator
        self.colon = colon

        if self.decorator is None:
            self.starter = ""
            self.stopper = ""
        else:
            self.starter = self.decorator[0]
            self.stopper = self.decorator[1]
        if self.system_inst is None:
            self.template = (
                self.starter
                + self.role1
                + self.colon
                + " {prompt}"
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role2
                + self.colon
            )
        else:
            self.template = (
                self.starter
                + self.system_inst
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role1
                + self.colon
                + " {prompt}"
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role2
                + self.colon
            )
        self.model_input = None

    def insert_prompt(self, input_prompt):
        self.model_input = self.template.format(prompt=input_prompt)

    def update_template(self, outputs, chunk_prefilling=0):
        if chunk_prefilling:
            self.template = (
                self.role1
                + ": {prompt}"
                + self.stopper
                + self.sen_spliter  # blank space
                + self.starter
                + self.role2
                + ":"
            )
        else:
            self.template = (
                self.model_input
                + " "
                + outputs.strip()
                + self.stopper
                + self.qa_spliter
                + self.starter
                + self.role1
                + ": {prompt}"
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role2
                + ":"
            )
        self.model_input = None


class OneShotBasePrompter(BasePrompter):
    def __init__(
        self,
        oneshot_example: list[str],  # User prompt + Assistant responce
        system_inst,
        role1,
        role2,
        sen_spliter="\n",
        qa_spliter="\n",
        decorator: list[str] | None = None,
    ):
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)
        assert len(oneshot_example) == 2, "One-shot example must be a List of 2 strs."
        self.user_example = oneshot_example[0]
        self.assistant_example = oneshot_example[1]
        self.insert_prompt(self.user_example)
        self.update_template(self.assistant_example)


class EmptyPrompter(BasePrompter):
    def __init__(self):
        system_inst = ""
        role1 = ""
        role2 = ""
        sen_spliter = ""
        qa_spliter = "</s>"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class VicunaPrompter(BasePrompter):
    def __init__(self):
        system_inst = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        role1 = "USER"
        role2 = "ASSISTANT"
        sen_spliter = " "
        qa_spliter = "</s>"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class Llama2Prompter(OneShotBasePrompter):
    def __init__(self, short_prompt=False):
        system_inst = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        role1 = "### Human"
        role2 = "### Assistant"
        sen_spliter = "\n"
        qa_spliter = "</s>"
        user_example = "Got any creative ideas for a 10 year old's birthday?"
        if short_prompt:
            assistant_example = (
                "Of course! Here are some creative ideas for a 10-year-old's birthday party:\n"
                + "1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n"
                + "2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n"
                + "Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!"
            )
        else:
            assistant_example = (
                "Of course! Here are some creative ideas for a 10-year-old's birthday party:\n"
                + "1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n"
                + "2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n"
                + "3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n"
                + "4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n"
                + "5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.\n"
                + "6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.\n"
                + "7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.\n"
                + "8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.\n"
                + "Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!"
            )
        oneshot_example = [user_example, assistant_example]
        super().__init__(oneshot_example, system_inst, role1, role2, sen_spliter, qa_spliter)


class Llama3Prompter(BasePrompter):
    """
    Example:
    <|start_header_id|>user<|end_header_id|>

    Show me some attractions in Boston.<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>

    """

    def __init__(self):
        system_inst = ""
        role1 = "<|start_header_id|>user<|end_header_id|>\n\n"
        role2 = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        sen_spliter = "<|eot_id|>"
        qa_spliter = ""
        colon = ""
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter, colon=colon)


class LlavaLlamaPrompter(BasePrompter):
    def __init__(self):
        system_inst = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        role1 = "USER"
        role2 = "ASSISTANT"
        sen_spliter = " "
        qa_spliter = "</s>"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class LlavaLlama3Prompter(BasePrompter):
    """
    Example:
    <|start_header_id|>user<|end_header_id|>

    Show me some attractions in Boston.<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>

    """

    def __init__(self):
        system_inst = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
            + "You are able to understand the visual content that the user provides, "
            + "and assist the user with a variety of tasks using natural language."
        )
        role1 = "<|start_header_id|>user<|end_header_id|>\n\n"
        role2 = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        sen_spliter = "<|end_of_text|>"
        qa_spliter = ""
        colon = ""
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter, colon=colon)


class Qwen2Prompter(BasePrompter):
    def __init__(self):
        # 在 Qwen2 的提示格式下，system_inst 将包含系统信息（如角色设定）
        system_inst = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

        # role1 用作 user 信息块的起始标记，这里不需要额外标记，只需在模板中插入即可
        # role2 用作 assistant 起始标记
        # 我们在构造时，会通过 template 来定义最终的格式。

        role1 = "<|im_start|>user\n"  # 用户块开始
        role2 = "<|im_start|>assistant\n"  # 助手块开始
        sen_spliter = "\n"
        qa_spliter = "\n"
        colon = ""  # 这里不再需要冒号

        # 调用父类构造函数
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter, colon=colon)

        # 重写模板:
        # 若存在 system_inst，则模板为：
        # <|im_start|>system
        # {system_inst}
        # <|im_end|>
        # <|im_start|>user
        # {prompt}
        # <|im_end|>
        # <|im_start|>assistant
        #
        # 若不存在 system_inst，则跳过 system 块，但这里我们默认有 system_inst。

        if self.system_inst is None:
            self.template = self.role1 + "{prompt}\n" + "<|im_end|>\n" + self.role2
        else:
            self.template = (
                "<|im_start|>system\n"
                + self.system_inst
                + "\n<|im_end|>\n"
                + self.role1
                + "{prompt}\n"
                + "<|im_end|>\n"
                + self.role2
            )

    def update_template(self, outputs, chunk_prefilling=0):
        # 对于 Qwen2 来说，我们通常不需要频繁更新模板。
        # 若有特殊需求，可在此根据逻辑微调。
        # 这里保持简单，不做改动：
        if chunk_prefilling:
            self.template = self.role1 + "{prompt}\n" + "<|im_end|>\n" + self.role2
        else:
            # 若需要将对话上下文追加到模板中，可在此实现
            # 简单起见，不做复杂处理
            self.template = (
                "<|im_start|>system\n"
                + self.system_inst
                + "\n<|im_end|>\n"
                + self.role1
                + "{prompt}\n"
                + "<|im_end|>\n"
                + self.role2
            )


class FalconSimplePrompter(BasePrompter):
    def __init__(self):
        system_inst = None
        role1 = "User"
        role2 = "Assistant"
        sen_spliter = "\n\n"
        qa_spliter = "\n\n"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class FalconPrompter(BasePrompter):
    def __init__(self):
        system_inst = (
            "The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, "
            + "and a human user, called User. In the following interactions, User and Falcon will converse in natural language, "
            + "and Falcon will answer User's questions. Falcon was built to be respectful, polite and inclusive. "
            + "Falcon was built by the Technology Innovation Institute in Abu Dhabi. "
            + "Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. "
            + "It knows a lot, and always tells the truth. The conversation begins."
        )
        role1 = "User"
        role2 = "Assistant"
        sen_spliter = "\n"
        qa_spliter = "\n"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class MPTPrompter(BasePrompter):
    def __init__(self):
        system_inst = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        role1 = "### Human"
        role2 = "### Assistant"
        sen_spliter = "\n"
        qa_spliter = "\n"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class MPTChatPrompter(BasePrompter):
    def __init__(self):
        system_inst = (
            "system\n"
            + "- You are a helpful assistant chatbot trained by MosaicML.\n"
            + "- You answer questions.\n"
            + "- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n"
            + "- You are more than just an information source, you are also able to write poetry, short stories, and make jokes."
        )
        role1 = "user"
        role2 = "assistant"
        sen_spliter = "\n"
        qa_spliter = "\n"
        decorator = ["<|im_start|>", "<|im_end|>"]
        # decorator 必须按关键字传:位置参数第 6 位是 colon,位置传参会把它
        # 塞进 colon 使模板拼接崩溃(历史遗留 bug,该类从未成功实例化)。
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter, decorator=decorator)


# ---------------------------------------------------------------------------
# 注册表:model_type -> (prompter 工厂, stop-token 规则)
#
# 工厂签名 ``(model_path, short_prompt) -> BasePrompter``;规则签名
# ``(model_path) -> list[int]``。新增模型家族只需一次 register 调用,
# 不再改动 if-elif 链(Open/Closed)。
# ---------------------------------------------------------------------------

_PROMPTER_FACTORIES = {}
_STOP_TOKEN_RULES = {}


def register_prompter(model_type, factory, stop_rule=None):
    """注册一个模型家族的 prompter 工厂,可选附带 stop-token 规则。"""
    _PROMPTER_FACTORIES[model_type.lower()] = factory
    if stop_rule is not None:
        _STOP_TOKEN_RULES[model_type.lower()] = stop_rule


def get_prompter(model_type, model_path="", short_prompt=False, empty_prompt=False):
    if empty_prompt:
        return EmptyPrompter()
    factory = _PROMPTER_FACTORIES.get(model_type.lower())
    if factory is None:
        raise ValueError(f"model type {model_type} is not supported")
    return factory(model_path, short_prompt)


def get_stop_token_ids(model_type, model_path=""):
    rule = _STOP_TOKEN_RULES.get(model_type.lower())
    if rule is None:
        raise ValueError(f"model type {model_type} is not supported")
    return rule(model_path)


def _is_llama3_name(path_lower):
    # "llama-3" 前缀已涵盖 "llama-3.2";30B 变体用的是 llama2 模板
    return ("llama-3" in path_lower or "llama3" in path_lower) and "30b" not in path_lower


def _llama_prompter(model_path, short_prompt):
    path = model_path.lower()
    if "vicuna" in path:
        return VicunaPrompter()
    if _is_llama3_name(path):
        # vila 变体默认带 system prompt
        return LlavaLlama3Prompter() if "vila" in path else Llama3Prompter()
    if "llava" in path or "vila" in path:
        return LlavaLlamaPrompter()
    return Llama2Prompter(short_prompt)


def _llama_stop_tokens(model_path):
    return [128001, 128009] if _is_llama3_name(model_path.lower()) else []


def _mpt_prompter(model_path, short_prompt):
    return MPTChatPrompter() if "chat" in model_path.lower() else MPTPrompter()


def _mpt_stop_tokens(model_path):
    return [50278, 0] if "chat" in model_path.lower() else []


register_prompter("llama", _llama_prompter, stop_rule=_llama_stop_tokens)
register_prompter(
    "falcon", lambda path, short: FalconSimplePrompter(), stop_rule=lambda path: list(range(12))
)
register_prompter("mpt", _mpt_prompter, stop_rule=_mpt_stop_tokens)
# Qwen2 与 Qwen3 共用 ChatML 模板(<|im_start|> / <|im_end|>)
register_prompter("qwen2", lambda path, short: Qwen2Prompter())
register_prompter("qwen3", lambda path, short: Qwen2Prompter())


if __name__ == "__main__":
    # 使用方法示例
    prompter = get_prompter("qwen2")
    prompter.insert_prompt("给出 c++ 多线程语法和编程示例代码.")
    print(prompter.model_input)
