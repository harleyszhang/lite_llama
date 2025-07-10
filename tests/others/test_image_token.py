import unittest
import torch
import re

# 假设 IMAGE_TOKEN_INDEX 为 1000
IMAGE_TOKEN_INDEX = 32000


class MockTokenizer:
    def __init__(self, bos_token_id=101, eos_token_id=102):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, text):
        # 简单模拟分词器，将每个字符转换为其 ASCII 值
        # 并在句首添加 BOS token，如果需要
        input_ids = []
        if text.startswith("<BOS>"):
            input_ids.append(self.bos_token_id)
            text = text[5:]
        for char in text:
            input_ids.append(ord(char))
        return MockEncoding(input_ids)


class MockEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    """
    处理包含特殊标记 <image> 的文本提示, 将其转换为相应的 token 序列，并在 <image> 位置插入指定的图像 token 索引。

    参数:
        prompt (str): 包含 <image> 标记的文本。
        tokenizer: 分词器对象，需支持调用 tokenizer(chunk).input_ids。
        image_token_index (int): 用于替换 <image> 标记的图像 token 索引。
        return_tensors (str, optional): 指定返回的张量类型，例如 'pt' 表示 PyTorch 张量。

    返回:
        list 或 torch.Tensor: 生成的 token 序列。
    """
    # 使用正则表达式分割，移除 '<image>' 前的空格，但保留后的空格
    prompt_chunks = re.split(r"\s?<image>", prompt)
    # 不过滤空片段，以处理多个连续的 '<image>' 标记
    token_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]

    input_ids = []
    offset = 0
    # 检查第一个片段是否以 BOS token 开始
    if (
        len(token_chunks) > 0
        and len(token_chunks[0]) > 0
        and token_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(token_chunks[0][0])

    # 插入图像 token
    for i, chunk in enumerate(token_chunks):
        # 添加当前片段的 token，跳过 BOS token（如果已添加）
        input_ids.extend(chunk[offset:])
        offset = 0  # 仅适用于第一个片段
        # 如果不是最后一个片段，插入图像 token
        if i < len(token_chunks) - 1:
            input_ids.append(image_token_index)

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


class TestTokenizerImageToken(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()

    def test_single_image(self):
        prompt = "Hello <image> world."
        # "Hello" -> [72, 101, 108, 108, 111]
        # " world." -> [32, 119, 111, 114, 108, 100, 46]
        # After insertion: [72,101,108,108,111,1000,32,119,111,114,108,100,46]
        expected_input_ids = [
            ord("H"),
            ord("e"),
            ord("l"),
            ord("l"),
            ord("o"),
            IMAGE_TOKEN_INDEX,
            ord(" "),
            ord("w"),
            ord("o"),
            ord("r"),
            ord("l"),
            ord("d"),
            ord("."),
        ]
        result = tokenizer_image_token(prompt, self.tokenizer)
        self.assertEqual(result, expected_input_ids)

    def test_multiple_images(self):
        prompt = "A cat <image> is sitting <image> on the mat."
        # "A cat" -> [65, 32, 99, 97, 116]
        # " is sitting" -> [32, 105, 115, 32, 115, 105, 116, 116, 105, 110, 103]
        # " on the mat." -> [32, 111, 110, 32, 116, 104, 101, 32, 109, 97, 116, 46]
        # After insertion: [65,32,99,97,116,1000,32,105,115,32,115,105,116,116,105,110,103,1000,32,111,110,32,116,104,101,32,109,97,116,46]
        expected_input_ids = [
            ord("A"),
            ord(" "),
            ord("c"),
            ord("a"),
            ord("t"),
            IMAGE_TOKEN_INDEX,
            ord(" "),
            ord("i"),
            ord("s"),
            ord(" "),
            ord("s"),
            ord("i"),
            ord("t"),
            ord("t"),
            ord("i"),
            ord("n"),
            ord("g"),
            IMAGE_TOKEN_INDEX,
            ord(" "),
            ord("o"),
            ord("n"),
            ord(" "),
            ord("t"),
            ord("h"),
            ord("e"),
            ord(" "),
            ord("m"),
            ord("a"),
            ord("t"),
            ord("."),
        ]
        result = tokenizer_image_token(prompt, self.tokenizer)
        self.assertEqual(result, expected_input_ids)

    def test_no_image(self):
        prompt = "This is a text without images."
        # "This is a text without images." -> [84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 120, 116, 32, 119, 105, 116, 104, 111, 117, 116, 32, 105, 109, 97, 103, 101, 115, 46]
        expected_input_ids = [
            ord("T"),
            ord("h"),
            ord("i"),
            ord("s"),
            ord(" "),
            ord("i"),
            ord("s"),
            ord(" "),
            ord("a"),
            ord(" "),
            ord("t"),
            ord("e"),
            ord("x"),
            ord("t"),
            ord(" "),
            ord("w"),
            ord("i"),
            ord("t"),
            ord("h"),
            ord("o"),
            ord("u"),
            ord("t"),
            ord(" "),
            ord("i"),
            ord("m"),
            ord("a"),
            ord("g"),
            ord("e"),
            ord("s"),
            ord("."),
        ]
        result = tokenizer_image_token(prompt, self.tokenizer)
        self.assertEqual(result, expected_input_ids)

    def test_leading_bos_token(self):
        prompt = "<BOS>Start <image> end."
        # "<BOS>Start" -> [101, 83, 116, 97, 114, 116]
        # " end." -> [32, 101, 110, 100, 46]
        # After insertion: [101,83,116,97,114,116,1000,32,101,110,100,46]
        expected_input_ids = [
            101,  # BOS token
            ord("S"),
            ord("t"),
            ord("a"),
            ord("r"),
            ord("t"),
            IMAGE_TOKEN_INDEX,
            ord(" "),
            ord("e"),
            ord("n"),
            ord("d"),
            ord("."),
        ]
        print("expected_input_ids ", expected_input_ids)
        result = tokenizer_image_token(prompt, self.tokenizer)
        self.assertEqual(result, expected_input_ids)

    def test_consecutive_images(self):
        prompt = "Image1 <image><image> Image2."
        # "Image1" -> [73, 109, 97, 103, 101, 49]
        # "" -> []
        # " Image2." -> [32, 73, 109, 97, 103, 101, 50, 46]
        # After insertion: [73,109,97,103,101,49,1000,1000,32,73,109,97,103,101,50,46]
        expected_input_ids = [
            ord("I"),
            ord("m"),
            ord("a"),
            ord("g"),
            ord("e"),
            ord("1"),
            IMAGE_TOKEN_INDEX,
            IMAGE_TOKEN_INDEX,
            ord(" "),
            ord("I"),
            ord("m"),
            ord("a"),
            ord("g"),
            ord("e"),
            ord("2"),
            ord("."),
        ]
        result = tokenizer_image_token(prompt, self.tokenizer)
        self.assertEqual(result, expected_input_ids)

    def test_return_tensors_pt(self):
        prompt = "Hello <image> world."
        # [72,101,108,108,111,1000,32,119,111,114,108,100,46]
        expected_tensor = torch.tensor(
            [
                ord("H"),
                ord("e"),
                ord("l"),
                ord("l"),
                ord("o"),
                IMAGE_TOKEN_INDEX,
                ord(" "),
                ord("w"),
                ord("o"),
                ord("r"),
                ord("l"),
                ord("d"),
                ord("."),
            ],
            dtype=torch.long,
        )
        result = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        self.assertTrue(torch.equal(result, expected_tensor))

    def test_return_tensors_unsupported(self):
        prompt = "Hello <image> world."
        with self.assertRaises(ValueError):
            tokenizer_image_token(prompt, self.tokenizer, return_tensors="np")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
