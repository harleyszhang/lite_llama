import torch
import logging

logger = logging.getLogger(__name__)

class ReqTokensManager:
    """管理请求序列的 kv 内存 tokens 的类。
    
    TokenTable 将一系列 kv tokens 映射到一组token 表中, 每个 token 表代表请求序列分配的 kv cache 内存空间。
    """
    def __init__(self, max_request_num, max_seq_len, mem_manager=None, device="cuda"):
        self.max_can_use_req_size = max_request_num
        self.can_use_req_size = max_request_num
        self.max_seq_len = max_seq_len
        self.req_state = torch.zeros((max_request_num), dtype=torch.int32, device=device)
        # 一个二维张量，形状为 [num_requests, max_seq_len]，用于存储每个请求的 Token 索引。
        # 每行表示一个请求，每列表示该请求在特定序列位置上的 Token 索引。
        self.b_req_tokens_table = torch.zeros((max_request_num, max_seq_len), dtype=torch.int32, device=device)
        # self.mem_manager = mem_manager

    # 分配批次请求需要的内存空间
    def alloc_req(self, request_num):
        if request_num > self.can_use_req_size:
            logger.error(f'Insufficient requested capacity, remaining {self.can_use_req_size}')
            return None

        logical_select_index = torch.nonzero(self.req_state==0).reshape(-1)[:request_num]
        self.req_state[logical_select_index] = 1
        self.can_use_req_size -= len(logical_select_index)
        return logical_select_index
    
    # 仅释放批次请求的索引
    def free_reqs(self, free_req_index, free_token_index):
        self.can_use_req_size += len(free_req_index)
        self.req_state[free_token_index] = 0 # 对应批次请求的索引重新置为 0
        if self.can_use_req_size == len(self.req_state):
            logger.debug(f"freed all request size {self.can_use_req_size}")
        # self.mem_manager.free(free_token_index)

    # 仅释放指定请求的索引
    def free_req(self, free_req_index):
        if free_req_index < 0 or free_req_index >= self.req_state.size(0):
            logger.error(f"Invalid free_req_index: {free_req_index}")
            return
        self.can_use_req_size += 1
        self.req_state[free_req_index] = 0
        return 
    
    # 释放所有请求的内存，将所有请求状态 req_state 重置为未分配（都归 0）。
    def free_all(self):
        self.can_use_req_size = self.max_can_use_req_size
        self.req_state[:] = 0

import unittest
import torch

class TestReqTokensManager(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mem_manager_mock = unittest.mock.MagicMock()
        self.table = ReqTokensManager(max_request_num=10, max_seq_len=5, mem_manager=self.mem_manager_mock, device=self.device)

    def test_alloc_req(self):
        indices = self.table.alloc_req(3)
        self.assertEqual(len(indices), 3)
        self.assertTrue((self.table.req_state[indices] == 1).all())

    def test_alloc_req_exceed_capacity(self):
        indices = self.table.alloc_req(11)
        self.assertIsNone(indices)

    def test_free_reqs(self):
        indices = self.table.alloc_req(3)
        self.table.free_reqs(indices, indices)
        self.assertTrue((self.table.req_state[indices] == 0).all())

    def test_free_all(self):
        self.table.alloc_req(5)
        self.table.free_all()
        self.assertTrue((self.table.req_state == 0).all())
        self.assertEqual(self.table.can_use_req_size, self.table.max_can_use_req_size)

    def test_invalid_free_req(self):
        self.table.free_req(-1)  # Should not raise an error
        self.table.free_req(100)  # Should not raise an error

if __name__ == "__main__":
    unittest.main()
