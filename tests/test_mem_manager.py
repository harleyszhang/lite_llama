# 代码可直接运行，用于测试 KVCacheMemoryManager 的结果

import unittest
import torch, os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from lite_llama.executor.mem_manager import KVCacheMemoryManager


class TestKVCacheMemoryManager(unittest.TestCase):
    def setUp(self):
        # 使用较小的参数值以便于测试
        self.head_dim = 64
        self.num_kv_heads = 4
        self.num_layers = 2
        self.gpu_num_blocks = 9
        self.dtype = torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.manager = KVCacheMemoryManager(
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            num_layers=self.num_layers,
            gpu_num_blocks=self.gpu_num_blocks,
            dtype=self.dtype,
            device=self.device,
        )

    def test_initialization(self):
        """测试初始化状态"""
        self.assertEqual(self.manager.can_use_mem_size, self.gpu_num_blocks)
        self.assertEqual(self.manager.kv_mem_use_state.numel(), self.gpu_num_blocks)
        self.assertTrue(torch.all(self.manager.kv_mem_use_state == 0))

    def test_alloc_kvcache_success(self):
        """成功分配 3 个非连续块"""
        need_size = 3
        select_index = self.manager.alloc_kvcache(need_size)
        self.assertIsNotNone(select_index)
        self.assertEqual(select_index.numel(), need_size)
        # 检查分配的索引是否被标记为使用
        used_state = self.manager.kv_mem_use_state[select_index]
        print(
            "After alloc_kvcache(3) alloc_kvcache kv_mem_use_state ",
            self.manager.kv_mem_use_state,
        )
        self.assertTrue(torch.all(used_state == 1))
        # 检查可用内存大小是否更新
        self.assertEqual(self.manager.can_use_mem_size, self.gpu_num_blocks - need_size)

        self.manager.release_ref(select_index)
        print("after release_ref kv_mem_use_state ", self.manager.kv_mem_use_state)

    def test_alloc_kvcache_failure(self):
        """尝试分配超过可用块数量的内存"""
        need_size = self.gpu_num_blocks + 1
        self.assertIsNone(self.manager.alloc_kvcache(need_size))
        # 确保内存状态未改变
        self.assertTrue(torch.all(self.manager.kv_mem_use_state == 0))
        self.assertEqual(self.manager.can_use_mem_size, self.gpu_num_blocks)

    def test_alloc_contiguous_kvcache_success(self):
        """成功分配 4 个连续块"""
        need_size = 4
        result = self.manager.alloc_contiguous_kvcache(need_size)
        self.assertIsNotNone(result)
        select_index, start, end = result
        self.assertEqual(select_index.numel(), need_size)
        self.assertEqual(end - start, need_size)

        # 检查分配的索引是否被标记为使用
        used_state = self.manager.kv_mem_use_state[select_index]
        self.assertTrue(torch.all(used_state == 1))
        # 检查可用内存大小是否更新
        self.assertEqual(self.manager.can_use_mem_size, self.gpu_num_blocks - need_size)
        self.manager.release_ref(select_index)

    def test_alloc_contiguous_kvcache_failure(self):
        # 先分配所有块，然后尝试分配更多
        need_size = self.gpu_num_blocks
        print("self.can_use_mem_size ", self.manager.can_use_mem_size)
        result = self.manager.alloc_contiguous_kvcache(need_size)

        print("result and need_size ", result, need_size)
        select_index, _, _ = result

        self.assertIsNotNone(result)

        # 可用内存大小应为0
        self.assertEqual(self.manager.can_use_mem_size, 0)

        self.manager.release_ref(select_index)

    def test_add_ref(self):
        # 分配 2 个块
        need_size = 2
        select_index = self.manager.alloc_kvcache(need_size)
        self.assertIsNotNone(select_index)

        # 检查引用计数是否为2
        used_state = self.manager.kv_mem_use_state[select_index]  # tensor([1, 1])
        self.assertTrue(
            torch.sum(used_state != 0) == 2,
            "The number of non-zero elements is not equal to 2",
        )
        # 检查可用内存大小是否正确
        self.assertEqual(self.manager.can_use_mem_size, self.gpu_num_blocks - need_size)
        self.manager.release_ref(select_index)

    def test_release_ref(self):
        # 分配 3 个块
        need_size = 3
        select_index = self.manager.alloc_kvcache(need_size)
        self.assertIsNotNone(select_index)
        # 减少引用计数
        self.manager.release_ref(select_index)
        # 检查引用计数是否为0
        used_state = self.manager.kv_mem_use_state[select_index]
        self.assertTrue(torch.all(used_state == 0))
        # 检查可用内存大小是否增加
        self.assertEqual(self.manager.can_use_mem_size, self.gpu_num_blocks)

    def test_free_all(self):
        # 分配一些块
        need_size = 5
        select_index = self.manager.alloc_kvcache(need_size)
        self.assertIsNotNone(select_index)
        # 释放所有内存
        self.manager.free_all()
        # 检查所有块是否标记为未使用
        self.assertTrue(torch.all(self.manager.kv_mem_use_state == 0))
        # 检查可用内存大小是否重置
        self.assertEqual(self.manager.can_use_mem_size, self.gpu_num_blocks)

        self.manager.release_ref(select_index)

    def test_alloc_contiguous_kvcache_with_insufficient_memory(self):
        # 分配 8 个块
        need_size = 8
        result = self.manager.alloc_contiguous_kvcache(need_size)
        self.assertIsNotNone(result)
        # 现在尝试分配 3 个连续块，应失败
        result2 = self.manager.alloc_contiguous_kvcache(3)
        self.assertIsNone(result2)
        # 可用内存大小应为2
        self.assertEqual(self.manager.can_use_mem_size, self.gpu_num_blocks - need_size)

        self.manager.release_ref(result[0])

    def test_alloc_contiguous_kvcache_after_release_ref(self):
        # 分配 4 个连续块
        need_size = 4
        result = self.manager.alloc_contiguous_kvcache(need_size)
        self.assertIsNotNone(result)
        select_index, _, _ = result
        # 减少引用计数以释放部分块
        self.manager.release_ref(select_index[:2])  # 释放前2个块
        # 现在尝试分配2个连续块，应成功
        new_result = self.manager.alloc_contiguous_kvcache(2)
        self.assertIsNotNone(new_result)
        new_select_index, new_start, new_end = new_result
        self.assertEqual(new_select_index.numel(), 2)
        # 可用内存大小应为 gpu_num_blocks - 2
        self.assertEqual(self.manager.can_use_mem_size, self.gpu_num_blocks - 4)

        self.manager.release_ref(new_select_index)  # 释放

    def test_in_alloc_contiguous_kvcache(self):
        # 分配一些块以创建非连续场景
        need_size = 5
        select_index = self.manager.alloc_kvcache(need_size)
        self.assertIsNotNone(select_index)
        # 手动设置块8为已使用以打破连续性
        self.manager.kv_mem_use_state[7] = 1
        self.manager.can_use_mem_size -= 1
        # 现在尝试分配 3 个连续块，应失败
        contiguous_result = self.manager.alloc_contiguous_kvcache(3)
        self.assertIsNone(contiguous_result)

        self.manager.release_ref(select_index)  # 释放

    def test_free_buffers(self):
        # 分配一些块
        need_size = 2
        select_index = self.manager.alloc_kvcache(need_size)
        self.assertIsNotNone(select_index)
        self.manager.release_ref(select_index)  # 释放
        # 释放缓冲区
        self.manager._free_buffers()
        # 检查 gpu_kv_buffer 是否为 None
        self.assertIsNone(self.manager.gpu_kv_buffer)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    tests = [
        "test_initialization",
        "test_add_ref",
        "test_alloc_kvcache_success",
        "test_alloc_kvcache_failure",
        "test_alloc_contiguous_kvcache_success",
        "test_alloc_contiguous_kvcache_failure",
        "test_release_ref",
        "test_free_all",
        "test_alloc_contiguous_kvcache_after_release_ref",
        "test_alloc_contiguous_kvcache_with_insufficient_memory",
        "test_in_alloc_contiguous_kvcache",
        "test_free_buffers",
    ]
    suite.addTests(
        unittest.TestLoader().loadTestsFromNames(tests, TestKVCacheMemoryManager)
    )
    unittest.TextTestRunner().run(suite)
