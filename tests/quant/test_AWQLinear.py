import torch
import torch.nn as nn
import time
import psutil
import os
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys, os, time
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from lite_llama.quantization.utils import pack_weight, unpack_weight
    from lite_llama.quantization.quant_config import AWQConfig
    from lite_llama.quantization.awq import AWQ, quantize_awq
    from lite_llama.kernels.awq_linear import AWQLinear

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


@dataclass
class BenchmarkResults:
    """Store benchmark results for comparison"""
    layer_size: Tuple[int, int]
    batch_size: int
    sequence_length: int

    # Speed metrics (milliseconds)
    fp16_time: float
    awq_time: float
    speedup: float

    # Accuracy metrics
    max_error: float
    mean_error: float
    rmse: float
    cosine_similarity: float

    # Memory metrics (MB)
    fp16_memory: float
    awq_memory: float
    memory_saving: float


class AWQBenchmark:
    """Comprehensive benchmark for AWQ vs nn.Linear"""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[BenchmarkResults] = []

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024

    def measure_model_memory(self, model: nn.Module) -> float:
        """Measure memory footprint of a model in MB"""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / 1024 / 1024

    def create_test_data(self, batch_size: int, seq_len: int, hidden_dim: int) -> torch.Tensor:
        """Create test input data"""
        return torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=self.device)

    def quantize_linear_layer(self, linear_layer: nn.Linear, group_size: int = 128) -> AWQLinear:
        """Quantize a linear layer using AWQ"""
        # Create state dict for the layer
        state_dict = {"test_layer.weight": linear_layer.weight.data}
        if linear_layer.bias is not None:
            state_dict["test_layer.bias"] = linear_layer.bias.data

        # Quantize using AWQ
        from lite_llama.quantization.quant_config import AWQConfig
        awq_config = AWQConfig()
        quantized_dict = quantize_awq(
            model_state_dict=state_dict,
            config=awq_config,
            target_layers=["test_layer.weight"],
            device=str(self.device)
        )

        # Extract quantized parameters
        qweight = quantized_dict["test_layer.qweight"]
        qscales = quantized_dict["test_layer.qscales"]
        qzeros = quantized_dict["test_layer.qzeros"]

        # Create AWQ linear layer
        awq_layer = AWQLinear.from_float(
            linear_layer, qweight, qscales, qzeros, group_size
        )

        return awq_layer.to(self.device)

    def warmup_triton_kernel(self, model: nn.Module, input_data: torch.Tensor, warmup_runs: int = 50):
        """Extensive warmup for Triton kernels to ensure compilation and caching"""
        model.eval()
        print(f"    Warming up Triton kernels ({warmup_runs} runs)...")

        with torch.no_grad():
            # First few runs trigger compilation
            for i in range(warmup_runs):
                _ = model(input_data)
                if i < 10 and self.device.type == "cuda":
                    # Extra synchronization for first few runs to handle compilation
                    torch.cuda.synchronize()

    def measure_inference_time(self, model: nn.Module, input_data: torch.Tensor,
                               warmup_runs: int = 20, test_runs: int = 100,
                               is_triton: bool = False) -> float:
        """Measure average inference time in milliseconds"""
        model.eval()

        # Extended warmup for Triton kernels
        if is_triton:
            self.warmup_triton_kernel(model, input_data, warmup_runs=50)
        else:
            # Regular warmup for PyTorch
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(input_data)

        # Clear cache and synchronize
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Measure time
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(test_runs):
                _ = model(input_data)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        avg_time_ms = (end_time - start_time) * 1000 / test_runs
        return avg_time_ms

    def compute_accuracy_metrics(self, fp16_output: torch.Tensor,
                                 awq_output: torch.Tensor) -> Dict[str, float]:
        """Compute accuracy metrics between FP16 and AWQ outputs"""
        # Flatten tensors for easier computation
        fp16_flat = fp16_output.view(-1).float()
        awq_flat = awq_output.view(-1).float()

        # Error metrics
        error = (fp16_flat - awq_flat).abs()
        max_error = error.max().item()
        mean_error = error.mean().item()
        rmse = torch.sqrt(((fp16_flat - awq_flat) ** 2).mean()).item()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            fp16_flat.unsqueeze(0), awq_flat.unsqueeze(0)
        ).item()

        return {
            "max_error": max_error,
            "mean_error": mean_error,
            "rmse": rmse,
            "cosine_similarity": cos_sim
        }

    def benchmark_layer_size(self, in_features: int, out_features: int,
                             batch_size: int = 16, seq_len: int = 128) -> BenchmarkResults:
        """Benchmark a specific layer configuration"""
        print(f"\nBenchmarking layer [{in_features} -> {out_features}], "
              f"batch_size={batch_size}, seq_len={seq_len}")

        # Create original FP16 linear layer
        fp16_layer = nn.Linear(in_features, out_features, bias=True)
        fp16_layer = fp16_layer.to(self.device).half()

        # Quantize to AWQ
        print("  Quantizing layer...")
        awq_layer = self.quantize_linear_layer(fp16_layer)

        # Create test input
        input_data = self.create_test_data(batch_size, seq_len, in_features)

        # Measure memory usage
        fp16_memory = self.measure_model_memory(fp16_layer)
        awq_memory = self.measure_model_memory(awq_layer)
        memory_saving = (fp16_memory - awq_memory) / fp16_memory * 100

        print(f"  Memory: FP16={fp16_memory:.2f}MB, AWQ={awq_memory:.2f}MB, "
              f"Saving={memory_saving:.1f}%")

        # Measure inference speed with proper warmup
        print("  Measuring FP16 speed...")
        fp16_time = self.measure_inference_time(fp16_layer, input_data, is_triton=False)

        print("  Measuring AWQ speed (with Triton warmup)...")
        awq_time = self.measure_inference_time(awq_layer, input_data, is_triton=True)

        speedup = fp16_time / awq_time if awq_time > 0 else 0.0

        print(f"  Speed: FP16={fp16_time:.3f}ms, AWQ={awq_time:.3f}ms, "
              f"Speedup={speedup:.2f}x")

        # Measure accuracy
        with torch.no_grad():
            fp16_output = fp16_layer(input_data)
            awq_output = awq_layer(input_data)

        accuracy_metrics = self.compute_accuracy_metrics(fp16_output, awq_output)

        print(f"  Accuracy: RMSE={accuracy_metrics['rmse']:.6f}, "
              f"CosSim={accuracy_metrics['cosine_similarity']:.6f}")

        # Store results
        result = BenchmarkResults(
            layer_size=(in_features, out_features),
            batch_size=batch_size,
            sequence_length=seq_len,
            fp16_time=fp16_time,
            awq_time=awq_time,
            speedup=speedup,
            max_error=accuracy_metrics["max_error"],
            mean_error=accuracy_metrics["mean_error"],
            rmse=accuracy_metrics["rmse"],
            cosine_similarity=accuracy_metrics["cosine_similarity"],
            fp16_memory=fp16_memory,
            awq_memory=awq_memory,
            memory_saving=memory_saving
        )

        self.results.append(result)
        return result

    def run_comprehensive_benchmark(self):
        """Run benchmark across different layer sizes and configurations"""
        print("Starting comprehensive AWQ vs FP16 benchmark...")
        print(f"Device: {self.device}")

        # Test configurations
        layer_configs = [
            (768, 768),  # Small transformer layer
            (768, 3072),  # FFN up projection
            (3072, 768),  # FFN down projection
            (1024, 1024),  # Medium layer
            (2048, 2048),  # Large layer
            (4096, 4096),  # Very large layer
        ]

        batch_configs = [
            (1, 128),  # Single sequence
            (8, 128),  # Small batch
            (16, 512),  # Medium batch + longer sequence
            (32, 128),  # Large batch
        ]

        # Run benchmarks
        for in_features, out_features in layer_configs:
            for batch_size, seq_len in batch_configs:
                try:
                    self.benchmark_layer_size(
                        in_features, out_features, batch_size, seq_len
                    )
                except Exception as e:
                    print(f"  Error: {e}")
                    continue

        print(f"\nCompleted {len(self.results)} benchmark tests")

    def analyze_results(self):
        """Analyze and summarize benchmark results"""
        if not self.results:
            print("No results to analyze")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Speed analysis
        avg_speedup = np.mean([r.speedup for r in self.results])
        max_speedup = max([r.speedup for r in self.results])
        min_speedup = min([r.speedup for r in self.results])

        print(f"\nSPEED ANALYSIS:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Max speedup: {max_speedup:.2f}x")
        print(f"  Min speedup: {min_speedup:.2f}x")

        # Memory analysis
        avg_memory_saving = np.mean([r.memory_saving for r in self.results])
        max_memory_saving = max([r.memory_saving for r in self.results])
        min_memory_saving = min([r.memory_saving for r in self.results])

        print(f"\nMEMORY ANALYSIS:")
        print(f"  Average memory saving: {avg_memory_saving:.1f}%")
        print(f"  Max memory saving: {max_memory_saving:.1f}%")
        print(f"  Min memory saving: {min_memory_saving:.1f}%")

        # Accuracy analysis
        avg_rmse = np.mean([r.rmse for r in self.results])
        max_rmse = max([r.rmse for r in self.results])
        avg_cosine_sim = np.mean([r.cosine_similarity for r in self.results])
        min_cosine_sim = min([r.cosine_similarity for r in self.results])

        print(f"\nACCURACY ANALYSIS:")
        print(f"  Average RMSE: {avg_rmse:.6f}")
        print(f"  Max RMSE: {max_rmse:.6f}")
        print(f"  Average cosine similarity: {avg_cosine_sim:.6f}")
        print(f"  Min cosine similarity: {min_cosine_sim:.6f}")

        # Find best and worst cases
        best_speedup_idx = np.argmax([r.speedup for r in self.results])
        worst_accuracy_idx = np.argmax([r.rmse for r in self.results])

        print(f"\nBEST SPEEDUP:")
        best_result = self.results[best_speedup_idx]
        print(f"  Layer size: {best_result.layer_size}")
        print(f"  Batch config: {best_result.batch_size}x{best_result.sequence_length}")
        print(f"  Speedup: {best_result.speedup:.2f}x")

        print(f"\nWORST ACCURACY:")
        worst_result = self.results[worst_accuracy_idx]
        print(f"  Layer size: {worst_result.layer_size}")
        print(f"  Batch config: {worst_result.batch_size}x{worst_result.sequence_length}")
        print(f"  RMSE: {worst_result.rmse:.6f}")
        print(f"  Cosine similarity: {worst_result.cosine_similarity:.6f}")

    def export_results_csv(self, filename: str = "awq_benchmark_results.csv"):
        """Export results to CSV file"""
        if not self.results:
            print("No results to export")
            return

        import csv

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'layer_size', 'batch_size', 'sequence_length',
                'fp16_time_ms', 'awq_time_ms', 'speedup',
                'max_error', 'mean_error', 'rmse', 'cosine_similarity',
                'fp16_memory_mb', 'awq_memory_mb', 'memory_saving_percent'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'layer_size': f"{result.layer_size[0]}x{result.layer_size[1]}",
                    'batch_size': result.batch_size,
                    'sequence_length': result.sequence_length,
                    'fp16_time_ms': result.fp16_time,
                    'awq_time_ms': result.awq_time,
                    'speedup': result.speedup,
                    'max_error': result.max_error,
                    'mean_error': result.mean_error,
                    'rmse': result.rmse,
                    'cosine_similarity': result.cosine_similarity,
                    'fp16_memory_mb': result.fp16_memory,
                    'awq_memory_mb': result.awq_memory,
                    'memory_saving_percent': result.memory_saving
                })

        print(f"Results exported to {filename}")


def quick_demo():
    """Quick demonstration of AWQ vs FP16 comparison"""
    print("Quick AWQ vs FP16 Demo")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create a simple test case
    in_features, out_features = 768, 768
    batch_size, seq_len = 16, 128

    print(f"Testing {in_features}→{out_features} layer with batch={batch_size}, seq_len={seq_len}")

    # Create FP16 linear layer
    fp16_layer = nn.Linear(in_features, out_features, bias=True)
    fp16_layer = fp16_layer.to(device).half()

    # Create test input
    input_data = torch.randn(batch_size, seq_len, in_features, dtype=torch.float16, device=device)

    # Create benchmark instance
    benchmark = AWQBenchmark(device=str(device))

    # Get FP16 timing
    print("Measuring FP16 performance...")
    fp16_time = benchmark.measure_inference_time(fp16_layer, input_data, is_triton=False)

    # Create quantized version
    print("Quantizing layer with AWQ...")
    awq_layer = benchmark.quantize_linear_layer(fp16_layer)

    # Get AWQ timing with proper Triton warmup
    print("Measuring AWQ performance (with Triton warmup)...")
    awq_time = benchmark.measure_inference_time(awq_layer, input_data, is_triton=True)

    # Get outputs for accuracy measurement
    with torch.no_grad():
        fp16_output = fp16_layer(input_data)
        awq_output = awq_layer(input_data)

    # Calculate metrics
    fp16_memory = benchmark.measure_model_memory(fp16_layer)
    awq_memory = benchmark.measure_model_memory(awq_layer)
    memory_saving = (fp16_memory - awq_memory) / fp16_memory * 100
    speedup = fp16_time / awq_time if awq_time > 0 else 0

    # Accuracy metrics
    accuracy_metrics = benchmark.compute_accuracy_metrics(fp16_output, awq_output)

    # Print results
    print(f"\nResults:")
    print(f"  Speed:")
    print(f"    FP16: {fp16_time:.3f}ms")
    print(f"    AWQ:  {awq_time:.3f}ms")
    print(f"    Speedup: {speedup:.2f}x")
    print(f"  Memory:")
    print(f"    FP16: {fp16_memory:.2f}MB")
    print(f"    AWQ:  {awq_memory:.2f}MB")
    print(f"    Saving: {memory_saving:.1f}%")
    print(f"  Accuracy:")
    print(f"    RMSE: {accuracy_metrics['rmse']:.6f}")
    print(f"    Cosine Similarity: {accuracy_metrics['cosine_similarity']:.6f}")
    print(f"    Max Error: {accuracy_metrics['max_error']:.6f}")


def main():
    """Main function to run the comprehensive benchmark"""

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on: {device}")

    if device == "cpu":
        print("Warning: Running on CPU. Triton kernels require CUDA for optimal performance.")

    # Ask user which test to run
    print("\nChoose test type:")
    print("1. Quick demo (single layer test)")
    print("2. Comprehensive benchmark (multiple configurations)")

    try:
        choice = input("Enter choice (1 or 2, default=1): ").strip()
        if choice == "2":
            # Create benchmark instance
            benchmark = AWQBenchmark(device=device)

            # Run comprehensive benchmark
            benchmark.run_comprehensive_benchmark()

            # Analyze results
            benchmark.analyze_results()

            # Export results
            benchmark.export_results_csv()

            print("\nBenchmark completed!")
        else:
            # Run quick demo
            quick_demo()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error running benchmark: {e}")
        # Fallback to quick demo
        print("Running quick demo instead...")
        quick_demo()


if __name__ == "__main__":
    main()