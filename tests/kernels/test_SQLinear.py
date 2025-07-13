import torch
import torch.nn as nn

import numpy as np
from typing import Dict, List, Tuple
import time, os, sys, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lite_llama.kernels.sq_linear import SmoothQuantLinear


# Import the SmoothQuant implementation
# from smoothquant_int4 import SmoothQuantLinear

def get_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def format_table(headers: List[str], rows: List[List[str]], title: str = "") -> str:
    """Simple table formatter without external dependencies"""
    if not rows:
        return ""

    # Calculate column widths
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Create format string
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    separator = "-+-".join("-" * w for w in widths)

    # Build table
    result = []
    if title:
        total_width = sum(widths) + 3 * (len(widths) - 1)
        result.append(f"\n{title}")
        result.append("=" * max(len(title), total_width))

    result.append(fmt.format(*headers))
    result.append(separator)

    for row in rows:
        result.append(fmt.format(*[str(cell) for cell in row]))

    return "\n".join(result)


class PerformanceComparison:
    """Class to compare SmoothQuant INT4 with nn.Linear"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = []

    def create_layers(self, in_features: int, out_features: int, group_size: int = 128):
        """Create both SmoothQuant and nn.Linear layers with same weights"""
        # Create standard linear layer
        linear_layer = nn.Linear(in_features, out_features, bias=True, dtype=torch.float16)
        linear_layer = linear_layer.to(self.device)

        # Create SmoothQuant layer
        sq_layer = SmoothQuantLinear(in_features, out_features, bias=True, group_size=group_size)
        sq_layer = sq_layer.to(self.device)

        # Copy weights and bias from linear to SmoothQuant
        with torch.no_grad():
            weight = linear_layer.weight.data.clone()
            bias = linear_layer.bias.data.clone() if linear_layer.bias is not None else None

            # Generate some sample activations to compute scales for SmoothQuant
            sample_input = torch.randn(32, 128, in_features, dtype=torch.float16, device=self.device)
            act_scales = sample_input.abs().amax(dim=(0, 1))

            # Quantize the weights
            sq_layer.quantize_weight(weight, act_scales)
            if bias is not None:
                sq_layer.bias.copy_(bias)

        return linear_layer, sq_layer

    def measure_memory(self, layer, input_tensor, layer_name: str):
        """Measure memory usage of a layer"""
        clear_memory()

        # Baseline memory
        baseline_memory = get_memory_usage()

        # Model memory (parameters) - handle both regular and buffer parameters
        model_memory = 0
        for param in layer.parameters():
            model_memory += param.numel() * param.element_size()

        # Also count registered buffers (important for SmoothQuant)
        for buffer in layer.buffers():
            model_memory += buffer.numel() * buffer.element_size()

        model_memory = model_memory / 1024 / 1024  # Convert to MB

        # Ensure we have a minimum memory value to avoid division by zero
        model_memory = max(model_memory, 0.001)  # At least 1KB

        # Forward pass memory
        with torch.no_grad():
            _ = layer(input_tensor)
            peak_memory = get_memory_usage()

        activation_memory = peak_memory - baseline_memory - model_memory

        return {
            'layer_name': layer_name,
            'model_memory_mb': model_memory,
            'activation_memory_mb': max(0, activation_memory),
            'total_memory_mb': peak_memory - baseline_memory
        }

    def measure_speed(self, layer, input_tensor, num_warmup: int = 10, num_runs: int = 100):
        """Measure inference speed of a layer"""
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = layer(input_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        # Timing
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()

                _ = layer(input_tensor)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()

                times.append((end - start) * 1000)  # Convert to ms

        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times)
        }

    def measure_accuracy(self, linear_layer, sq_layer, input_tensor):
        """Compare numerical accuracy between layers"""
        with torch.no_grad():
            # Get outputs
            linear_output = linear_layer(input_tensor)
            sq_output = sq_layer(input_tensor)

            # Calculate differences
            abs_diff = (linear_output - sq_output).abs()
            rel_diff = abs_diff / (linear_output.abs() + 1e-8)

            return {
                'mean_abs_error': abs_diff.mean().item(),
                'max_abs_error': abs_diff.max().item(),
                'mean_rel_error': rel_diff.mean().item(),
                'max_rel_error': rel_diff.max().item(),
                'mse': ((linear_output - sq_output) ** 2).mean().item(),
                'cosine_similarity': torch.nn.functional.cosine_similarity(
                    linear_output.flatten(), sq_output.flatten(), dim=0
                ).item()
            }

    def run_comparison(self, test_configs: List[Dict]):
        """Run comparison across multiple configurations"""
        print(f"Running comparison on {self.device}")
        print("=" * 80)

        for config in test_configs:
            print(f"\nTesting: {config['name']}")
            print("-" * 40)

            # Create input
            input_tensor = torch.randn(
                config['batch_size'],
                config['seq_len'],
                config['in_features'],
                dtype=torch.float16,
                device=self.device
            )

            # Create layers
            linear_layer, sq_layer = self.create_layers(
                config['in_features'],
                config['out_features'],
                config.get('group_size', 128)
            )

            # Measure memory
            linear_memory = self.measure_memory(linear_layer, input_tensor, 'nn.Linear')
            sq_memory = self.measure_memory(sq_layer, input_tensor, 'SmoothQuant')

            # Measure speed
            linear_speed = self.measure_speed(linear_layer, input_tensor)
            sq_speed = self.measure_speed(sq_layer, input_tensor)

            # Measure accuracy
            accuracy = self.measure_accuracy(linear_layer, sq_layer, input_tensor)

            # Calculate throughput (tokens/sec)
            total_tokens = config['batch_size'] * config['seq_len']
            linear_throughput = total_tokens / (linear_speed['mean_time_ms'] / 1000)
            sq_throughput = total_tokens / (sq_speed['mean_time_ms'] / 1000)

            # Store results
            result = {
                'config': config,
                'linear_memory': linear_memory,
                'sq_memory': sq_memory,
                'linear_speed': linear_speed,
                'sq_speed': sq_speed,
                'accuracy': accuracy,
                'linear_throughput': linear_throughput,
                'sq_throughput': sq_throughput,
                'speedup': linear_speed['mean_time_ms'] / sq_speed['mean_time_ms'],
                'memory_reduction': linear_memory['model_memory_mb'] / sq_memory['model_memory_mb']
            }

            self.results.append(result)

            # Print summary for this config
            self.print_config_summary(result)

    def print_config_summary(self, result):
        """Print summary for a single configuration"""
        config = result['config']

        print(f"Input shape: [{config['batch_size']}, {config['seq_len']}, {config['in_features']}]")
        print(f"Output features: {config['out_features']}")

        # Speed comparison
        print(f"\n🕐 Speed Comparison:")
        print(
            f"  nn.Linear:    {result['linear_speed']['mean_time_ms']:.3f} ± {result['linear_speed']['std_time_ms']:.3f} ms")
        print(f"  SmoothQuant:  {result['sq_speed']['mean_time_ms']:.3f} ± {result['sq_speed']['std_time_ms']:.3f} ms")
        print(f"  Speedup:      {result['speedup']:.2f}x")

        # Throughput
        print(f"\n🚀 Throughput:")
        print(f"  nn.Linear:    {result['linear_throughput']:.0f} tokens/sec")
        print(f"  SmoothQuant:  {result['sq_throughput']:.0f} tokens/sec")

        # Memory comparison
        print(f"\n💾 Memory Usage (Model Parameters):")
        print(f"  nn.Linear:    {result['linear_memory']['model_memory_mb']:.2f} MB")
        print(f"  SmoothQuant:  {result['sq_memory']['model_memory_mb']:.2f} MB")
        print(f"  Reduction:    {result['memory_reduction']:.2f}x")

        # Accuracy
        print(f"\n📊 Accuracy:")
        print(f"  Mean Abs Error:     {result['accuracy']['mean_abs_error']:.6f}")
        print(f"  Max Abs Error:      {result['accuracy']['max_abs_error']:.6f}")
        print(f"  Mean Rel Error:     {result['accuracy']['mean_rel_error']:.4f}")
        print(f"  Cosine Similarity:  {result['accuracy']['cosine_similarity']:.6f}")

    def generate_report(self):
        """Generate comprehensive report with tables"""
        if not self.results:
            print("No results to report!")
            return

        print("\n" + "=" * 80)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("=" * 80)

        # Summary table
        self.print_summary_table()

        # Additional detailed analysis
        self.print_detailed_analysis()

    def print_summary_table(self):
        """Print summary table of all results"""
        headers = [
            "Config", "Batch×Seq", "Features",
            "Linear (ms)", "SmoothQ (ms)", "Speedup",
            "Linear (MB)", "SmoothQ (MB)", "Mem Reduction",
            "Mean Abs Err", "Cos Sim"
        ]

        rows = []
        for result in self.results:
            config = result['config']
            rows.append([
                config['name'],
                f"{config['batch_size']}×{config['seq_len']}",
                f"{config['in_features']}→{config['out_features']}",
                f"{result['linear_speed']['mean_time_ms']:.2f}",
                f"{result['sq_speed']['mean_time_ms']:.2f}",
                f"{result['speedup']:.2f}x",
                f"{result['linear_memory']['model_memory_mb']:.1f}",
                f"{result['sq_memory']['model_memory_mb']:.1f}",
                f"{result['memory_reduction']:.2f}x",
                f"{result['accuracy']['mean_abs_error']:.4f}",
                f"{result['accuracy']['cosine_similarity']:.4f}"
            ])

        print(format_table(headers, rows, "📋 Summary Table"))

        # Overall statistics
        speedups = [r['speedup'] for r in self.results]
        memory_reductions = [r['memory_reduction'] for r in self.results]
        accuracies = [r['accuracy']['cosine_similarity'] for r in self.results]

        print(f"\n📈 Overall Statistics:")
        print(f"  Average Speedup:       {np.mean(speedups):.2f}x (±{np.std(speedups):.2f})")
        print(f"  Average Memory Reduction: {np.mean(memory_reductions):.2f}x (±{np.std(memory_reductions):.2f})")
        print(f"  Average Cosine Similarity: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")

    def print_detailed_analysis(self):
        """Print detailed analysis without plots"""
        if not self.results:
            return

        print("\n" + "=" * 60)
        print("DETAILED ANALYSIS")
        print("=" * 60)

        # Performance analysis
        print("\n🚀 Performance Analysis:")
        print("-" * 30)

        best_speedup = max(self.results, key=lambda x: x['speedup'])
        worst_speedup = min(self.results, key=lambda x: x['speedup'])

        print(f"Best speedup:  {best_speedup['speedup']:.2f}x ({best_speedup['config']['name']})")
        print(f"Worst speedup: {worst_speedup['speedup']:.2f}x ({worst_speedup['config']['name']})")

        # Memory analysis
        print("\n💾 Memory Analysis:")
        print("-" * 30)

        best_memory = max(self.results, key=lambda x: x['memory_reduction'])
        worst_memory = min(self.results, key=lambda x: x['memory_reduction'])

        print(f"Best memory reduction:  {best_memory['memory_reduction']:.2f}x ({best_memory['config']['name']})")
        print(f"Worst memory reduction: {worst_memory['memory_reduction']:.2f}x ({worst_memory['config']['name']})")

        # Accuracy analysis
        print("\n📊 Accuracy Analysis:")
        print("-" * 30)

        best_accuracy = max(self.results, key=lambda x: x['accuracy']['cosine_similarity'])
        worst_accuracy = min(self.results, key=lambda x: x['accuracy']['cosine_similarity'])

        print(
            f"Best accuracy:  {best_accuracy['accuracy']['cosine_similarity']:.6f} ({best_accuracy['config']['name']})")
        print(
            f"Worst accuracy: {worst_accuracy['accuracy']['cosine_similarity']:.6f} ({worst_accuracy['config']['name']})")

        # Efficiency analysis
        print("\n⚡ Efficiency Analysis:")
        print("-" * 30)

        for result in self.results:
            config = result['config']
            efficiency_score = result['speedup'] * result['memory_reduction'] * result['accuracy']['cosine_similarity']
            print(f"{config['name']:12}: Efficiency Score = {efficiency_score:.3f}")

        # Scaling analysis
        print("\n📈 Scaling Analysis:")
        print("-" * 30)

        # Sort by problem size (total parameters)
        sorted_results = sorted(self.results, key=lambda x: x['config']['in_features'] * x['config']['out_features'])

        for result in sorted_results:
            config = result['config']
            total_params = config['in_features'] * config['out_features']
            print(
                f"{config['name']:12}: {total_params:>10,} params, {result['speedup']:.2f}x speedup, {result['memory_reduction']:.2f}x memory")

        # Recommendations
        print("\n💡 Recommendations:")
        print("-" * 30)

        avg_speedup = np.mean([r['speedup'] for r in self.results])
        avg_memory = np.mean([r['memory_reduction'] for r in self.results])
        avg_accuracy = np.mean([r['accuracy']['cosine_similarity'] for r in self.results])

        if avg_speedup > 1.5:
            print("✅ SmoothQuant shows significant speed improvements")
        else:
            print("⚠️  SmoothQuant speed improvements are marginal")

        if avg_memory > 3.0:
            print("✅ SmoothQuant provides excellent memory savings")
        else:
            print("⚠️  SmoothQuant memory savings are lower than expected")

        if avg_accuracy > 0.99:
            print("✅ SmoothQuant maintains high numerical accuracy")
        elif avg_accuracy > 0.95:
            print("⚠️  SmoothQuant shows moderate accuracy degradation")
        else:
            print("❌ SmoothQuant shows significant accuracy degradation")


def run_comprehensive_test():
    """Run comprehensive comparison test"""

    # Test configurations
    test_configs = [
        {
            'name': 'Small',
            'batch_size': 1,
            'seq_len': 128,
            'in_features': 512,
            'out_features': 512,
            'group_size': 128
        },
        {
            'name': 'Medium',
            'batch_size': 4,
            'seq_len': 256,
            'in_features': 1024,
            'out_features': 1024,
            'group_size': 128
        },
        {
            'name': 'Large',
            'batch_size': 8,
            'seq_len': 512,
            'in_features': 2048,
            'out_features': 2048,
            'group_size': 128
        },
        {
            'name': 'Very Large',
            'batch_size': 16,
            'seq_len': 1024,
            'in_features': 4096,
            'out_features': 4096,
            'group_size': 128
        },
        {
            'name': 'LLaMA-like',
            'batch_size': 1,
            'seq_len': 2048,
            'in_features': 4096,
            'out_features': 11008,  # Typical MLP dimension
            'group_size': 128
        }
    ]

    # Run comparison
    comparison = PerformanceComparison()
    comparison.run_comparison(test_configs)
    comparison.generate_report()

    return comparison


if __name__ == "__main__":
    print("SmoothQuant INT4 vs nn.Linear Comprehensive Comparison")
    print("=" * 60)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("Using CPU (GPU not available)")

    print()

    # Run the comprehensive test
    comparison = run_comprehensive_test()

    print(f"\n🎉 Comparison complete!")