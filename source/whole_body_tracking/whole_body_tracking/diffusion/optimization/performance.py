"""Performance optimization utilities for diffusion policy deployment."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import time
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking inference speed."""
    
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    memory_usage_mb: float
    batch_size: int
    
    @property
    def frequency_hz(self) -> float:
        """Compute control frequency in Hz."""
        return 1000.0 / self.total_time_ms if self.total_time_ms > 0 else 0.0
    
    @property
    def meets_target(self) -> bool:
        """Check if meets 20ms target latency."""
        return self.total_time_ms < 20.0


class PerformanceOptimizer:
    """
    Optimizes diffusion model for real-time performance.
    
    Key optimizations:
    - Model compilation with TorchScript
    - Mixed precision inference
    - Batch processing
    - Memory management
    """
    
    def __init__(
        self,
        target_latency_ms: float = 20.0,
        use_mixed_precision: bool = True,
        use_torch_compile: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize performance optimizer.
        
        Args:
            target_latency_ms: Target latency in milliseconds
            use_mixed_precision: Whether to use fp16/bf16
            use_torch_compile: Whether to compile model with torch.compile
            device: Device for computation
        """
        self.target_latency_ms = target_latency_ms
        self.use_mixed_precision = use_mixed_precision
        self.use_torch_compile = use_torch_compile
        self.device = device
        
        # Performance tracking
        self.metrics_history = []
        
        logger.info(
            f"PerformanceOptimizer initialized with target={target_latency_ms}ms"
        )
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply optimizations to model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        logger.info("Applying performance optimizations to model...")
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        # Apply mixed precision
        if self.use_mixed_precision and self.device == "cuda":
            logger.info("Enabling mixed precision (fp16)")
            model = model.half()
        
        # Compile with torch.compile (PyTorch 2.0+)
        if self.use_torch_compile and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile")
            model = torch.compile(
                model,
                mode="reduce-overhead",  # Optimize for latency
                backend="inductor",
            )
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def profile_inference(
        self,
        model: nn.Module,
        input_tensors: Dict[str, torch.Tensor],
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> PerformanceMetrics:
        """
        Profile model inference performance.
        
        Args:
            model: Model to profile
            input_tensors: Input tensors for model
            num_warmup: Number of warmup iterations
            num_iterations: Number of profiling iterations
            
        Returns:
            Performance metrics
        """
        logger.info(f"Profiling inference with {num_iterations} iterations...")
        
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(**input_tensors)
        
        # Synchronize CUDA
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Profile
        times = []
        memory_usage = []
        
        for _ in range(num_iterations):
            # Memory tracking
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            # Time inference
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(**input_tensors)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            times.append(elapsed)
            
            # Memory tracking
            if self.device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append((peak_memory - start_memory) / 1024 / 1024)  # MB
        
        # Compute statistics
        avg_time = np.mean(times)
        avg_memory = np.mean(memory_usage) if memory_usage else 0.0
        
        metrics = PerformanceMetrics(
            inference_time_ms=avg_time,
            preprocessing_time_ms=0.0,
            postprocessing_time_ms=0.0,
            total_time_ms=avg_time,
            memory_usage_mb=avg_memory,
            batch_size=input_tensors[list(input_tensors.keys())[0]].shape[0],
        )
        
        self.metrics_history.append(metrics)
        
        logger.info(
            f"Profiling results: {avg_time:.2f}ms @ {metrics.frequency_hz:.1f}Hz "
            f"({'PASS' if metrics.meets_target else 'FAIL'})"
        )
        
        return metrics
    
    def optimize_batch_size(
        self,
        model: nn.Module,
        base_input: Dict[str, torch.Tensor],
        max_batch_size: int = 32,
    ) -> int:
        """
        Find optimal batch size for target latency.
        
        Args:
            model: Model to optimize
            base_input: Base input tensors (batch_size=1)
            max_batch_size: Maximum batch size to test
            
        Returns:
            Optimal batch size
        """
        logger.info("Finding optimal batch size...")
        
        optimal_batch_size = 1
        
        for batch_size in range(1, max_batch_size + 1):
            # Create batched input
            batched_input = {}
            for key, tensor in base_input.items():
                # Expand to batch size
                shape = [batch_size] + list(tensor.shape[1:])
                batched_input[key] = tensor.expand(*shape)
            
            # Profile with this batch size
            metrics = self.profile_inference(
                model,
                batched_input,
                num_warmup=5,
                num_iterations=20,
            )
            
            if metrics.meets_target:
                optimal_batch_size = batch_size
            else:
                break
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def create_tensorrt_engine(
        self,
        model: nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        fp16: bool = True,
    ) -> Optional[Any]:
        """
        Create TensorRT engine for maximum performance.
        
        Args:
            model: Model to convert
            example_inputs: Example inputs for tracing
            fp16: Whether to use FP16 precision
            
        Returns:
            TensorRT engine or None if not available
        """
        try:
            import torch_tensorrt
            
            logger.info("Creating TensorRT engine...")
            
            # Trace the model
            traced_model = torch.jit.trace(model, example_inputs)
            
            # Compile with TensorRT
            trt_model = torch_tensorrt.compile(
                traced_model,
                inputs=[
                    torch_tensorrt.Input(
                        shape=tensor.shape,
                        dtype=torch.float16 if fp16 else torch.float32,
                    )
                    for tensor in example_inputs.values()
                ],
                enabled_precisions={torch.float16} if fp16 else {torch.float32},
                workspace_size=1 << 30,  # 1GB workspace
                truncate_long_and_double=True,
            )
            
            logger.info("TensorRT engine created successfully")
            return trt_model
            
        except ImportError:
            logger.warning("TensorRT not available, skipping optimization")
            return None
        except Exception as e:
            logger.error(f"Failed to create TensorRT engine: {e}")
            return None
    
    def print_optimization_summary(self):
        """Print summary of optimization results."""
        if not self.metrics_history:
            logger.warning("No metrics collected yet")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE OPTIMIZATION SUMMARY")
        print("="*60)
        
        best_metric = min(self.metrics_history, key=lambda m: m.total_time_ms)
        worst_metric = max(self.metrics_history, key=lambda m: m.total_time_ms)
        avg_time = np.mean([m.total_time_ms for m in self.metrics_history])
        
        print(f"Target Latency: {self.target_latency_ms:.1f}ms")
        print(f"Best Performance: {best_metric.total_time_ms:.2f}ms @ {best_metric.frequency_hz:.1f}Hz")
        print(f"Worst Performance: {worst_metric.total_time_ms:.2f}ms @ {worst_metric.frequency_hz:.1f}Hz")
        print(f"Average: {avg_time:.2f}ms")
        print(f"Memory Usage: {best_metric.memory_usage_mb:.1f}MB")
        
        if best_metric.meets_target:
            print(f"✓ MEETS TARGET LATENCY")
        else:
            print(f"✗ FAILS TARGET LATENCY (needs {avg_time - self.target_latency_ms:.1f}ms improvement)")
        
        print("="*60)


class CachedInference:
    """
    Caches recent inferences to avoid redundant computation.
    
    Useful for scenarios with similar or repeated inputs.
    """
    
    def __init__(self, cache_size: int = 100):
        """
        Initialize cached inference.
        
        Args:
            cache_size: Maximum number of cached entries
        """
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_key(self, tensor: torch.Tensor) -> str:
        """Create cache key from tensor."""
        # Use a hash of the tensor's data
        return str(hash(tensor.cpu().numpy().tobytes()))
    
    def __call__(
        self,
        model_fn,
        inputs: Dict[str, torch.Tensor],
    ) -> Any:
        """
        Execute model with caching.
        
        Args:
            model_fn: Model forward function
            inputs: Input tensors
            
        Returns:
            Model output
        """
        # Create cache key
        key = self.get_cache_key(inputs[list(inputs.keys())[0]])
        
        # Check cache
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        
        # Cache miss - compute
        self.cache_misses += 1
        output = model_fn(**inputs)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = output
        
        return output
    
    def print_statistics(self):
        """Print cache statistics."""
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_rate = self.cache_hits / total * 100
            logger.info(
                f"Cache statistics: {self.cache_hits} hits, "
                f"{self.cache_misses} misses ({hit_rate:.1f}% hit rate)"
            )