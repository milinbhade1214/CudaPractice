import torch 
import triton
import triton.language as tl
import triton.testing
import torch.profiler

@triton.jit
def softmax_kernel(
        output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
        BLOCK_SIZE: tl.constexpr,):
    # Get the program id
    row_idx = tl.program_id(axis=0)

    # Compute the memory offset for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride

    row = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))

    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)

    # Subtract max from row and exponentiate
    numerator = tl.exp(row - row_max)

    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)

    # Normalize
    softmax_output = numerator / denominator

    # Store the output
    tl.store(out_row_start_ptr + tl.arange(0, BLOCK_SIZE), softmax_output, mask=tl.arange(0, BLOCK_SIZE) < n_cols)





def triton_softmax(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # Determine block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    grid = (n_rows, )
    softmax_kernel[grid](
        output, x, 
        x.stride(0), output.stride(0),
        n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return output


torch.manual_seed(0)
x = torch.randn(256, 1024, device='cuda')

torch_result = torch.softmax(x, dim=1)

triton_result = triton_softmax(x)

is_close = torch.allclose(torch_result, triton_result, atol=1e-6, rtol=1e-5)
print("Are the results close?", is_close)

max_diff = torch.max(torch.abs(torch_result - triton_result))
print(f"Maximum difference between pytorch and triton result: {max_diff:.2e}")


# Profile softmax execution
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,  # Records input shapes
    profile_memory=True,  # Records memory usage
    with_stack=True  # Captures stack traces (optional)
) as prof:
    torch_result = torch.softmax(x, dim=1)

# Print the profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# Benchmark Triton softmax execution
bench_time = triton.testing.do_bench(lambda: triton_softmax(x)) * 1000
print(f"Triton softmax benchmark time: {bench_time:.3f} ms")


softmax_event = next((e for e in prof.key_averages() if "softmax" in e.key), None)

if softmax_event:
    print(f"Softmax Avg CUDA Time: {softmax_event.device_time:.3f} µs")
    print(f"Softmax Avg CPU Time: {softmax_event.cpu_time:.3f} µs")

soft_time = softmax_event.device_time

print(f"SpeedUp: {soft_time/bench_time}")
