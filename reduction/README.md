## Step by step improvement of Reduction (Sum) kernel

### Reference: 
    1. https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    2. Programming massively parallel processors: a hands-on approach, David B Kirk, W Hwu Wen-Mei



1. Reduction 1 
    Result:
   
    > Total sum from CPU: 4.1943e+06
    > Total sum from GPU: 4.1943e+06
    > CPU and GPU results match
    > CPU time: 15.5031 ms
    > GPU time: 1.4103 ms
    > Speedup: 10.9928

    > DRAM Throughput  8.09% of peak


3. Reduction 2
    Total sum from CPU: 4.1943e+06
    Total sum from GPU: 4.1943e+06
    CPU and GPU results match
    CPU time: 15.577 ms
    GPU time: 1.04214 ms
    Speedup: 14.9471

4. Reduction 3
    Total sum from CPU: 4.1943e+06
    Total sum from GPU: 4.1943e+06
    CPU and GPU results match
    CPU time: 15.5184 ms
    GPU time: 1.01245 ms
    Speedup: 15.3276

    DRAM Throughput   13.40%

5. Reduction 4
    Total sum from CPU: 4.1943e+06
    Total sum from GPU: 4.1943e+06
    CPU and GPU results match
    CPU time: 15.7075 ms
    GPU time: 0.726816 ms
    Speedup: 21.6114

    DRAM Throughput   25.40%

6. Reduction 5
    Total sum from CPU: 4.1943e+06
    Total sum from GPU: 4.1943e+06
    CPU and GPU results match
    CPU time: 15.6635 ms
    GPU time: 0.623808 ms
    Speedup: 25.1095

    DRAM Throughput   25.40%

7. Reduction 6
    Total sum from CPU: 8.38861e+06
    Total sum from GPU: 8.38861e+06
    CPU and GPU results match
    CPU time: 31.2611 ms
    GPU time: 0.7824 ms
    Speedup: 39.9554

    DRAM Throughput   55.51%

8. Reduction 7
    Total sum from CPU: 8.38861e+06
    Total sum from GPU: 8.38861e+06
    CPU and GPU results match
    CPU time: 15.5017 ms
    GPU time: 0.60016 ms
    Speedup: 25.8292

    DRAM Throughput   52.09%


DRAM Throughput  12.56%
