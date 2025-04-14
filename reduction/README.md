## Step-by-Step Improvement of Reduction (Sum) Kernel

### References
1. [NVIDIA CUDA Reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
2. *Programming Massively Parallel Processors: A Hands-on Approach* - David B. Kirk, W. Hwu Wen-Mei

---

### **Reduction 1**
**Results:**
- **Total sum from CPU:** 4.1943e+06
- **Total sum from GPU:** 4.1943e+06
- **CPU and GPU results match**
- **CPU time:** 15.5031 ms
- **GPU time:** 1.4103 ms
- **Speedup:** 10.9928x
- **DRAM Throughput:** 8.09% of peak

---

### **Reduction 2**
**Results:**
- **Total sum from CPU:** 4.1943e+06
- **Total sum from GPU:** 4.1943e+06
- **CPU and GPU results match**
- **CPU time:** 15.577 ms
- **GPU time:** 1.04214 ms
- **Speedup:** 14.9471x

---

### **Reduction 3**
**Results:**
- **Total sum from CPU:** 4.1943e+06
- **Total sum from GPU:** 4.1943e+06
- **CPU and GPU results match**
- **CPU time:** 15.5184 ms
- **GPU time:** 1.01245 ms
- **Speedup:** 15.3276x
- **DRAM Throughput:** 13.40%

---

### **Reduction 4**
**Results:**
- **Total sum from CPU:** 4.1943e+06
- **Total sum from GPU:** 4.1943e+06
- **CPU and GPU results match**
- **CPU time:** 15.7075 ms
- **GPU time:** 0.726816 ms
- **Speedup:** 21.6114x
- **DRAM Throughput:** 25.40%

---

### **Reduction 5**
**Results:**
- **Total sum from CPU:** 4.1943e+06
- **Total sum from GPU:** 4.1943e+06
- **CPU and GPU results match**
- **CPU time:** 15.6635 ms
- **GPU time:** 0.623808 ms
- **Speedup:** 25.1095x
- **DRAM Throughput:** 25.40%

---

### **Reduction 6**
**Results:**
- **Total sum from CPU:** 8.38861e+06
- **Total sum from GPU:** 8.38861e+06
- **CPU and GPU results match**
- **CPU time:** 31.2611 ms
- **GPU time:** 0.7824 ms
- **Speedup:** 39.9554x
- **DRAM Throughput:** 55.51%

---

### **Reduction 7**
**Results:**
- **Total sum from CPU:** 8.38861e+06
- **Total sum from GPU:** 8.38861e+06
- **CPU and GPU results match**
- **CPU time:** 15.5017 ms
- **GPU time:** 0.60016 ms
- **Speedup:** 25.8292x
- **DRAM Throughput:** 52.09%

---

### **Overall DRAM Throughput:** 12.56%

This document presents the progressive optimization of the GPU reduction kernel, highlighting performance improvements in terms of execution time, speedup, and memory throughput.
