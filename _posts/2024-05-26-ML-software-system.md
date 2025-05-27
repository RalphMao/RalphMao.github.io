---
layout: post
title: LLM training and inference system - a software view
---

Optimizing an LLM inference software system requires a thorough understanding of the full stack. As the inference ecosystem becomes more complex over time, understanding a system requires different levels of abstraction. This article is not a review, nor a best practice guide, but a sharing of my perspective on the full picture, key bottlenecks.

First of all, a system is designed for objectives under constraints. 
For an LLM system, the most important objectives are **throughput** (total tokens per second) and **latency** (tokens per second per user), and the constraints are **compute** (operation speed and types), **memory** (memory size and hierarchy), and **communication** (bandwidth, latency and hierarchy).
 Google has a great book that talks about these three concepts, see [Scaling Book - Section Roofline](https://jax-ml.github.io/scaling-book/roofline/).  

### Understanding LLM system with 3-layer abstraction model

To understand a system, layered abstraction is a good approach. 
I personally like to abstract a machine learning system into 3 layers \- **kernel layer**, **model layer** and **system layer**. Each abstraction layer handles a specific type and granularity of constraints and exposes unique optimization opportunities. 

Below is the summary of the 3 abstraction layers.   
For example, Triton and CUDA are grouped into the same layer because they are essentially optimizing the same targets at chip/micro-architecture level, even though Triton provides a higher-level interface than CUDA. Such a abstraction model sets up a holistic view of the key trade-offs at different levels.

| Abstraction Layer |  Operations | Representative Software |
| :---- | :---- | :---- |
| Kernel programming | Scalar/Tile instructions | CUDA, Triton |
| Model programming | Tensor operations| PyTorch, JAX, TensorRT, ONNXRuntime |
| System programming | Sharding, batching, offloading, etc| TensorRT-LLM, vLLM, Megatron-LM |

#### Kernel programming layer
Kernel layer focuses on software optimization at the micro-architecture level. A kernel is the smallest unit of workload executed on an accelerator like GPU.

The programming model at this layer maps the available hardware resources (general-purpose core, matrix multiplication unit, near-processor memory, GPU memory) to programming concepts (thread & block, local memory, global memory) and exposes necessary instructions to manipulate them.

More and more choices at this layer has become available in the past few years. Two different trends are observed, which are not mutually exclusive:

1. **Tile languages** like Triton, CuTile, TileLang. They elevate the control granularity from thread to block and data granularity from scalar to tile. Users only handle the block level logic and the intra-block arrangement is delegated to compiler. Two advantages \- simpler for users, especially for machine learning engineers and researchers; easier to maintain cross-platform compatibility. Whether using 128x32 MMA instruction or 32x32 MMA instruction is now decided by compiler instead of users.   

2. **Template frameworks** like CUTLASS. Since matrix multiplication is the most important optimization problem in kernel programming, CUTLASS takes care of the core matrix multiplication and leave the customizable “peripheral” code to users

Kernel optimization targets latency (throughput is not directly exposed to this layer), and the common solutions are exploiting data locality and avoid bank conflict (communication), utilizing near-processor registers and memory (memory), and leveraging special instructions and overlapped workload (compute). 

One good example is [matrix multiplication from official Triton example](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py). Several things to be noted:

![Visualization of a GEMM kernel executed on a GPU](/images/blog1/kernel.png)

1. This kernel uses output tiling, i.e., splits the workload by the output and distributes the sub-workload (block) to each processor.   
2. Each output tile requires a row of tiles from input 1 and a column of tiles from input 2, which are sequentially loaded. The accumulated output tile can be held locally, avoiding data movement to the main memory.  
3. Inside a tile, Triton handles the mapping of the computation and data loading to accelerated hardware instructions. This is different in CUDA, where the user has fine-grained control together with the liability to orchestrate the low-level CUDA cores, TensorCore, and shared memory.

Note that the tile size depends on the processor’s local memory size and compute element size. For example, the tile size on a TPU is typically larger than GPU.

Another great example is the use of online softmax in flash attention. Softmax is computed on a full row of the big attention matrix, which requires a lot of data movement (in contrast to data locality\!) Online softmax solves this problem by converting the global formula to a recurrent formula. In the new recurrent formula, no read/write from global memory is required at all.

#### Model programming layer

Programming at this layer handles tensors and tensor operations. This layer focuses on model-level optimizations 

This layer emphasizes the composability and ease of change for ML models. Early in the day, composability usually sacrifices speed, and that’s why dedicated frameworks like ONNXRuntime and TensorRT are used in serious inference scenarios.  
Nowadays, PyTorch also captures a great share of inference market due to two reasons: reduced overhead of PyTorch by CUDA graph, torch compile, etc; greater model sizes that dwarf the framework overhead.

By looking at the model as a whole. New optimization opportunities are exposed hereby and they are usually associated with the properties of the machine learning models.  
Generally, the goal is still to reduce communication and memory size and improve compute efficiency. Specifically, there are the following types of optimization:

Merging is a low-hanging optimization. It converts two tensor operations into one mathematically equivalent operation. Example is Conv/BatchNorm merging and Multi-FC merging. One recent merging method is from MLA, which comes at the cost of increased computation but reduces data movement.

Fusion is another commonly used technique and often confused with merging. Fusion doesn’t alter the mathematical formula \- it simply combines two kernels to overlap communication and computation, avoid writeback to main memory, and/or reduce kernel launch time. An example is Conv/ReLU fusion, which performs in-place ReLU instead of writing data to main memory and reading it back just for the cheap ReLU operation. In some situations, people also do GEMM/AllReduce fusion, by which partial output is immediately synchronized once computed in order to reduce latency.

An example of Llama3 implemented on GPU is shown in the following figure. Each rectangle is a kernel launched to GPU with necessary merging (represented by &) and fusion(represented by \+).  
![Visualization of Llama3 model implementation](/images/blog1/model.png)

Low precision, aka quantization, is one of the most widely used techniques in ML system optimization and a great example of algorithm/hardware co-evolution. I won’t spend time on how quantization works, but some general observation on the current trend:

1. Inference is usually one step ahead of training on lowering precision.
2. Beyond the common weight quantization, many other places that can be quantized - e.g., activation, KV cache, gradients, communication bits, etc.
3. There is no clear answer where the low precision will stop at. Some works suggest going lower than 4 bits doesn’t further help while BitNet shows 1.58bit works fairly well

Sparsity \- in the early days, people were more focused on static sparsity, e.g. channel pruning, 2:4 sparsity. In the era of LLM, more evidences show that total parameters matters a lot and dynamic sparsity starts to get more traction. Examples are attention sparsity. 

There are more system optimizations at the start of model design. In the early days, MobileNet was invented to alleviate the high computational density of convolution. Now in the era of LLM, bandwidth becomes the key bottleneck and we are seeing optimization like MoE and GQA/MLA/SSM in the architecture design. MoE could also be viewed as trained dynamic sparsity, and it should be pointed out that MoE and hard attention have many similarities if we view expert weights as activated tokens

#### System programming layer

This layer deals with the resources needed by a model and the constraints of a pod \- the minimum repeatable deployment unit. As an example, DeepSeek V3 has an official implementation with the inference pod of 25x GPUs and the training pod of 2048 GPUs. At the system level, we no longer care the internal details of a model, but abstract it as an elastic program with computation, memory and communication budget.

A simplified version of the system architecture from Mooncake paper (https://arxiv.org/pdf/2407.00079)  
![Visualization of inference system architecture](/images/blog1/system.png)

At this layer, inference and training frameworks diverge significantly. Both of them are booming since the rise of LLM. Inference frameworks, like vLLM, SGLang, TRT-LLM, emphasizes efficient model implementation, parallelism, and batching, and KV management. Training frameworks iterate fast as algorithms evolve, and many of them just act as scaffolds of parallelism implementation, e.g., Megatron-LM, DeepSpeed. With the recent rise of Reinforment Learning, there is more connections between training and inference, like VeRL emphasizes seamless integration of Megatron-LM, vLLM, etc.

Again, we look at optimizations that better utilize system computation, memory and communication. 

**Parallelism is the key** for both training and inference. There are many detailed explanations, e.g., Nemo documentation ([https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html)). The general rule of thumb is:

1. All parallelism improves throughput. Not all of them improve latency.  
2. Parallelism is selected mainly based on memory and communication constraints. Computation is overall a constant value, except for corner cases like FSDP, and tensor parallel of MLA (SGLang DP-attention: https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

| Type | Description | Pros & Cons |
| :--- | :--- | :--- |
| Data parallel | Parallelize at batch dimension | No improvement for latency. No communication at inference and low communication at training. |
| Pipeline parallel | Parallelize at batch and layer dimension | No improvement for latency. Low communication at inference time and training time. Saves model memory |
| Tensor parallel | Parallelize FC layer at row/column dimension and attention at head dimension | Improves latency. High communication cost. Saves model memory |
| Expert parallel | Parallelize MoE at expert dimension | No improvement for latency at small batch size. Low communication cost when experts are highly sparse. Saves model memory. |
| Context parallel | Parallelize all layers at sequence dimension | Improve latency for long context prefill. High communication cost.  |
| Sequence parallel | Parallelize attention layers at sequence dimension | Very similar to context parallel. Can be extended to the decoding phase. |
| FSDP | Split FC layers weights | Overlaps computation and communication, but each shard still carries full computation. |

For inference-time scaling, when the model size is not too large (\<100GB), the common practice is to increase tensor parallelism size to the point where communication latency becomes comparable to compute, and then further scale up with data parallelism. This has changed with very large models, e.g., DeepSeek V3, because 

1. GPU memory of a single node cannot support large enough batch size; 

2. Tensor Parallelism doesn’t scale as good as Expert Parallelism for very sparse MoE. In addition, long-context inference also creates need for more parallelism like context parallel.

While a general rule can be found for inference-time scaling, training-time scaling should be designed more specifically based on model architecture and datacenter configurations.

Next we talk about specific optimizations for training and inference frameworks. For inference, latency and throughput are two main optimization targets. For training, latency is typically not a concern, and the whole system is throughput oriented with abundant data and compute available.

**For system-level inference optimization, the majority of the effort is using smart ways of batching to increase throughput**. And when the workload per GPU is large enough, the rest of effort is to improve user-centric latency metrics, e.g., time to first token (TTFT) and time per output token (TPOT).

Talking about smart ways of batching \- i.e., increasing the batch size under limited memory and various input patterns. Examples are continuous batching, paged attention, 

Optimization around KV cache also improves batching, but it comes with another benefits that avoid unnecessary re-computation. Prefix caching, KV offloading are examples.

For latency optimization, increasing tensor parallelism size is a way, although at the cost of overall throughput  
PD disaggregation. Why it helps \-\> further scaling model parallelism (tensor parallel, pipeline parallel, expert parallel) has diminishing return. PD disaggregation can be regarded as a general form of parallelism, similar to context parallel.  
Speculative is a special case that will be discussed later.

**Training system**

Training system operates at a much larger scale of GPUs that the single-point failure rate becomes non-negligible and requires re-design the recovery mechanism.   
Fault tolerance, checkpointing optimization  
Example: fault tolerant plugin

### Exceptions of 3-layer abstraction

**Exception 1: dataflow architecture.** Some ASICs (e.g., SambaNova, Cerebras, Groq) and their programming models are designed around data availability, i.e., execute immediately when partial data is ready. With this design, a “kernel” in the usual sense could be a whole transformer block or even the full model (cite sambanova). This certainly provides an edge on latency, combining with the high bandwidth of on-chip SRAM, at the cost of ease of programming.

**Exception 2: speculative decoding.** Speculative decoding is a beautiful technique that stems from system insights (i.e., batching is faster than execution one by one), and is well-engineered with both ML and statistics knowledge to match the original model distribution perfectly. It shows the power of full-stack optimization and it is hard to classify it into any single optimization layer. I have no doubt that if transformer continues to prevail in the next ten years, speculative decoding will be a cornerstone just like speculative execution is for modern CPUs.

### Final words
Any system design is built on shifting sands, as the trade-offs between model architecture and physical constraints are constantly evolving. 


### Reference

1. [https://jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/)  
2. Semi-analysis: https://semianalysis.com/2024/09/03/the-memory-wall/  
3. AI and memory wall: [https://arxiv.org/pdf/2403.14123](https://arxiv.org/pdf/2403.14123)  
4. Tensor Core analysis: [https://zhuanlan.zhihu.com/p/638129792](https://zhuanlan.zhihu.com/p/638129792)  
   1. [https://arxiv.org/html/2402.13499v1](https://arxiv.org/html/2402.13499v1)  
5. [https://hazyresearch.stanford.edu/blog/2024-05-12-tk](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)  
6. [https://chatgpt.com/share/67eb86bc-3804-8009-be96-102499679fb9](https://chatgpt.com/share/67eb86bc-3804-8009-be96-102499679fb9)  
7. [https://en.wikipedia.org/wiki/File:High\_Bandwidth\_Memory\_schematic.svg](https://en.wikipedia.org/wiki/File:High_Bandwidth_Memory_schematic.svg)  
8. [https://zhuanlan.zhihu.com/p/25529708891](https://zhuanlan.zhihu.com/p/25529708891)  
9. Thunderkitten: 2D tile, 1d vector: [https://hazyresearch.stanford.edu/blog/2024-05-12-tk](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)  
10. SambaNova hotchips talk: [https://hc2024.hotchips.org/assets/program/conference/day1/48\_HC2024.Sambanova.Prabhakar.final-withoutvideo.pdf](https://hc2024.hotchips.org/assets/program/conference/day1/48_HC2024.Sambanova.Prabhakar.final-withoutvideo.pdf)  
11. CUDA GTC talk: https://www.nvidia.com/en-us/on-demand/session/gtc25-s72383/  
    1. ![][image4]  
    2. ![][image5]  
12. Deepseek inference system: [https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day\_6\_one\_more\_thing\_deepseekV3R1\_inference\_system\_overview.md](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)  
13. Mooncake  
14. PD disaggregated  
    1. ![][image6]  
    2. ![][image7]
