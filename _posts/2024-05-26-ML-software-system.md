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

2. **Template frameworks** like CUTLASS. Since matrix multiplication is the most important optimization problem in kernel programming, CUTLASS takes care of the core matrix multiplication and leave the customizable "peripheral" code to users.

Kernel optimization targets latency (throughput is not directly exposed to this layer), and the common solutions are exploiting data locality and avoid bank conflict (communication), utilizing near-processor registers and memory (memory), and leveraging special instructions and overlapped workload (compute). 

One good example is [matrix multiplication from official Triton example](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py). Several things to be noted:

![Visualization of a GEMM kernel executed on a GPU](/images/blog1/kernel.png)

1. This kernel uses output tiling, i.e., splits the workload by the output and distributes the sub-workload (block) to each processor.   
2. Each output tile requires a row of tiles from input 1 and a column of tiles from input 2, which are sequentially loaded. The accumulated output tile can be held locally, avoiding data movement to the main memory.  
3. Inside a tile, Triton handles the mapping of the computation and data loading to accelerated hardware instructions. This is different in CUDA, where the user has fine-grained control together with the liability to orchestrate the low-level CUDA cores, TensorCore, and shared memory.

Note that the tile size depends on the processor's local memory size and compute element size. For example, the tile size on a TPU is typically larger than GPU.

Another great example is the use of online softmax in flash attention. Softmax is computed on a full row of the big attention matrix, which requires a lot of data movement (in contrast to data locality\!) Online softmax solves this problem by converting the global formula to a recurrent formula. In the new recurrent formula, no read/write from global memory is required at all.

#### Model programming layer

Programming at this layer emphasizes the composability and ease of change for ML models. Early in the day, composability usually sacrifices speed, and that's why dedicated frameworks like ONNXRuntime and TensorRT are used in serious inference scenarios.  
Nowadays, PyTorch also captures a great share of inference market due to two reasons: reduced overhead of PyTorch by CUDA graph, torch compile, etc; greater model sizes that dwarf the framework overhead.


As the name suggests, this layer focuses on model-level optimizations that are associated with the properties of the machine learning models.  
Generally, the goal is still to reduce communication and memory size and improve compute efficiency. Specifically, there are the following common types of optimization:

**Merging** is a low-hanging fruit. It converts two tensor operations into one mathematically equivalent operation. Example is Conv/BatchNorm merging and Multi-FC merging. One recent merging method is MLA BMM merging, which comes at the cost of increased computation but reduces data movement (To fill in).

**Fusion** is another commonly used technique and often confused with merging. Fusion doesn't alter the mathematical formula \- it simply combines two kernels to overlap communication and computation, avoid writeback to main memory, and/or reduce kernel launch time. An example is Conv/ReLU fusion, which performs in-place ReLU instead of writing data to main memory and reading it back just for the cheap ReLU operation. In some situations, people also do GEMM/AllReduce fusion, by which partial output is immediately synchronized once computed in order to reduce latency.

An example of Llama3 implemented on GPU is shown in the following figure. Each rectangle is a kernel launched to GPU with necessary merging (represented by &) and fusion(represented by \+).  
![Visualization of Llama3 model implementation](/images/blog1/model.png)

**Quantization** aka low precision, is one of the most widely used techniques in ML system optimization and a great example of algorithm/hardware co-evolution. I won't spend time on how quantization works, but some general observation on the current trend:

1. Inference is usually one step ahead of training on lowering precision. As of May 2025, for inference FP4 is mostly de-risked and FP8 has become the mainstream, while for training, FP8 has shown early signs of success while BF16 is still the widely used precision.
2. Beyond the common weight quantization, many other places that can be quantized - e.g., activation, KV cache, gradients, communication bits, etc.
3. There is no clear answer where the low precision will stop at. Some works suggest going lower than 4 bits doesn't further help while BitNet shows 1.58bit works fairly well

**Sparsity** can be dated back to Yann LeCun's [Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) paper in 1989. In the early days, people were more focused on static sparsity, e.g. channel pruning, 2:4 sparsity. In the era of LLM, more evidences show that total parameters matters a lot and dynamic sparsity starts to get more traction. Examples are attention sparsity. 

Many optimizations are actually done at the model design phase, with hardware constraints in mind. In the early days, MobileNet was invented to alleviate the high computational density of convolution. Now in the era of LLM, bandwidth becomes the key bottleneck and we are seeing optimization like MoE and GQA/MLA/SSM in the architecture design. MoE could also be viewed as trained dynamic sparsity, and it should be pointed out that MoE and hard attention have many similarities if we view expert weights as activated tokens

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
| Sequence parallel | Parallelize layernorm and/or attention proj layers at sequence dimension | Augment tensor parallel. Low communication cost. |
| Context parallel | Parallelize all layers at sequence dimension | Improve latency for long context prefill. High communication cost.  |
| FSDP | Split FC layers weights | Overlaps computation and communication, but each shard still carries full computation. For training only.|
| ZeRO | Split optimizer states (stage 1), gradients (stage 2), and weights (stage 3) | for training only. |

For inference-time scaling, when the model size is not too large (\<100GB), the common practice is to scale up with tensor parallel to the point where communication latency becomes nontrivial, and then to scale out with pipeline parallel and data parallel. For very large models, e.g., DeepSeek V3, the optimal parallelism is a much more complex problem. That's why we are seeing tools like [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) to optimize the parallelism and etc.
The similar situation is also true for training-time parallelism, which is usually designed and hand-tuned specifically for model architecture and datacenter configurations.


#### Inference system optimizations

For inference, the most important throughput metric is total tokens per second (TPS) and the latency metrics are time per output token (TPOT) and time to first token (TTFT).

The no-brainer optimization is to avoid recomputation by KV cache, prefix caching, offloading, etc. Beyond that, **most of the throughput optimizations are just smart ways of batching**. Batching is the most effective way to improve GPU utilization. Continuous batching was designed to aggregate requests of different sizes. KV cache optimization, like paged attention, increases maximum batch size under fixed memory. Even latency optimization techniques like speculative decoding can be viewed as a form of batching.

Regarding latency, it is generally a trade-off with throughput adjusted by parallelism. Generally, if you want every single token to be generated faster, you need to sacrifice the number of tokens that are simultaneously generated (the concurrency). A trade-off curve was shown in GTC 2025. Somehow, quite a few friends complained to me this figure is difficult to understand.

![Image](/images/blog1/latency_throughput_tradeoff.png)

Optimizations to further push the pareto frontier include PD disaggregation and speculative decoding.
PD disaggregation has been adopted by serious LLM serving companies for a while and recently it starts to get supported by [SGLang](https://lmsys.org/blog/2025-05-05-large-scale-ep/) and [TRTLLM](https://nvidia.github.io/TensorRT-LLM/advanced/disaggregated-service.html).
Speculative decoding is a great example of algorithm and system codesign that will be discussed later.

#### Training system optimizations

A major challenge for training system is that it requires a fundamental redesign when training methodology changes. It has become a pattern that a new LLM generation usually requires an overhaul of the training system. 
Compared with inference, due to less concern on latency and faster evolving training methods and model architecture, I sometimes find training is not as deeply optimized as inference. (Disclaimer: I take no responsibility for this statement. There are exceptions like DeepSeek and MLPerf)

Pretraining used to be the center of focus. It operates at a very large scale of GPUs in a throughput-oriented manner, with Model Flops Utilization (MFU) being the key metric. At this scale, resiliency to failure becomes a critical measure to improve MFU beyond choosing the right parallelism. Examples methods are fault tolerance and async checkpointing, see more details at [NVIDIA Resiliency Extension](https://github.com/NVIDIA/nvidia-resiliency-ext).

Post-training was once viewed as a lesser important problem due to its small scale. The rise of Reinforcement learning (RL) changed this. RL poses many new system-level challenges beyond parallelism. I call it **parallelism and placement** problem, which has been discussed in [HybridFlow](https://arxiv.org/pdf/2409.19256):

1. Multiple models (actor, critic, reward model, etc) and their data dependency lead to such a dilemma that \- either colocate multiple models at the cost of precious GPU memory, or host models on separate GPUs at the cost of idle compute time.
2. Training and rollout (generation) are completely different workloads. Using a dedicated inference framework and resharding the weights every iteration could be essential.

There are additional challenges with Reinforcement Learning from Verified Reward (RLVR). The GPU clusters built for pretraining have high network bandwidth but relatively weak CPU. CPU execution of external reward signals could become a bottleneck.

Though designing a RL system encapsulates both training and inference frameworks, it is still within the scope of allocating datacenter resources. Therefore it is still categorized as the system layer. It would be interesting to see companies like OpenAI or robotics companies actually succeed in bridging the training loop with the Internet, the simulation world or the physical world.

### Exceptions of 3-layer abstraction

**Exception 1: dataflow architecture and Megakernel.** Some ASICs (e.g., SambaNova, Cerebras, Groq)'s programming models are data-centric, i.e., execute immediately when partial data is ready. With this design, a "kernel" in the usual sense could be a whole transformer block or even the full model (see [SambaNova talk at HotChips 2024](https://hc2024.hotchips.org/assets/program/conference/day1/48_HC2024.Sambanova.Prabhakar.final-withoutvideo.pdf)). This certainly provides an edge on latency, combining with the high bandwidth of on-chip SRAM, at the cost of ease of programming. Recently there is also effort on one gigantic kernel on GPU, e.g., [MegaKernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).

**Exception 2: speculative decoding.** Speculative decoding is a beautiful technique that stems from system insights (i.e., batching is faster than execution one by one), and is well-engineered with both ML and statistics knowledge to match the original model distribution perfectly. It shows the power of full-stack optimization and it is hard to classify it into any single optimization layer. I have no doubt that if transformer continues to prevail in the next ten years, speculative decoding will be a cornerstone just like speculative execution is for modern CPUs.

### Final words
Any system design is built on shifting sands, as the trade-offs between model architecture, physical constraints, and application requirements are constantly evolving. What works optimally today may become suboptimal tomorrow. Before LLM became the most important AI use case, 2-layer abstraction (without the system layer) is sufficient to describe most of the ML workloads. With the new training paradigm and application ecosystems, we may soon see the need of another abstraction layer beyond the 3-layer abstraction discussed in this article.


### Reference

1. [https://jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/)  
2. Semi-analysis: https://semianalysis.com/2024/09/03/the-memory-wall/  
3. AI and memory wall: [https://arxiv.org/pdf/2403.14123](https://arxiv.org/pdf/2403.14123)  
4. Tensor Core analysis: [https://zhuanlan.zhihu.com/p/638129792](https://zhuanlan.zhihu.com/p/638129792)  
   1. [https://arxiv.org/html/2402.13499v1](https://arxiv.org/html/2402.13499v1)  
5. [https://hazyresearch.stanford.edu/blog/2024-05-12-tk](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)  
6. [https://chatgpt.com/share/67eb86bc-3804-8009-be96-102499679fb9](https://chatgpt.com/share/67eb86bc-3804-8009-be96-102499679fb9)  
7. [https://en.wikipedia.org/wiki/File:High_Bandwidth_Memory_schematic.svg](https://en.wikipedia.org/wiki/File:High_Bandwidth_Memory_schematic.svg)  
8. [https://zhuanlan.zhihu.com/p/25529708891](https://zhuanlan.zhihu.com/p/25529708891)  
9. Thunderkitten: 2D tile, 1d vector: [https://hazyresearch.stanford.edu/blog/2024-05-12-tk](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)  
10. SambaNova hotchips talk: [https://hc2024.hotchips.org/assets/program/conference/day1/48_HC2024.Sambanova.Prabhakar.final-withoutvideo.pdf](https://hc2024.hotchips.org/assets/program/conference/day1/48_HC2024.Sambanova.Prabhakar.final-withoutvideo.pdf)  
11. CUDA GTC talk: https://www.nvidia.com/en-us/on-demand/session/gtc25-s72383/  
    1. ![][image4]  
    2. ![][image5]  
12. Deepseek inference system: [https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)  
13. Mooncake  
14. PD disaggregated  
    1. ![][image6]  
    2. ![][image7]
