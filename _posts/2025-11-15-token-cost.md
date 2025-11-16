---
layout: post
title: The Next 1000x Cost Saving of LLM 
tags: [LLM, System]
---

LLM inference costs have fallen roughly 1000x over the past three years. At a comparable quality to early ChatGPT, average per-token prices are ~1000x lower, as [observed by a16z](https://a16z.com/llmflation-llm-inference-cost/). That drop has been driven by advances across the stack: better data and distillation; quantization and sparsity; optimized kernels/compilers and serving systems; larger effective batch sizes and KV-cache reuse; and denser, cheaper hardware with HBM and interconnects.

While the many of the past optimizations look like low-hanging fruits, people may doubt that costs will keep dropping, just like the long-running doubts about Moore’s law. The good thing is that it’s easier to predict cost trends than to predict the intelligence trends of LLMs.

In this article, I argue we’ll see another ~1000x reduction in effective LLM application cost over the next 3 years. But the metrics will shift - not just “input tokens” and “output tokens.” Token-based pricing will persist, yet the economic model must evolve to capture the emerging needs.


When we look at an optimization problem in a 3-year horizon, everything including the model and the way people use models will change. 
LLM workload will not trivially and proportionally scale. The cost of today’s workload won’t reduce by 1000x, but if you look at the dominant workload in 3 years and look back, you may find it 1000x more expensive to run with today's technology. 
<figure>
  <img src="/images/token-cost/workload_future_and_now.png" alt="">
</figure>

In 2024, I saw an estimation that even if all the 7 billion people around the world talk to a chatbot for 30 minute every day, the corresponding traffic could be easily served by 100k-ish GPUs.
In 2025, we are seeing all types of agents, especially coding agents, becoming a major source of token consumption. Compared with chatbot, agents is not simply an application that consumes more tokens. It presents a much different workload pattern.


## Predict the shift of workload patterns

**Trend 1: longer context and horizon** of LLM inference.
Longer context means not only more tokens but also higher cost per token, for example, see the [DeepSeek per-token cost curve](https://api-docs.deepseek.com/news/news250929).

Longer horizon means more steps and longer time.
From API perspective, cached tokens have quadratic costs to the number of steps, as illustrated in the following figure. Cached token read has already taken more than half of the overall cost of Cursor or Claude code, e.g. as reported in [this post](https://www.reddit.com/r/ClaudeAI/comments/1m53292/remember_the_fact_that_most_of_your_usage_is/).
From system perspective, cached token is just KV cache. Longer lasting time of KV cache implies higher memory to compute ratio required for the system.

<figure>
  <img src="/images/token-cost/multi_round_cache_token_explain.png" alt="">
  <figcaption>An agentic task consists multiple LLM calls separated by tool calling and user interactions. Cached tokens grows linearly with LLM calls while total cache reads grows quadratically.</figcaption>
</figure>

Over the next 3 years, we may witness the evolution of single-task agentic LLM to forever-running agents, which will pose a greater stress on the current systems. Anthropic is among the first to emphasize the long running capability of their models, and the first to support [1-hr cache lifespan](https://www.anthropic.com/news/claude-4) in their API.

**Trend 2: wider execution paths**, represented by [asynchronous tool calling](https://arxiv.org/pdf/2412.07017), [parallel thinking](https://blog.google/products/gemini/gemini-2-5-deep-think/), specialized model calling, or [asynchronously calling itself](https://arxiv.org/pdf/2504.15466v1).
Allowing LLM to simultaneously perform multiple tasks and scale out its thinking power could greatly enhance its capability, meanwhile reduce the time to results.

This will make the token-based pricing model more difficult to sustain in the long term.
Agent providers like [Manus](https://manus.im/pricing) has already realized this issue and they are not priced by tokens, but "credits" which considers the overall compute cost as a whole.


**Speculation: training being a part of serving**. Inference is already part of training in Reinforcement Learning, and it is not a fundamental issue from system perspective to incorporate some sort of training in inference serving. After all, intelligence requires adaptation in environments, and the current way of adaptation is only context manipulation. There are already efforts to solve fundamental issues like forgetting in [Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/), and also approaches to alleviate system cost like [LoRA](https://thinkingmachines.ai/blog/lora/).


## Forecast the next 1000x cost saving
Now the question is, if we predict the dominant workload pattern in 3 years, instead of looking at the most dominant workload now, what would be the major cost savings? 
I made some bold predictions in the table below, and will explain it in more details. 

| Optimization                                                         | Improvement <br> -measured by today's applications | Improvement <br> -measured by future applications |
| -------------------------------------------------------------------- | ------------------------------- | ------------------------------ |
| Memory process                                                       | 2x                              | 2x                             |
| Chip process & microarchitecture<br>-on top of memory improvement | 1.25x                           | 2x                             |
| Heterogeneous system scaling                                         | 2x                              | 5x                             |
| Model & algorithm                                                    | 2x                              | 10x                            |
| Agent scaffold                                                       | 2x                              | 10x                            |

**Memory** capacity and bandwidth is currently the biggest bottleneck in LLM serving system. Of all different memory types, the most important HBM is [projected](https://newsletter.semianalysis.com/p/the-memory-wall) to advance 2 generations (HBM3 -> HBM3e -> HBM4) in 3 years and brings approximately 2x bandwidth improvement.

**Chip manufacture process & Microarchitecture** advances measured by FLOPs will be larger than memory improvement. However, the today's workload is more memory bottlenecked, therefore 
Wider execution with increased concurrency could also better leverage this gain

**Heterogeneous system scaling** means scaling up/out the inference serving system with different specialized chips and node configurations. One of the example is [Rubin CPX](https://newsletter.semianalysis.com/p/another-giant-leap-the-rubin-cpx-specialized-accelerator-rack), a specialized prefilling chip as a cheaper substitute to the high-end HBM chip.
Future agentic applications would enable more such opportunities like cross-node hierachical KV cache offloading

Some may wonder where compiler and system software sit on this table. They are parts of microarchitecture and heterogeneous system. Newer architecture will be more difficult to optimize with current compiler stack, and require great innovations to maintain the current utilization level. Similarly, larger scales of heterogeneous hardware system requires significant work from software to orchestrate.

**Model & algorithm** have historically been the greatest factor of cost reduction since the CNN era. 

 - Better data quality and smart ways to train models will keep elevating model intelligence thus saving costs. We have seen recent advances like on-policy distillation, training to reduce reasoning lengths that grealy improves
 - Efficient model architecture, especially the use of sparse/linear attentions, will drive the cost down a little bit for today's applications and a lot for future usecases.
 - Quantization-wise, many model providers now serve models in 8-bit formats, while some labs with inferior system capability still stick with bf16. 4-bit formats, represented by NVFP4 and MXFP4 (see [my previous quantization blog](/quantization/)), will expected to be widely used in 3 years.
Overall, I will comfortably predict at least 10x from model and algorithm.

**Agent and its scaffold** is an emerging field largely driven by application needs. It determines the interaction between LLM and the environment, and also the LLM with other LLMs including itself. 
- Interface is critical. While researchers are training models to better manipulate the interfaces of the real world, developers also realize not all interfaces are effective and efficient for LLM to operate on. [Anthropic's recent blog](https://www.anthropic.com/engineering/code-execution-with-mcp) suggests changing the interface could reduce token usage by 98.7%.
- Memory management, specifically, managing model's short-term memory (context window) and persisting memory is also an important direction to work on.

There are a lot of opportunities in this new field and I have no doubt it could easily bring another 10x cost saving.

**Summary** - with the aforementioned optimizations, we arrive at an estimation of 2000x efficiency improvement. On the other hand, the cost of a HBM module and latest-node chip area will likely increase by 1.5-2x, depending on the manufacture cost and rising demand.


## Final words

Good optimizations save the costs of today. Great optimizations enable the needs of future.


Reference:


DeepSeek infra, memory offloading (disaggregated):
https://github.com/deepseek-ai/open-infra-index

Memory offloading (Single node):
https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/
And TRTLLM already supports it
