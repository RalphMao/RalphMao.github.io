---
layout: post
title: The Next 1000x Cost Saving of LLM 
tags: [LLM, System]
---

LLM inference costs have dropped a lot over the past three years. At a comparable quality to early ChatGPT, average per-token prices are ~1000x lower, as observed in [an a16z blog](https://a16z.com/llmflation-llm-inference-cost/). That drop has been driven by advances across the stack: 
better GPUs, quantization, software optimizations, better models and training methods, and open-source competition driving down profit margins.

While many of the past optimizations look like low-hanging fruits, people may doubt that costs will keep dropping so fast, just like the long-running doubts about Moore's law. The good thing is that it's easier to predict cost trends than to predict the intelligence trends of LLMs.

In this article, I argue that **we’ll see another 1000x cost reduction in LLM applications over the next 3 years**. But the metrics will shift - not just by “input tokens” and “output tokens.” Token-based pricing will persist, yet the economic model must evolve to capture the emerging needs.

When we look at an optimization problem on a 3‑year horizon, everything changes: the models, the systems, and how people use them.
LLM workloads will not scale in a simple, proportional way. The cost of today’s workloads may not fall by 1000x; however, if you take the dominant workload 3 years from now, you will find it is 1000x more expensive on today’s stack.
<figure>
  <img src="/images/token-cost/workload_future_and_now.png" alt="">
</figure>

In 2024, there was an estimate that even if all 8 billion people on Earth talked to a chatbot for 30 minutes every day, the corresponding traffic could be served by on the order of 100k GPUs.
In 2025, we are seeing reasoning models and coding agents becoming a major source of token consumption. Compared with chatbots, agents are not just applications that burn more tokens; they exhibit fundamentally different workload patterns.


## Predict the shift of workload patterns

**Trend 1: longer context and horizon**. Longer context means not only more tokens but also higher cost per token; see, for example, the [DeepSeek per-token cost curve](https://api-docs.deepseek.com/news/news250929).

Longer horizon means more steps and longer runtimes. 
 - From an API perspective, cached tokens incur costs that grow quadratically with the number of steps, as illustrated in the following figure. Cache reads already account for more than half of the overall cost of coding tools like Cursor or Claude, one example reported in [this post](https://www.reddit.com/r/ClaudeAI/comments/1m53292/remember_the_fact_that_most_of_your_usage_is/). 
 - From a systems perspective, cached tokens are just KV cache. Keeping KV cache alive for longer implies a higher memory-to-compute ratio for the serving system.

<figure>
  <img src="/images/token-cost/multi_round_cache_token_explain.png" alt="">
  <figcaption>An agentic task consists of multiple LLM calls separated by tool calls and user interactions. Cached tokens grow linearly with the number of LLM calls, while total cache reads grow quadratically.</figcaption>
</figure>

Over the next 3 years, we may witness an evolution from single-task agentic LLMs to always-on, long-lived agents, which will put much greater stress on current systems. Anthropic is among the first to emphasize long-running capabilities in their models and to support a [1-hour cache lifespan](https://www.anthropic.com/news/claude-4) in their API.

**Trend 2: wider execution paths**, represented by [asynchronous tool calling](https://arxiv.org/pdf/2412.07017), [parallel thinking](https://blog.google/products/gemini/gemini-2-5-deep-think/), specialized models, or [asynchronous self-calling](https://arxiv.org/pdf/2504.15466v1).  

Allowing an LLM to perform multiple tasks concurrently and scale out its “thinking” can significantly enhance capability while reducing time-to-result. This will make a purely token-based pricing model harder to sustain in the long term. Agent providers like [Manus](https://manus.im/pricing) have already recognized this and price in “credits” instead of tokens, to better reflect total compute cost.

**Speculation: training as part of serving**. Inference is already part of training in reinforcement learning, and there is no fundamental systems barrier to incorporating some sort of training into inference serving. After all, intelligence requires adaptation to the environment, and today’s adaptation is mostly limited to context engineering. There are active efforts to address fundamental issues like catastrophic forgetting in [Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/), as well as approaches like [LoRA](https://thinkingmachines.ai/blog/lora/) to reduce the system cost.


## Forecast the next 1000x
Now the question is: if we predict the dominant workload pattern in 3 years, instead of focusing on the dominant workload today, where will the major cost savings come from?  

The baseline is set as the [DeepSeek V3/R1 inference system](https://github.com/deepseek-ai/open-infra-index#day-6---one-more-thing-deepseek-v3r1-inference-system-overview), which features FP8, Wide-EP, and PD disaggregated serving. With ~80% MFU for prefill and ~15% MFU for decode, DeepSeek's inference efficiency still outperforms most US model providers, despite being released almost a year ago.

I made some bold predictions in the table below and will explain them in more detail.

| Optimization                                                         | Improvement <br> measured by today's applications | Improvement <br> measured by future applications |
| -------------------------------------------------------------------- | ------------------------------- | ------------------------------ |
| Memory technology                                         | 2x                              | 2x                             |
| Process node & microarchitecture<br> (on top of memory improvement) | 1.25x                           | 2x                             |
| Heterogeneous system scaling                                         | 2.5x                              | 5x                             |
| Model & algorithm                                                    | 4x                              | 10x                            |
| Agent scaffold                                                       | 4x                              | 10x                            |

**Memory** capacity and bandwidth are currently the biggest bottlenecks in LLM serving systems. Among all memory types, HBM is the most important and is [projected](https://newsletter.semianalysis.com/p/the-memory-wall) to advance two generations (HBM3e → HBM4 → HBM4e) in 3 years, bringing roughly a 2x bandwidth and capacity improvement.

**Semiconductor process node & microarchitecture** improvements, measured by FLOPS, will likely outpace memory improvements. However, most of today’s workloads are memory-bound, so increased FLOPS would not help significantly (estimated as 1.25x). Future applications with wider execution paths and hardware-friendly model design will better exploit these gains.

**Heterogeneous system scaling** refers to scaling the inference serving system up and out with specialized chips and different node configurations. One example is [Rubin CPX](https://newsletter.semianalysis.com/p/another-giant-leap-the-rubin-cpx-specialized-accelerator-rack), a prefill-optimized GPU that can substitute for high-end HBM GPUs at lower cost. Beyond prefill-specialization, future applications will create more opportunities for specialized chips and nodes.

Some may wonder where compilers and system software fit in this table. They are embedded in microarchitecture and heterogeneous systems. New architectures will be harder to optimize with the current compiler stack and demand significant innovation to maintain utilization. Likewise, larger heterogeneous clusters will require substantial software work for orchestration.

**Model & algorithm** innovations have historically contributed the most to cost reduction ever since the CNN era.

- Better data quality and smarter training methods will continue to boost model capability and thus reduce effective cost. Recent advances like on-policy distillation and training to reduce traces are some good exmaples.
- Efficient model architectures, especially sparse/linear attention, will reduce cost modestly for today’s workloads and dramatically for future long-context, long-horizon applications.
- On quantization, many providers nowadays serve models in 8-bit formats, while some with inferior capability still run bf16. 4-bit formats such as NVFP4 and MXFP4 (also see [my previous quantization blog](/quantization/)) are likely to be widely deployed within 3 years.

Overall, I’m comfortable predicting at least a 10x gain from models and algorithms alone.

**Agent and its scaffold** is an emerging area driven by application needs. It designs how the LLM interacts with the environment, other LLMs, and itself.

- Interface is critical. While researchers train models to better operate on real-world APIs, developers are also learning that not all interfaces are equally effective or token-efficient for LLMs. [Anthropic’s recent blog](https://www.anthropic.com/engineering/code-execution-with-mcp) shows that changing the interface can reduce token usage by 98.7%.
- Memory management, specifically, managing short-term memory (context window) and persistent memory, is another important lever.
- Multi-agent system with specialization could further reduce the overall computation cost.

There are many opportunities in this space, and it could easily deliver another 10x cost reduction.

**Summary** – combining the optimizations above, we arrive at an estimated ~2000x efficiency improvement. On the other hand, the cost of HBM modules and leading-edge chip area will likely increase by 1.5–2x, depending on manufacturing costs and demand.


## Final words

Good optimizations cut today’s costs. Great optimizations unlock tomorrow’s needs.
