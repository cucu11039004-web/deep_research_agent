# 450 行写一个 deep research agent，然后 trace 数据吓了我一跳

我用不到 500 行 Python 写了一个 deep research agent，跑通后跑了 5 个真实 query。聚合数据吓了我一跳——80% 的 run 触发了我自己加的 forced closure 兜底机制。我以为是模型能力不行，或者 max_steps 设小了。深入读 trace 后我才发现，真正的元凶在 pipeline 最上游，是 planner。

这篇文章讲我怎么用 < 500 行代码加一个朴素的 trace 系统，撞见了 agent 工程里一个被严重低估的 bug 类型。

## 起点：为什么是这件事

博士快毕业，下家是创业公司的人机交互工程师，中间有一段真空 gap。我对自己的判断是：当下这个 AI 时代，单纯的 LLM 难以充分发挥潜力，agent 才是真正释放 LLM 能力的形态。但我对 agent 的全局画面是空的，零散听过 LangChain、MCP 这些名词，没有自己写过一行 agent 代码。

我做了几个决策：

第一，选 deep research agent 作为起手项目。不是 coding agent，不是 browser agent，不是垂直 agent。理由是闭环最短：planner 加几个工具再加 writer，一个周末能跑通 MVP；每一层都能往深做（并行调度、context 压缩、citation tracking、eval）；MCP 天然契合；自己每天会用，正反馈最强。

第二，用 minimal harness，不用 LangGraph。虽然我已经速通过 LangGraph 一遍。理由是 deep research 本质上就是 harness 形态——planner 决定下一步搜什么，搜了，看结果，再决定。这是模型在驱动循环，不是代码在驱动流程。强行套 LangGraph 会把简单的事复杂化。更重要的是：必须先看见裸的 agent loop，否则永远只是 LangGraph 用户，不是真正懂 agent 的人。

第三，硬约束 < 500 行。这是一个简洁性的 forcing function，逼我把所有"这里可以再加点功能"的冲动延后到 Sprint 2/3。

## Agent 的形状：一个 50 行的 dumb loop

Agent 的本质，是 **LLM + Tools + Loop**。三件事缺一个都不是 agent：没有工具是 chatbot，没有循环是单次工具调用，没有 LLM 是脚本。

化学反应在哪里？在循环让 LLM 能看到自己工具调用的结果，并基于结果决定下一步。这个 "看到 → 决定 → 看到 → 决定" 的循环，是 agent 区别于一切其他形态的本质。

写成代码就是这个东西：

```python
def harness(user_query, tools, llm):
    messages = [{"role": "user", "content": user_query}]
    while True:
        response = llm.call(messages, tools=tools)
        if response.is_final_answer():
            return response.text
        for tool_call in response.tool_calls:
            result = execute(tool_call)
            messages.append(tool_result_message(result))
```

就这么多。Claude Code 的核心循环也是这个形状，Cursor 是，Aider 是。所有 harness 派 agent 都是这个形状。

我把这个心智凝结成一句话：**代码定义能力，模型决定意图**（Code defines capability, model decides intent）。我的工具函数定义了 agent 能做什么（capability），prompt 定义了它应该做什么的边界，但具体这次调几次、调什么参数、什么时候停，是 LLM 自己的事。

我的 deep research agent 把这个心智 scale 成一棵树：

```
                  [user query]
                       ↓
                   [planner]              ← 1 次 LLM 调用
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   [researcher 1] [researcher 2] [researcher 3]   ← 并发,每个独立 loop
        ↓              ↓              ↓
     [note 1]       [note 2]       [note 3]
        └──────────────┼──────────────┘
                       ↓
                   [writer]               ← 1 次 LLM 调用
                       ↓
                [final report]
```

这是 Anthropic 那篇 Building Effective Agents 里的 Orchestrator-Workers 模式。Planner 拆问题，多个 researcher 并发研究每一面，writer 综合。每个 researcher 是一次完整的 harness 循环，跑完只把浓缩后的 note 返回——这个设计让主流程的 context 不会爆炸，是 deep research 能 scale 的根本原因。

工具就三个：calculator、web_search（Tavily）、fetch_url（httpx + readability）。LLM 用 DeepSeek（已有 key，便宜，多轮调用最划算）。整套代码 < 500 行 Python。

## 第一次迭代：修了 AI 老师代码里的一个 bug

我跟着一份 sprint 计划手写每一段代码，写到 sub-agent 那一节，max_steps 触发时教程的代码这样写：

```python
return f"[Researcher reached max_steps={max_steps} on: {sub_question}]"
```

这一行看起来人畜无害——max_steps 是兜底嘛，触发了就返回个错误信息。但我盯着它看了几分钟，觉得不对劲。

为什么？因为 sub-agent 架构下，max_steps 触发的代价完全不一样——前面 N 次工具调用累积的所有信息全部蒸发。Writer 拿到一个 "[Researcher reached max_steps]" 字符串，根本没法用。整个 pipeline 因为一个 sub-agent 没收敛就部分残废。

我把这个修复称为 **forced closure**——预算耗尽时让 LLM 自己用一次额外调用，把 messages 里的事实压缩成 note：

```python
# max_steps 用尽时强制封笔
messages.append({
    "role": "user",
    "content": (
        "You have used your full search budget. Based ONLY on the tool "
        "results already in this conversation, write the final note now "
        "in the required format (ANSWER:/KEY_FACTS:/LIMITATIONS:). Do not "
        "call any tools. If the information is genuinely insufficient, "
        "say so clearly in LIMITATIONS rather than refusing to answer."
    ),
})
closing = await client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tool_choice="none",  # 硬约束:禁止再调工具
    temperature=0.3,
)
return closing.choices[0].message.content
```

两个细节值得注意。第一，`tool_choice="none"` 是 API 层的硬约束，不是靠 prompt 软求"你不要再调工具了"——能用 API 参数硬约束的，绝不靠 prompt 软约束。第二，让 LLM 自己做归纳而不是用代码硬拼 messages，因为只有 LLM 才知道 messages 里哪些是有用信息。

写下这段修复时我没意识到，它会在下一次迭代里救我的命。

## 第二次迭代：trace 数据吓了我一跳

跑通之后我没急着写博客，而是给 agent 加了一个朴素的 trace 系统——每次 LLM 调用、每次工具调用、每次 forced closure 触发，都写一条 JSON Line 到 `runs/<timestamp>_<run_id>.jsonl`。同时写一个 `analyze_trace.py` 做聚合分析。

跑了 5 个真实 query：

- Anthropic 最新模型是什么
- DeepSeek R1 和 V3 的主要区别
- 对比 DeepSeek R1、Qwen3、Claude Opus 4.7 在编程任务上的最新表现
- 为什么 Mamba 在长序列上比 Transformer 高效
- VLA 模型 2024-2026 主要进展

聚合输出是这样的：

```
Total runs: 5
Avg duration: 215.0s
Worst duration: 292.0s
Runs with forced closure: 4/5
Total tool warnings: 8
```

5 个 run 里有 4 个触发了 forced closure，触发率 80%。

我的第一反应是错的。我以为是模型不行，或者 max_steps 设小了。把 max_steps 从 8 调到 12？给 researcher 换个更强的模型？

幸好我没这么干。我打开了那个最反常的 trace 文件——`4ced9c7f`，那个 "DeepSeek R1 和 V3 的主要区别" 的 run，4 个 researcher 里 r2 和 r4 都强制封笔了。

Planner 把这个 query 拆成了 4 个 sub-question：

| ID | Sub-question | 步数 | 结局 |
|----|--------------|------|------|
| r1 | DeepSeek R1 和 V3 的**模型架构和参数规模**有何区别 | 3 | natural ✓ |
| r2 | DeepSeek R1 和 V3 的**训练方法和数据**有何不同 | 8 | **forced ✗** |
| r3 | DeepSeek R1 和 V3 在**性能基准（如代码、数学、推理）**上的表现差异如何 | 4 | natural ✓ |
| r4 | DeepSeek R1 和 V3 在**推理成本、响应速度和输出格式**上有何区别 | 8 | **forced ✗** |

我的第一个假设是 "包含多个 aspect 的 sub-question 容易爆"。r4 有 3 个 aspect、r2 有 2 个、r1 有 2 个、r3 有 1 个。看起来对得上。

但 r1 也是 2 个 aspect（架构 + 参数规模），r1 收敛了，r2 没收敛。同样 2 个 aspect，结局完全不同。

我把 r1 和 r2 的 token 增长曲线拉出来对比：

```
r1 (架构 + 参数规模, 3 步收敛):
  step 1: input=506,  output=82
  step 2: input=2308, output=87
  step 3: input=2596, output=900  ← 直接出 final note

r2 (训练方法 + 数据, 8 步耗尽):
  step 1-8: output_tokens 都在 50-100 之间,从来没进入"该写 note"的状态
```

r1 第 3 步 output 飙到 900，意味着模型决定 "我够了，写 note"。r2 每一步 output 都 50-100 tokens，意味着每一步都在调工具——它从来没有进入 "该写 note 了" 的状态。

为什么？r1 和 r2 的 multi-aspect 程度看起来差不多，但任务难度天壤之别：

- **r1（架构 + 参数规模）**：架构和参数规模在 R1/V3 的官方文档里几乎共现——同一段话同时提到。一次搜索 + 一次 fetch 就拿到两个 aspect。
- **r2（训练方法 + 数据）**：训练方法（SFT、RL、reward model）和训练数据（语料组成、清洗策略）是两个独立维度，往往在不同的页面讲。模型搜了"训练方法"，要再搜"训练数据"。

这就引出了比 multi-aspect 更精确的诊断：

> **真正决定 sub-question 能否收敛的，不是 aspect 数量，而是覆盖所有 aspect 所需的"独立信息源"数量。**

我的 planner system prompt 里写的是 "Each sub-question focuses on ONE aspect"。这个约束在工程上是错的。"ONE aspect" 太抽象，模型会按字面理解（语法上 "X 和 Y 在 a/b/c 上有什么区别" 是一个问题）。**正确的约束应该是 "ONE independent information source per question"**。

更进一步的二阶发现：我把 4 个 fetch_url warning 对齐到具体 researcher，发现 r2 和 r4 各有 2 个 warning，且每个都是同一个 URL 出现两次。模型第一次 fetch 拿到 warning 后，没有"长期记忆"，messages 越长，"我刚才试过这个 URL 失败了"的信息在中段被弱化，几步之后又选择同一个 URL 重试。

这就是经典的 lost-in-the-middle 在 agent 行为上的物理表现——不是模型变笨，是它读不到自己 5 步前的失败记录。这种 bug 在 print 输出里你压根看不见，必须 trace 才能浮出水面。

## 退一步：trace 系统的工程价值

整篇文章里所有的"洞察"都依赖一件事——我有 trace 系统。没有它，这两个 finding 一个都不会出现。

Print 不是 trace 的弱化版，是完全不同的东西。Print 服务的是单次 debug——agent 跑完一遍我看到了过程。Trace 服务的是积累——结构化、可回放、可聚合、可对比。一个完整 run 的 trace 大概 50KB，跑了 100 次的 100 个 jsonl 文件就是 dataset 雏形。半年后你回来看 Sprint 1 的 trace 和 Sprint 3 的 trace，进步会让你自己惊讶。

Anthropic 那篇 effective context engineering 里有一句话我深以为然：70% 的企业 AI 工作其实是 observability，但大多数学习材料根本不教。能写 LLM-as-judge、能搭 regression eval、能做 trace replay 的人非常稀缺。

我的 tracer.py 简单到不能再简单——一个全局单例，几个 log 方法，写 JSONL 文件。50 行 Python。但就是这 50 行，把这个 agent 从"能跑"提升到了"能学习"。

## 没修的两个 bug，留给 Sprint 2/3

我没在 Sprint 1 修这两个 bug。理由不是没时间，是 Sprint 1 的目标是端到端跑通 + 有传播力的对外输出。修 prompt 涉及多次实验验证，是 Sprint 3 "harness 升级" 的范畴。

但两个 bug 的修复方向我已经写在 repo 的 findings.md 里：

**Finding 1（planner）的修法**有两个层次。Cheap 的版本：把 prompt 里的 "ONE aspect" 改成 "ONE independent information source"，并加 anti-example（"BAD: X 和 Y 在 a, b, c 上的区别 / GOOD: X 和 Y 在 a 上的区别 + ... + 在 c 上的区别"）。Proper 的版本：让 planner 拆完之后做一次自反思（self-reflection），对每个 sub-question 评估 "is this single-source?"，是 ACE（Agentic Context Engineering）那条路线。

**Finding 2（researcher）的修法**：在每次 LLM call 之前往 system prompt 注入 "你已经搜过 X、fetch 过 Y、其中 Z 失败了" 的状态摘要。这是 Claude Code 那种生产级 harness 的标准做法之一。

把这两个 bug 留在 repo 里，比我硬修了它们更有价值——它们是 Sprint 3 的 backlog，是博客 to be continued 的钩子，也是下一篇博客的具体素材。

## 一些我没在文章里展开的事

Sprint 1 还有一些发现没写进主线，列在这里：

- LLM 是 stochastic system，不是 deterministic system。同样 query、同样代码、同样 temperature=0，5 次 run 的步数会有微小波动。Agent 工程师的工作是降低方差，不是消除方差。
- Tool description 是给模型读的，不是给人读的。我把 fetch_url 的 description 加了一句 "If you get 'Warning: short content', that URL is dead. Try a different one"，worst case 步数从 8 降到 6。改了一行警告，节省了 30% 的 LLM 调用。
- Agent 不能等于"用了一个酷的框架"。我故意没用 LangChain 的 @tool 装饰器，因为那个装饰器藏起了 schema 这件事——你看不见模型看见了什么。手写 schema 一遍，下次调试 agent 时就会本能地去看 schema 哪里写得不对，而不是去换模型。

## 下一步

Sprint 1 结束。Sprint 2 是学术专精版本——把这个通用 deep research 改造成接 ArXiv / Semantic Scholar / OpenReview 的学术 deep research，输出包含论文清单、主线综述、关键 baseline 表、GitHub 复现可行性的报告。我的博士经验在这个方向是天然护城河。

Sprint 3 把它包成 MCP server，让 Cursor / Claude Desktop / Cline 都能直接调用。同时验证 MCP server 作者这个 2026 年的硬通货标签。

Sprint 4 接 Deep Research Bench，跑一遍看在 leaderboard 什么位置。70% 的 agent 工作是 observability，跨过去就和 80% 的 agent 工程师拉开差距。

如果 planner 的 bug 比 researcher 隐蔽 10 倍，那么真正生产级 agent 的 observability 应该怎么设计？这是 Sprint 2 / 3 我会回来回答的问题。

代码在这里：[GitHub repo link]

---

*这是我"AI 时代工程师"养成档案的 Sprint 1。下一篇见。*
