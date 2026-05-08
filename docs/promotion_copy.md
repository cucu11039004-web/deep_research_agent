# Sprint 1 推广文案合集

四个渠道的钩子文案，按渠道分别准备。直接拷贝使用。

---

## 1. Twitter / X（英文）

```
I built a deep research agent in <500 lines of Python.

Then I added a JSONL trace system. Data showed 80% of runs hit my fallback mechanism.

I expected: "model not strong enough" or "max_steps too low".

Real cause: a planner bug. The prompt said "one aspect per sub-question". 
Should have said "one independent information source per sub-question". 

Same number of aspects, different convergence behavior — depending on whether the aspects co-occur in one document or live on different pages.

Full write-up + code: [link]

#LLM #AIagents #DeepResearch
```

---

## 2. 知乎 / 小红书（中文）

```
博士 gap 期，450 行手写了一个 deep research agent。

跑通后跑了 5 个真实 query，trace 数据吓了我一跳——80% 的 run 都触发了我自己加的 forced closure 兜底。

第一反应:模型不行、max_steps 太小。
错。

读完 trace 才发现真正的元凶在 pipeline 最上游——planner。我的 prompt 写的是 "每个 sub-question 关注 ONE aspect",这个约束在工程上是错的。
正确的约束应该是 "每个 sub-question 应该能从 ONE independent information source 找到答案"。

同样 2 个 aspect 的两个 sub-question,一个 3 步收敛、一个 8 步爆掉。差别在于 aspect 是否共现在同一份文档里。

完整文章 + 代码: [link]
```

---

## 3. 朋友圈（中文，短钩子）

```
Sprint 1 收工:450 行 deep research agent + 一个发现深层 bug 的 trace 系统 + 这篇 4000 字的"工程派"博客。

不是教程,是故事——我修了 AI 老师的代码 bug,然后用 trace 数据撞见了 agent 工程里被严重低估的一类 bug。

[link]
```

---

## 4. Hacker News 标题（英文）

```
Show HN: 450-line deep research agent and what JSONL traces revealed about my planner
```

提交 HN 时附带的 first comment（解释你的 motivation，HN 高赞 Show HN 通常都有这种 comment）：

```
Built this during a gap period before joining a startup. Goal was to learn agent engineering hands-on without leaning on LangGraph or any heavy framework — minimal harness, hand-written tool schemas, JSONL tracing.

The interesting part wasn't getting it to work. It was what the trace data showed afterwards: 80% of runs hit my forced-closure fallback. I assumed model issue or wrong max_steps. The actual root cause was a planner-prompt bug that's invisible without trace analysis.

Two unfixed bugs documented in findings.md as the backlog for the next sprint. Happy to discuss the design choices.
```

---

## 5. r/MachineLearning（英文，发深度版本）

标题：

```
[D] Trace analysis on a from-scratch deep research agent revealed a planner bug I'd never seen documented
```

正文：

```
Spent a few weeks building a minimal deep research agent (<500 lines, harness-style, no LangGraph) as part of a self-study sprint. Added JSONL tracing from day one.

Trace stats over 5 real queries:
- Avg duration: 215s
- 80% of runs triggered my forced_closure fallback (max_steps exhausted)
- 8 fetch_url quality warnings, half of which were the same URL hit twice in one run

My initial diagnosis was wrong (model capacity / max_steps). The real cause turned out to be in the planner's decomposition: the prompt told it to ensure "one aspect per sub-question", but the actual convergence behavior depended on whether the aspects could be answered from a single information source or required multiple independent sources.

Concrete example: "DeepSeek R1 vs V3 — architecture and parameter scale" converges in 3 steps because both aspects live in the same model card. "DeepSeek R1 vs V3 — training methodology and training data" hits max_steps every time, because methodology and data composition live on different pages.

The fix isn't deployed yet (it's the backlog for sprint 3). Posting this because:
1. The "single source" abstraction seems more useful than "single aspect" for planner design, and I haven't seen it discussed
2. The bug is invisible without per-step tracing — the symptom looks like "agent is bad at hard questions"

Code + full writeup: [link]

Curious if anyone else has hit this and how you handled it.
```

---

## 发布顺序建议

```
Day 0:
  - 个人博客发布(主仓)
  - 朋友圈发钩子(种子读者)

Day 1:
  - 知乎发布(中文长尾流量)
  - Twitter/X 发布(英文圈子)

Day 2-3:
  - HN 提交(挑工作日早上 PT 时间提交,周二周三最佳)
  - r/MachineLearning 发布(挑周末发,讨论氛围最浓)

Day 5-7:
  - 收集反馈,如果有有价值的评论,做一次"补充澄清"短帖子
```
