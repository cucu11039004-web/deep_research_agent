# Findings from Sprint 1

Two bugs / design issues discovered via trace analysis but not yet fixed.
Documented here as the backlog for Sprint 2/3.

The data behind these findings: 5 real queries, 18 sub-agent runs, 80% forced closure rate.

---

## Finding 1: Planner's "ONE aspect" rule fails on multi-source questions

### Symptom

In a 5-query test, **80% of runs trigger forced_closure on at least one researcher**. Specifically:

```
Query                                              | sub-Qs | forced
---------------------------------------------------+--------+-------
Anthropic 最新模型是什么                           | 2      | 0
DeepSeek R1 和 V3 的主要区别                       | 4      | 2
对比 R1/Qwen3/Opus 4.7 在编程任务上的表现          | 5      | 3
为什么 Mamba 在长序列上比 Transformer 高效         | 4      | 1
VLA 模型 2024-2026 主要进展                        | 4      | 2
```

### Investigation

I picked the most reasonable-looking failure case (`4ced9c7f`, "DeepSeek R1 vs V3"). Planner output:

| ID | Sub-question | aspects | result |
|----|--------------|---------|--------|
| r1 | 模型架构和参数规模 | 2 | natural, 3 steps |
| r2 | 训练方法和数据 | 2 | **forced, 8 steps** |
| r3 | 性能基准（代码/数学/推理）| 1 | natural, 4 steps |
| r4 | 推理成本、响应速度、输出格式 | 3 | **forced, 8 steps** |

The "multi-aspect → fail" hypothesis seemed obvious. But r1 also has 2 aspects and converged in 3 steps.

Token-level comparison:

```
r1 (架构 + 参数规模, converged):
  step 1: input=506,  output=82
  step 2: input=2308, output=87
  step 3: input=2596, output=900  ← decided to write the note

r2 (训练方法 + 数据, forced):
  step 1-8: output_tokens stay 50-100, never enters "write note" state
```

### Root cause

It is not aspect count. It is **the number of independent information sources required to cover all aspects**.

- r1: 架构 and 参数规模 co-occur in the same DeepSeek paper / model card. One fetch covers both.
- r2: 训练方法 (SFT/RL/reward model) and 训练数据 (corpus composition / cleaning) live on **different pages**. The researcher has to do two independent search-fetch cycles.

The planner prompt currently says:

```
- Each sub-question focuses on ONE aspect.
```

"ONE aspect" is the wrong abstraction. The right abstraction is:

```
- Each sub-question should be answerable from ONE independent information source.
```

### Proposed fix (Sprint 3)

**Option A — cheap, prompt-only**:

```
- Each sub-question should be answerable from a single source/article.
- BAD: "X 和 Y 在 a, b, c 上的区别"  (likely needs multiple sources)
- GOOD: "X 和 Y 在 a 上的区别" + "X 和 Y 在 b 上的区别" + ...
- If a question contains multiple aspects connected by 和/以及/commas
  AND the aspects span different document types (e.g. training methodology
  vs training data), split each aspect into its own sub-question.
```

Few-shot anti-examples are the key — models absorb "BAD: ... GOOD: ..." much better than negative imperatives alone.

**Option B — proper, two-pass**:

After planner outputs N sub-questions, run a second LLM call: "For each sub-question, answer: would a single web article answer this completely? If no, split further." Costs one extra LLM call per query but is far more robust. This is the ACE (Agentic Context Engineering) pattern.

### Verification plan

Re-run the same 5 queries after each fix. Target: forced_closure rate drops from 80% to < 30%.

---

## Finding 2: Researcher repeats failed fetch_url calls

### Symptom

In the same `4ced9c7f` run, fetch_url tool warnings:

```
2 warnings from r2 → emergent.sh/learn/deepseek-r1-vs-v3 (the SAME URL twice)
2 warnings from r4 → hiberus.com/.../deepseek-r1-vs (the SAME URL twice)
```

Each researcher fetched its dead URL twice within a single run, several steps apart.

### Root cause

Classic lost-in-the-middle. messages array gets long, attention weight on mid-conversation tool results decays, the model forgets it tried URL X 5 steps ago and got "Warning: short content".

### Proposed fix (Sprint 3)

Inject a "context summary" into the system prompt at every iteration:

```
You have already:
- Searched: ["query1", "query2", ...]
- Fetched (successful): ["url1", "url3"]
- Fetched (failed/dead): ["url2"]   ← model needs to see this prominently

Do not repeat any of the above unless you have a strong reason.
```

This is the same pattern Claude Code uses to avoid re-reading the same file three times.

### Verification plan

Track `repeated_failed_fetch_count` per run. Target: drop from current ~2 per complex run to 0.

---

## Open questions for Sprint 2 / 3

- Is "single information source" itself a stable enough concept for prompt engineering, or does it need further refinement?
- Should the planner have access to a quick web search to verify "is this question single-source-answerable?" before finalizing the plan? (Trade-off: cost vs accuracy)
- For domain-specific deep research (Sprint 2: academic), the "single source" rule is much stricter — a paper's contributions and its experimental setup live in the same PDF, but its citations live elsewhere. How do we encode this?
