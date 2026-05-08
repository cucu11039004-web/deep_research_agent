"""
第 4 课:Sub-Agents & Orchestration
端到端的 deep research agent。
形状: planner → N 个并发 researcher → writer
"""

import os
import json
import math
import asyncio
from openai import AsyncOpenAI  # ★ 注意是 AsyncOpenAI
from dotenv import load_dotenv
from tavily import TavilyClient
import httpx
from readability import Document
from tracer import tracer  # ★ trace
import time

load_dotenv()

# ★ 用异步客户端
# client = AsyncOpenAI(
#     api_key=os.environ["DEEPSEEK_API_KEY"],
#     base_url="https://api.deepseek.com",
# )
# MODEL = "deepseek-v4-pro" # deepseek-v4-flash deepseek-chat

# client = AsyncOpenAI(
#     api_key=os.environ["MINIMAX_API_KEY"],
#     base_url="https://api.minimaxi.com/v1",
# )
# MODEL = "MiniMax-M2.7" # MiniMax-M2.7

client = AsyncOpenAI(
    api_key=os.environ["GLM_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)
MODEL = "glm-5.1" # glm-5.1 glm-4.7

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ============================================================
# 1. 工具实现 (大部分照搬第 3 课,但 fetch_url 改成异步)
# ============================================================

def calculator(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str, max_results: int = 5) -> str:
    """Tavily 是同步 SDK,但很快,在 async 里直接调用问题不大"""
    try:
        response = tavily.search(query=query, max_results=max_results, search_depth="basic")
        results = [
            {"url": r["url"], "title": r["title"], "snippet": r.get("content", "")[:300]}
            for r in response.get("results", [])
        ]
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: web_search failed: {e}"


async def fetch_url(url: str, max_chars: int = 4000) -> str:
    """★ 改成 async,用 httpx.AsyncClient"""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as ac:
            response = await ac.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchAgent/0.1)"},
            )
            response.raise_for_status()
        
        doc = Document(response.text)
        title = doc.title()
        content = doc.summary(html_partial=True)
        import re
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
        
        if len(text) < 200:
            return (
                f"Warning: extracted content from {url} is suspiciously short "
                f"({len(text)} chars). Likely SPA, paywall, or homepage. "
                f"Try a different URL — preferably a direct article URL."
            )
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Content truncated at {max_chars} chars]"
        return f"# {title}\n\n{text}"
    except Exception as e:
        return f"Error: fetch_url failed: {e}"


# Sub-agent 的工具集——只有 web_search 和 fetch_url
RESEARCHER_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Returns JSON list of "
                "{url, title, snippet}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, 1-6 keywords"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch full article content from a URL. Use AFTER web_search. "
                "If 'Warning: short content' returns, the URL is dead — try a different one."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


# ============================================================
# # 异步 Tool dispatch + ★ trace 插桩
# ============================================================
async def dispatch_tool(name: str, args: dict, researcher_id: str = "main") -> str:
    """每次工具调用都记 trace。"""
    t0 = time.time()  # ★ trace

    if name == "web_search":
        result = web_search(**args)  # 同步函数,直接调
    elif name == "fetch_url":
        result = await fetch_url(**args)  # 异步函数,await
    elif name == "calculator":
        result = calculator(**args)
    else:
        result = f"Error: unknown tool '{name}'"
    
    duration_ms = int((time.time() - t0) * 1000)  # ★ trace

    # ★ trace: 记 tool 调用结果
    tracer.log(
        "tool_result",
        researcher_id=researcher_id,
        tool=name,
        args=args,
        result_chars=len(result),
        result_preview=result[:200].replace("\n", " "),
        is_warning=result.startswith("Warning:"),
        is_error=result.startswith("Error:"),
        duration_ms=duration_ms,
    )

    return result


# ============================================================
# 2. Planner +  ★ trace
# ============================================================

PLANNER_SYSTEM_PROMPT = """\
# Identity
You are a research planner. Decompose a user's question into 2-5 focused 
sub-questions that, together, fully address the original.

# Decomposition principles
- Each sub-question must be INDEPENDENTLY researchable.
- Each sub-question focuses on ONE aspect.
- Cover the full scope. Don't leave gaps.
- Simple "what is X" → 2 sub-questions.
- Comparisons or analyses → 4-5 sub-questions.

# Output format

You MUST respond with ONLY a valid JSON object:

{
  "reasoning": "<one sentence on your decomposition logic>",
  "sub_questions": ["<sub-q 1>", "<sub-q 2>", ...]
}

No markdown, no code blocks, no extra text.
"""


async def plan(user_query: str) -> dict:
    t0 = time.time()  # ★ trace
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    plan_result = json.loads(response.choices[0].message.content)

    # ★ trace
    tracer.log(
        "plan",
        reasoning=plan_result["reasoning"],
        sub_questions=plan_result["sub_questions"],
        n_sub_questions=len(plan_result["sub_questions"]),
        duration_ms=int((time.time() - t0) * 1000),
    )
    return plan_result


# ============================================================
# Researcher + ★ trace 插桩 + forced closure 修复
# ============================================================

RESEARCHER_SYSTEM_PROMPT = """\
# Identity
You are a research worker assigned a SINGLE focused question. Your job is 
to research it via web search and produce a concise, source-backed note.

# Workflow
- Use web_search → fetch_url pattern.
- After 2-3 searches without finding the answer, stop and report what you have.
- If fetch_url returns "Warning: short content", that URL is dead. Try a 
  different one — prefer direct article URLs over homepages.

# Stopping criteria
Stop and produce your note when:
- You have a clear, source-backed answer, OR
- You've made 3 searches without progress, OR  
- You've fetched 2 substantive articles on the topic

# Output format

Your final response (when you have no more tool calls) MUST be a concise 
note in this exact format:

ANSWER: <one or two sentences directly answering the sub-question>
KEY_FACTS:
- <fact 1> [source: <url>]
- <fact 2> [source: <url>]
- ...
LIMITATIONS: <if you couldn't fully answer, say what's missing>

Be brief. The note will be one of several merged into a final report.
"""


async def run_researcher(sub_question: str, researcher_id: str, max_steps: int = 8) -> str:
    """
    一个完整的 sub-agent loop。
    返回的是浓缩后的 note(几百字),不是完整 messages。
    
    ★★ 这是 context isolation 的关键 ★★
    主流程拿到的是这个 return value,看不到内部 messages。
    ★ trace 在每个关键点都打了点。
    """
    
    t0 = time.time()  # ★ trace
    tracer.log("researcher_start", researcher_id=researcher_id, sub_q=sub_question)  # ★ trace
    
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Sub-question: {sub_question}"},
    ]

    for step in range(max_steps):
        llm_t0 = time.time()  # ★ trace
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=RESEARCHER_TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # ★ trace: 每次 LLM 调用都记
        tracer.log(
            "llm_call",
            researcher_id=researcher_id,
            step=step + 1,
            n_tool_calls=len(msg.tool_calls) if msg.tool_calls else 0,
            stopped=not msg.tool_calls,
            duration_ms=int((time.time() - llm_t0) * 1000),
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )

        if not msg.tool_calls:
            # 正常收尾
            tracer.log(  # ★ trace
                "researcher_end",
                researcher_id=researcher_id,
                steps_used=step + 1,
                exit_reason="natural",
                note_chars=len(msg.content or ""),
                duration_ms=int((time.time() - t0) * 1000),
            )
            # 模型给出最终 note,直接返回
            return msg.content

        # 并发执行多个 tool call (DeepSeek 支持一次返回多个 tool_calls)
        tool_tasks = []
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool_tasks.append((tool_call.id, name, dispatch_tool(name, args)))
        
        # 等所有 tool 跑完
        for tool_call_id, name, task in tool_tasks:
            result = await task
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            })

    # ★ max_steps 用尽时强制封笔:让模型把 messages 里的事实压缩成 note,
    # 否则 sub-agent 内部的工具产出会全部蒸发。
    tracer.log("forced_closure_triggered", researcher_id=researcher_id)  # ★ trace
    print(f"  [researcher on '{sub_question[:40]}...' hit max_steps, forcing closure]")
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
        tool_choice="none",
        temperature=0.3,
    )
    note = closing.choices[0].message.content

    # ★ trace
    tracer.log(
        "researcher_end",
        researcher_id=researcher_id,
        steps_used=max_steps,
        exit_reason="forced_closure",
        note_chars=len(note),
        duration_ms=int((time.time() - t0) * 1000),
    )
    return note


# ============================================================
# Writer + ★ trace
# ============================================================

WRITER_SYSTEM_PROMPT = """\
# Identity
You are a research writer. You will receive a user's original question, 
plus several research notes from sub-agents who each researched one aspect.
Your job: synthesize them into a final report.

# Critical rules
- Use ONLY information from the provided notes. Do NOT add facts from 
  your training data, even if you "know" something relevant.
- If notes contradict each other, acknowledge the contradiction with 
  both sources.
- Organize by THEME, not by sub-question order.
- Every factual claim must have a citation: [1], [2], etc.
- At the end, provide a Sources section with markdown links.

# Output format

# <Concise title>

<Direct answer to the user's original question in 1-2 sentences.>

<Body, organized in clear paragraphs by theme. Each fact gets [N] citation.>

## Sources
[1] [Title](url)
[2] [Title](url)
...

Match length to question complexity. Don't pad.
"""


async def write_report(user_query: str, plan_result: dict, notes: list[str]) -> str:
    """
    把 N 段 note + 原始 query + plan 喂给 writer,产出最终报告。
    """
    t0 = time.time()  # ★ trace

    research_block = "\n\n---\n\n".join(
        f"## Research on: {q}\n\n{note}"
        for q, note in zip(plan_result["sub_questions"], notes)
    )
    
    user_message = f"""\
                    Original question: {user_query}

                    Planner's decomposition: {plan_result['reasoning']}

                    Research notes from sub-agents:

                    {research_block}

                    Now write the final report. Remember: use ONLY information from these notes.
                    """
    
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        # temperature=0.3,
    )
    report = response.choices[0].message.content

    tracer.log(  # ★ trace
        "writer_done",
        report_chars=len(report),
        duration_ms=int((time.time() - t0) * 1000),
    )
    return report


# ============================================================
# Orchestrator + ★ trace 包住整个 run
# ============================================================

async def deep_research(user_query: str) -> str:
    """
    完整的 deep research pipeline:
      planner → N 个并发 researcher → writer
    """
    run_id = tracer.start_run(query=user_query)  # ★ trace
    print(f"\n[run {run_id}] {user_query}")

    try:
        plan_result = await plan(user_query)
        print(f"  Decomposed into {len(plan_result['sub_questions'])} sub-questions")
        
        notes = await asyncio.gather(*[
            run_researcher(sq, researcher_id=f"r{i+1}")
            for i, sq in enumerate(plan_result["sub_questions"])
        ])
        
        report = await write_report(user_query, plan_result, notes)
    finally:
        tracer.end_run(report_chars=len(report) if 'report' in dir() else 0)  # ★ trace
    
    return report


# ============================================================
# 6. 试跑
# ============================================================

if __name__ == "__main__":
    # query = "Anthropic 最新模型是什么"  # 简单
    # query = "DeepSeek R1 和 V3 的主要区别"  # 中等
    # query = "对比 DeepSeek R1、Qwen3、Claude Opus 4.7 在编程任务上的最新表现" # 复杂
    # query = "为什么 Mamba 在长序列上比 Transformer 高效"   # 深度
    query = "VLA 模型 2024-2026 主要进展和代表工作"  # 你的领域
    # query = "调研一下 2026 年 agent harness 的设计哲学"
    
    report = asyncio.run(deep_research(query))
    print(f"\n{'='*70}\nFINAL REPORT\n{'='*70}\n{report}")
    print(f"\n[trace saved to runs/]")