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
MODEL = "glm-5.1" # glm-5.1

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

# 异步 dispatch
async def dispatch_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return web_search(**args)  # 同步函数,直接调
    elif name == "fetch_url":
        return await fetch_url(**args)  # 异步函数,await
    elif name == "calculator":
        return calculator(**args)
    else:
        return f"Error: unknown tool '{name}'"


# ============================================================
# 2. Planner (第 3 课的,基本没变,只是改成 async)
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
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)


# ============================================================
# 3. ★ Sub-Agent: Researcher
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


async def run_researcher(sub_question: str, max_steps: int = 8) -> str:
    """
    一个完整的 sub-agent loop。
    返回的是浓缩后的 note(几百字),不是完整 messages。
    
    ★★ 这是 context isolation 的关键 ★★
    主流程拿到的是这个 return value,看不到内部 messages。
    """
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Sub-question: {sub_question}"},
    ]

    for step in range(max_steps):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=RESEARCHER_TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content # 模型出口1： 模型给出最终 note,直接返回

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

    # 模型出口2：超过 max_steps 后，强制做一次不带 tools 的 finalization。
    # 这样仍然保持 context isolation：主流程只拿到压缩 note，看不到内部 messages。
    messages.append({

        "role": "user",
        "content": f"""\
                    You have reached max_steps={max_steps} for this sub-question:

                    {sub_question}

                    Now stop using tools and produce the best possible final research note
                    based ONLY on the search/fetch results already present in this conversation.
                    Do NOT invent missing facts or sources.

                    Your response MUST follow this exact format:

                    ANSWER: <one or two sentences directly answering the sub-question, or say if evidence is insufficient>
                    KEY_FACTS:
                    - <fact 1> [source: <url>]
                    - <fact 2> [source: <url>]
                    LIMITATIONS: Mention that this note was forced because the researcher reached max_steps={max_steps}, and state what remains uncertain.
                    """,
                        })

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
        )
        final_note = response.choices[0].message.content
        if final_note:
            return final_note
        return (
            "ANSWER: Unable to produce a complete answer from the available research trace.\n"
            "KEY_FACTS:\n"
            "- No reliable final note was produced. [source: unavailable]\n"
            f"LIMITATIONS: Researcher reached max_steps={max_steps}, and finalization returned empty content."
        )
    except Exception as e:
        return (
            "ANSWER: Unable to produce a complete answer from the available research trace.\n"
            "KEY_FACTS:\n"
            "- No reliable final note was produced. [source: unavailable]\n"
            f"LIMITATIONS: Researcher reached max_steps={max_steps}, and finalization failed: {e}"
        )


# ============================================================
# 4. ★ Writer: 综合 N 个 note 成最终报告
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
    # 把 sub-question 和对应的 note 配对呈现
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
    return response.choices[0].message.content


# ============================================================
# 5. ★★★ Orchestrator: 把整个流水线串起来
# ============================================================

async def deep_research(user_query: str) -> str:
    """
    完整的 deep research pipeline:
      planner → N 个并发 researcher → writer
    """
    print(f"\n{'='*70}")
    print(f"USER QUERY: {user_query}")
    print(f"{'='*70}")
    
    # ---- Phase 1: Planning ----
    print("\n[PHASE 1] Planning...")
    plan_result = await plan(user_query)
    print(f"  Reasoning: {plan_result['reasoning']}")
    print(f"  Decomposed into {len(plan_result['sub_questions'])} sub-questions:")
    for i, sq in enumerate(plan_result["sub_questions"], 1):
        print(f"    {i}. {sq}")
    
    # ---- Phase 2: Concurrent Research ----
    print(f"\n[PHASE 2] Researching {len(plan_result['sub_questions'])} sub-questions concurrently...")
    
    # ★ 并发的核心: asyncio.gather
    # 同时启动 N 个 researcher,等所有都完成
    notes = await asyncio.gather(*[
        run_researcher(sq) for sq in plan_result["sub_questions"]
    ])
    
    print(f"  All {len(notes)} researchers completed.")
    for i, note in enumerate(notes, 1):
        preview = note[:120].replace("\n", " ")
        print(f"  Note {i} preview: {preview}...")
    
    # ---- Phase 3: Synthesis ----
    print("\n[PHASE 3] Writing final report...")
    report = await write_report(user_query, plan_result, notes)
    
    return report


# ============================================================
# 6. 试跑
# ============================================================

if __name__ == "__main__":
    query = "对比 DeepSeek R1、Qwen3、Claude Opus 4.7 在编程任务上的最新表现"
    # query = "Anthropic 最新模型是什么"  # 简单的也能跑,只是 sub_questions 会少
    # query = "VLA 模型 2024-2026 的主要进展和代表工作有哪些"
    
    report = asyncio.run(deep_research(query))
    print(f"\n{'='*70}\nFINAL REPORT\n{'='*70}\n")
    print(report)
