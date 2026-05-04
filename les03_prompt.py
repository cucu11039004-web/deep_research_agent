"""
第 3 课：Prompt as Program
两件事:
1. 把 agent 的 system prompt 工程化
2. 写一个独立的 planner，把 query 拆成 sub-questions
"""

import os
import json
import math
from openai import OpenAI
from dotenv import load_dotenv
from tavily import TavilyClient
import httpx
from readability import Document

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-v4-flash" # deepseek-v4-flash deepseek-chat
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ============================================================
# 1. 工具实现 (从第 2 课直接拿过来,带上你修补好的 fetch_url)
# ============================================================

def calculator(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str, max_results: int = 5) -> str:
    try:
        response = tavily.search(query=query, max_results=max_results, search_depth="basic")
        results = [
            {"url": r["url"], "title": r["title"], "snippet": r.get("content", "")[:300]}
            for r in response.get("results", [])
        ]
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: web_search failed: {e}"


def fetch_url(url: str, max_chars: int = 4000) -> str:
    try:
        response = httpx.get(url, timeout=10.0, follow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchAgent/0.1)"})
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
                f"({len(text)} chars). This is likely a JavaScript-rendered page (SPA), "
                f"a paywall, or a navigation page. Try a different URL — preferably a "
                f"direct article URL, not a homepage.\n\n"
                f"Extracted content was: {text!r}"
            )
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Content truncated at {max_chars} chars]"
        return f"# {title}\n\n{text}"
    except Exception as e:
        return f"Error: fetch_url failed: {e}"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use for recent events, "
                "current state, or anything that may have changed after training cutoff. "
                "Returns JSON list of {url, title, snippet}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, 1-6 keywords"},
                    "max_results": {"type": "integer", "default": 5},
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
                "Returns extracted main text. If you get a 'Warning: short content' "
                "message, the URL is useless — try a different one."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Use for any arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

TOOL_IMPL = {"calculator": calculator, "web_search": web_search, "fetch_url": fetch_url}


# ============================================================
# 2. ★ 这一课的核心: 工程化的 system prompt
# ============================================================

RESEARCHER_SYSTEM_PROMPT = """\
# Identity
You are a academic research assistant. Your job is to answer the user's question by 
searching the web, reading articles, and synthesizing findings — NOT by 
relying on your training memory.

# Workflow

## Decision rules
- For questions about current events, latest products, recent news, or 
  any "what is the latest X" — ALWAYS use web_search first. Never answer 
  such questions from memory.
- For arithmetic, ALWAYS use the calculator tool. Never compute mentally.
- Prefer arxiv.org / acm.org / openreview.net over blog posts and news.
- After 3 searches without progress, stop searching and synthesize what 
  you have. Tell the user what you tried.

## Tool usage patterns
- The standard pattern is: web_search → fetch_url. Search first, fetch 
  the most promising URL second.
- When fetching arxiv URLs, prefer the /abs/ version over /pdf/ for first read.
- If fetch_url returns "Warning: short content", that URL is dead. 
  Pick a different URL from search results — preferably a direct article 
  URL (path contains /news/, /blog/, /article/), not a homepage.
- Don't repeat similar searches. If your second search returns the same 
  top URLs as the first, fetch one of them instead of searching again.

## Stopping criteria
You should STOP and produce a final answer when:
- You have a clear, source-backed answer to the user's question, OR
- You've made 3 searches without finding the answer (acknowledge limits), OR
- You've fetched 2-3 articles that cover the topic well

# Output Format

Your final answer MUST include:
1. A direct answer in the FIRST sentence
2. Supporting details, organized in clear paragraphs (not heavy bullet lists)
3. A final "Sources" section with markdown links: [Title](URL)
4. When citing papers, include authors and year, e.g. (Smith et al., 2024)

Do not include decorative emojis. Use markdown sparingly. Be concise.
"""


# ============================================================
# 3. Harness (和第 2 课结构一致, 只换了 system prompt)
# ============================================================

def run_researcher(user_query: str, max_steps: int = 10) -> str:
    """带工程化 system prompt 的 researcher agent."""
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for step in range(max_steps):
        print(f"\n{'='*60}\n[Step {step + 1}]")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.3,  # ★ 低温但不为 0,平衡稳定性和适应性
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            print(f"[STOP] {msg.content[:100]}...")
            return msg.content

        print(f"[{len(msg.tool_calls)} tool call(s)]")
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            args_preview = {k: (v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v) 
                           for k, v in args.items()}
            print(f"  → {name}({args_preview})")
            
            result = TOOL_IMPL[name](**args) if name in TOOL_IMPL else f"Error: unknown tool '{name}'"
            preview = result[:150].replace("\n", " ")
            print(f"  ← {preview}{'...' if len(result) > 150 else ''}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return f"[Reached max_steps={max_steps}]"


# ============================================================
# 4. ★★ 这一课的另一个核心: 独立的 Planner
# ============================================================
# Planner 不是 agent,是一次 LLM 调用。它的工作是把一个复杂 query 
# 拆成 N 个独立可研究的 sub-question。
# 它没有工具,没有循环,只有一个 prompt + structured output。

PLANNER_SYSTEM_PROMPT = """\
# Identity
You are a research planner. Your job is to decompose a user's question 
into 2-5 focused sub-questions that, when answered together, fully address 
the original question.

# Decomposition principles
- Each sub-question must be INDEPENDENTLY researchable (a separate person 
  could answer it without knowing the others).
- Each sub-question should focus on ONE aspect — don't combine multiple 
  angles into one question.
- Cover the full scope of the original question. Don't leave gaps.
- For simple questions (e.g. "what is X"), 2 sub-questions is enough.
- For complex comparisons or analyses, 4-5 sub-questions.

# Output format

You MUST respond with ONLY a valid JSON object, no other text:

{
  "reasoning": "<one sentence explaining how you decomposed the question>",
  "sub_questions": [
    "<sub-question 1>",
    "<sub-question 2>",
    ...
  ]
}

Do not wrap in markdown code blocks. Do not add explanation outside JSON.
"""


def plan(user_query: str) -> dict:
    """
    把 user_query 拆成 sub-questions。
    
    关键技术: response_format={"type": "json_object"} 强制返回 JSON。
    这是 OpenAI/DeepSeek 都支持的 "JSON mode"。
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},  # ★ 强制 JSON 输出
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


# ============================================================
# 5. 试跑 - 对比 planner 在不同复杂度 query 上的输出
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PART 1: Planner 演示")
    print("="*60)
    
    test_queries = [
        "Anthropic 最新模型是什么",
        "对比 DeepSeek R1、Qwen3、Claude Opus 4.7 在编程任务上的最新表现",
        "为什么 transformer 比 RNN 更适合处理长序列",
    ]
    
    for q in test_queries:
        print(f"\n[Query] {q}")
        plan_result = plan(q)
        print(f"[Reasoning] {plan_result['reasoning']}")
        print(f"[{len(plan_result['sub_questions'])} sub-questions]")
        for i, sq in enumerate(plan_result['sub_questions'], 1):
            print(f"  {i}. {sq}")
    
    print("\n" + "="*60)
    print("PART 2: 升级版 researcher 跑一个 query")
    print("="*60)
    
    # answer = run_researcher("Anthropic 最新模型是什么")
    answer = run_researcher("最近一年 VLA 模型有什么进展")
    print(f"\n{'='*60}\nFINAL:\n{answer}")