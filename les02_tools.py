"""
第 2 课：给 agent 加上 web_search 和 fetch_url 两个工具。
重点：让你看见 "工具就是契约" —— 模型只看 schema。
"""

import os
import json
import math
from openai import OpenAI
from dotenv import load_dotenv
from tavily import TavilyClient
import httpx
from readability import Document  # readability-lxml: 抽取网页正文

load_dotenv()

# ============================================================
# 1. 客户端
# ============================================================
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-v4-flash" # deepseek-v4-flash deepseek-chat

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ============================================================
# 2. 工具实现
# ============================================================

def calculator(expression: str) -> str:
    """老朋友，从第 1 课带过来"""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str, max_results: int = 5) -> str:
    """
    用 Tavily 搜索网页。
    
    设计选择 1：返回 JSON 字符串，不返回 dict。
    为什么？因为 tool result 必须是 string（OpenAI API 规定）。
    JSON 字符串既保留了结构，模型又能解析。
    
    设计选择 2：每条结果只返回 url/title/snippet 三个字段。
    Tavily 实际返回的字段更多，但塞进 context 是浪费。
    
    设计选择 3：max_results 默认 5。
    太少模型信息不够，太多 context 浪费。5 是经验值。
    """
    try:
        response = tavily.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # "advanced" 更准但慢且贵
        )
        # 只保留模型需要的字段
        results = [
            {
                "url": r["url"],
                "title": r["title"],
                "snippet": r.get("content", "")[:300],  # 截断到 300 字符
            }
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
        
        # ★ 新增：质量检查
        if len(text) < 200:
            return (
                f"Warning: extracted content from {url} is suspiciously short "
                f"({len(text)} chars). This is likely a JavaScript-rendered page (SPA), "
                f"a paywall, or a navigation page with no real article content. "
                f"Try a different URL — preferably a direct article/blog post URL, "
                f"not a homepage or index page.\n\n"
                f"Extracted content was: {text!r}"
            )
        
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Content truncated at {max_chars} chars]"
        
        return f"# {title}\n\n{text}"
    except httpx.TimeoutException:
        return f"Error: fetch_url timed out for {url}"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error: fetch_url failed: {e}"
    
def read_pdf(url: str, max_chars: int = 4000) -> str:
    """
    抓一个 arxiv 论文的正文。
    
    设计选择 1：用 httpx 下载给出的arxiv url对应的 PDF 文件到内存，
    使用 PyPDF 解析文本。
    
    设计选择 2：max_chars 默认 4000。
    粗略估算：4000 字符 ≈ 1000 token ≈ 一篇论文红摘要的核心内容。
    长文超出部分截断 + 提示模型"内容被截断"。
    
    设计选择 3：明确 timeout，10 秒。
    没 timeout 的工具会让 agent 卡死，是常见 bug。
    """
    try:
        response = httpx.get(
            url,
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchAgent/0.1)"},
        )
        response.raise_for_status()
        
        from io import BytesIO
        from PyPDF2 import PdfReader
        
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Content truncated at {max_chars} chars]"
        
        return text
    except httpx.TimeoutException:
        return f"Error: read_pdf timed out for {url}"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error: read_pdf failed: {e}"



# ============================================================
# 3. 工具声明 —— 模型实际"看见"的全部
# ============================================================
# 重点观察：这里的 description 是给模型读的，不是给人读的。
# 每个 description 都包含：做什么、什么时候用、返回什么。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a math expression. Supports +, -, *, /, **, "
                "sqrt, sin, cos, log, pi, e, etc. (Python math module). "
                "Use this for any arithmetic; do not compute in your head."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. 'sqrt(1972)' or '23 * 47'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. "
                "Use this for: recent events, current prices, news, "
                "anything that may have changed after your training cutoff, "
                "or topics where you're unsure of the latest details. "
                "Returns a JSON list of {url, title, snippet}. "
                "Snippets are short — call fetch_url if you need full content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, 1-6 keywords. Be specific.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many results (default 5, max 10)",
                        "default": 5,
                    },
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
                "Fetch the main content of a webpage. "
                "Use this AFTER web_search when a snippet looks relevant "
                "and you need the full article body. "
                "Returns extracted main text (no HTML/ads/nav). "
                "Content is truncated to 4000 chars; "
                "for long articles, the cutoff is noted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL including https://",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf",
            "description": (
                "Read and extract text from a PDF file. "
                "Use this for academic papers, reports, and other PDF documents. "
                "Returns the extracted text (no images/tables). "
                "Content is truncated to 4000 chars; "
                "for long articles, the cutoff is noted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL including https://",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

TOOL_IMPL = {
    "calculator": calculator,
    "web_search": web_search,
    "fetch_url": fetch_url,
    "read_pdf": read_pdf,
}


# ============================================================
# 4. Harness —— 和第 1 课基本一样,只多了一点细节
# ============================================================

def run_agent(user_query: str, max_steps: int = 15) -> str:
    """
    和第 1 课的骨架完全一样。证明 harness 是和工具数量无关的。
    加工具是纯加法,不需要改循环。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant. "
                "Use web_search to find current information, "
                "fetch_url to read full articles, "
                "and calculator for any math. "
                "Always cite sources with URLs in your final answer."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    for step in range(max_steps):
        print(f"\n{'='*60}\n[Step {step + 1}]")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=1.0,  # 让模型更确定地选择工具和生成内容
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            print(f"[LLM decided to STOP]")
            print(f"[Final Answer]\n{msg.content}")
            return msg.content

        print(f"[LLM wants to call {len(msg.tool_calls)} tool(s)]")
        total_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) # 观察字符串爆炸
        print(f"  [context size: {total_chars} chars, {len(messages)} messages]")   # 观察字符串爆炸
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # 打印参数时截断,避免 fetch 长 URL 时刷屏
            args_preview = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v) 
                            for k, v in args.items()}
            print(f"  → {name}({args_preview})")

            if name in TOOL_IMPL:
                result = TOOL_IMPL[name](**args)
            else:
                result = f"Error: unknown tool '{name}'"
            
            # 同样截断结果预览
            preview = result[:200].replace("\n", " ")
            print(f"  ← {preview}{'...' if len(result) > 200 else ''}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
        
        

    return f"[Reached max_steps={max_steps} without final answer]"


# ============================================================
# 5. 试跑
# ============================================================
if __name__ == "__main__":
    # 一个需要"上网 + 看文章 + 综合"的真实问题 
    # query = "Anthropic 最新发布的 Claude 模型是哪个版本？它有什么主要改进？请给出来源链接。"
    query = "Anthropic 最新模型是什么"
    # query = "对比 DeepSeek R1、Qwen3、Claude Opus 4.7 在编程任务上的最新表现，给出引用"
    # query = "看一下 https://arxiv.org/pdf/2510.23059.pdf 这篇论文，告诉我它的主要贡献"
    query = "最近一年 VLA 模型有什么进展"
    answer = run_agent(query)
    print(f"\n{'='*60}\nFINAL:\n{answer}")
