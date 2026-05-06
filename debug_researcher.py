"""
debug_researcher.py
单独跟踪一个 researcher 的每一步,排查 DeepSeek R1 / Qwen3 这类 sub-question
为什么会跑到 max_steps 还出不了 final note。

直接复用 les04_orchestration 里的 client/MODEL/工具/schema/prompt,
只把 run_researcher 的 loop 包一层 verbose 打印。
"""

import asyncio
import json

from les04_orchestration import (
    client,
    MODEL,
    RESEARCHER_TOOLS_SCHEMA,
    RESEARCHER_SYSTEM_PROMPT,
    dispatch_tool,
)


def _short(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + f"... [+{len(s)-n} chars]"


def _classify(name: str, result: str) -> str:
    """把工具返回归到几个类别,方便一眼看出 researcher 在跟什么搏斗"""
    if result.startswith("Error:"):
        return "ERROR"
    if name == "fetch_url" and result.startswith("Warning:"):
        return "SHORT_CONTENT"  # readability 抽不到正文
    if name == "web_search":
        try:
            arr = json.loads(result)
            return f"OK ({len(arr)} hits)"
        except Exception:
            return "OK?"
    if name == "fetch_url":
        return f"OK ({len(result)} chars)"
    return "OK"


async def run_researcher_verbose(sub_question: str, max_steps: int = 8) -> str:
    print(f"\n{'='*80}\nSUB-QUESTION: {sub_question}\n{'='*80}")
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Sub-question: {sub_question}"},
    ]

    fetched_urls: list[str] = []
    searches: list[str] = []

    for step in range(1, max_steps + 1):
        print(f"\n--- STEP {step} ---")
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
            print(f"[FINAL NOTE @ step {step}]")
            print(msg.content)
            return msg.content

        if msg.content:
            print(f"  thought: {_short(msg.content, 280)}")
        print(f"  -> {len(msg.tool_calls)} tool call(s):")
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            arg_preview = json.dumps(args, ensure_ascii=False)
            print(f"     {tc.function.name}({_short(arg_preview, 160)})")

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            if name == "web_search":
                searches.append(args.get("query", ""))
            elif name == "fetch_url":
                fetched_urls.append(args.get("url", ""))

            result = await dispatch_tool(name, args)
            tag = _classify(name, result)
            print(f"     <- {name} [{tag}]: {_short(result, 240)}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    print(f"\n[!! REACHED max_steps={max_steps} — no final note produced !!]")
    print(f"  total searches: {len(searches)}")
    for i, q in enumerate(searches, 1):
        print(f"    s{i}: {q}")
    print(f"  total fetches: {len(fetched_urls)}")
    for i, u in enumerate(fetched_urls, 1):
        print(f"    f{i}: {u}")
    return f"[Researcher reached max_steps={max_steps} on: {sub_question}]"


async def main():
    sub_qs = [
        "What is the latest performance of DeepSeek R1 on programming tasks?",
        "What is the latest performance of Qwen3 on programming tasks?",
    ]
    for q in sub_qs:
        await run_researcher_verbose(q)


if __name__ == "__main__":
    asyncio.run(main())
