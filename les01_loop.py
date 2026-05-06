"""
第 1 课：minimal agent loop
一个能用 calculator 工具回答数学问题的 agent。
重点：让你看见那个 while 循环。
"""

import os
import json
import math
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. LLM 客户端
# ============================================================
# DeepSeek 兼容 OpenAI 的 API 格式，所以直接用 openai 库即可。
# 这是 2025 年的事实标准——绝大多数 LLM 提供商都兼容 OpenAI API。
# client = OpenAI(
#     api_key=os.environ["DEEPSEEK_API_KEY"],
#     base_url="https://api.deepseek.com",
# )
# MODEL = "deepseek-chat"  # 也可以用 deepseek-reasoner，但慢且贵

# client = OpenAI(
#     api_key=os.environ["MINIMAX_API_KEY"],
#     base_url="https://api.minimaxi.com/v1",
# )
# MODEL = "MiniMax-M2.7" # MiniMax-M2.7

client = OpenAI(
    api_key=os.environ["GLM_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)
MODEL = "glm-5.1" # glm-5.1


# ============================================================
# 2. 工具定义
# ============================================================
# 这里有两个东西要分清：
#   (a) 工具的"实现"：真正干活的 Python 函数
#   (b) 工具的"声明"：告诉 LLM "你有这个工具" 的 JSON Schema
# LLM 看不到 Python 函数，它只看 JSON Schema。
# 第 2 课会深入讲这个 contract，本课先用着。

# def calculator(expression: str) -> str:
#     """
#     工具实现：算一个数学表达式。
#     用 eval 是不安全的（生产环境绝对不能这么干），但教学够用。
#     我们限制了允许的名字空间到 math 模块，稍微减少一点风险。
#     """
#     try:
#         # 只允许 math 模块的函数 + 基础运算
#         allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
#         result = eval(expression, {"__builtins__": {}}, allowed)
#         return str(result)
#     except Exception as e:
#         # 关键：工具出错也要返回字符串给 LLM，让它知道并尝试修正
#         return f"Error: {e}"

def get_current_time() -> str:
    """
    工具实现：获取当前时间。
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculator(expression: str) -> str:
    """
    工具实现：算一个数学表达式。
    用 eval 是不安全的（生产环境绝对不能这么干），但教学够用。
    我们限制了允许的名字空间到 math 模块，稍微减少一点风险。
    """
    
    try:
        if "sqrt" in expression:
            return "Error: sqrt() is not supported. Use ** 0.5 instead, e.g. '4 ** 0.5' for sqrt(4)."
        # 只允许 math 模块的函数 + 基础运算
        else:
            allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return str(result)
    except Exception as e:
        # 关键：工具出错也要返回字符串给 LLM，让它知道并尝试修正
        return f"Error: {e}"


# 工具的"声明"。这就是 OpenAI/DeepSeek 的 function calling 格式。
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, "
                           "sqrt, sin, cos, log, pi, e, etc. (Python math module).",
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
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# 名字 → 实现 的映射，dispatch 时用
TOOL_IMPL = {
    "calculator": calculator,
    "get_current_time": get_current_time,
}


# ============================================================
# 3. 核心 harness —— 这是这一课的灵魂
# ============================================================

def run_agent(user_query: str, max_steps: int = 10) -> str:
    """
    Minimal agent harness.
    
    这是整个 Sprint 1 最重要的一段代码。
    把它读 5 遍，背下来。后面所有东西都是在这个骨架上长出来的。
    """
    # 消息历史 = agent 的"记忆"。每一步都往这里 append。
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the calculator tool for any arithmetic. Don't compute in your head."},
        {"role": "user", "content": user_query},
    ]

    for step in range(max_steps):  # max_steps 是兜底，正常情况模型会主动停
        print(f"\n{'='*60}\n[Step {step + 1}]")

        # ---- (1) 调用 LLM ----
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,  # ★ 关键：把工具声明传进去
            tool_choice="auto",   # 让模型自己决定是否调用工具，required：强制调用工具，none：不调用工具
        )
        msg = response.choices[0].message

        # ---- (2) 把模型的回复 append 进历史 ----
        # 注意：必须 append assistant 消息，下一轮 LLM 才能看到自己说过什么
        messages.append(msg.model_dump(exclude_none=True))

        # ---- (3) 判断：模型说要调工具，还是给最终答案？ ----
        if not msg.tool_calls:
            # 没有工具调用 → 模型认为已经能回答了 → 退出循环
            print(f"[Step {step+1}] LLM decided to STOP. No tool_calls in response.") # 尝试
            print(f"[Final Answer] {msg.content}")
            return msg.content

        # ---- (4) 模型要调工具，依次执行 ----
        print(f"[LLM wants to call {len(msg.tool_calls)} tool(s)]")
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  → {name}({args})")

            # dispatch 到具体实现
            if name in TOOL_IMPL:
                result = TOOL_IMPL[name](**args)
            else:
                result = f"Error: unknown tool '{name}'"
            print(f"  ← {result}")

            # ★ 关键：工具结果也要 append 回历史，下一轮 LLM 才能看见
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
        # 循环继续 → 模型基于新的工具结果再决定下一步

    # 走到这里说明 max_steps 跑完了模型还没给最终答案
    return f"[Reached max_steps={max_steps} without final answer]"


# ============================================================
# 4. 试跑
# ============================================================
if __name__ == "__main__":
    # query = "What is the square root of (23 * 47 + 891)? Show me the steps."
    # query = "hello"
    # query = "23 * 47 是多少？" 
    # query = "sqrt(1972)是多少？"
    query = "现在是几点？再算一下从现在到2027年元旦还有多少天？"
    answer = run_agent(query)
    print(f"\n{'='*60}\nFINAL: {answer}")