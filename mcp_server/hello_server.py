"""
Sprint 2-new 阶段 2:最小 MCP server,仅用于排除 transport 层的所有坑。

设计目标:
- 一个工具:echo(text: str) -> str
- 无任何外部依赖(httpx / openai 全不用)
- 跑通 stdio transport,在 Cursor 里能 @ 调用
"""


from mcp.server.fastmcp import FastMCP

# ★ 创建 server 实例，第一个参数是 server name,Cursor 列出 MCP 服务时会显示这个名字
mcp = FastMCP("hello-mcp")


# ★ @mcp.tool() 装饰器把一个普通 Python 函数注册成 MCP tool
# - 函数名 → tool name
# - docstring → tool description (这就是 Sprint 1 你写的 OpenAI tool schema 的 description 字段)
# - 类型 hint → tool input schema (自动从 Python type hints 生成 JSON schema)
@mcp.tool()
def mcp_echo(text: str) -> str:
    """Echo back the input text. Useful for testing the MCP connection."""
    return f"MCP_Response:{text}"


# ★ 第二个 tool,验证多 tool 注册
@mcp.tool()
def mcpc_add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


# ★ MCP server 的入口，transport="stdio" 是默认值(可省略),这里显式写出来强调
# 这个 if __name__ == "__main__" 不能省 —— Cursor 会以脚本方式启动这个文件
if __name__ == "__main__":
    mcp.run(transport="stdio")