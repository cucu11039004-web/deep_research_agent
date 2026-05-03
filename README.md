# Deep Research Agent

这是一个用于学习 minimal agent loop 的 Python 示例项目。当前示例在 `les01_loop.py` 中实现了一个能调用工具的简单 agent，包括数学计算工具和获取当前时间工具，重点是理解 LLM、工具调用和循环执行之间的关系。

## 环境配置

项目使用 conda 环境 `deepagent`。先激活环境：

```bash
conda activate deepagent
```

然后安装依赖：

```bash
pip install -r requirements.txt
```

在项目根目录创建 `.env` 文件，并配置 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

运行示例：

```bash
python les01_loop.py
```
