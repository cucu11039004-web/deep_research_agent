"""
最小的 trace 系统:JSONL 文件 + 全局单例。
设计目标:
- 零侵入性 (装 trace 不改业务逻辑)
- 易消费 (人类读 + 脚本聚合)
- 失败安全 (trace 写失败不影响主流程)
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(exist_ok=True)
        self.run_id: str | None = None
        self.path: Path | None = None
        self.start_ts: float | None = None

    def start_run(self, query: str) -> str:
        """开启一个新 run。返回 run_id。"""
        import time
        self.run_id = uuid.uuid4().hex[:8]
        self.start_ts = time.time()
        # 文件名带时间戳 + run_id,排序方便
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.runs_dir / f"{ts}_{self.run_id}.jsonl"
        self.log("run_start", query=query)
        return self.run_id

    def end_run(self, report_chars: int = 0):
        import time
        duration_ms = int((time.time() - self.start_ts) * 1000) if self.start_ts else 0
        self.log("run_end", report_chars=report_chars, total_duration_ms=duration_ms)
        self.run_id = None
        self.path = None
        self.start_ts = None

    def log(self, event_type: str, **fields):
        """
        记一条事件。
        失败安全:写文件失败不抛异常,只 print warning。
        """
        if self.path is None:
            return  # 没在 run 里,丢弃
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "type": event_type,
            **fields,
        }
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[tracer warning] failed to write: {e}")


# 全局单例。所有模块共享一个 tracer。
tracer = Tracer()