"""
读 runs/ 下的所有 trace 文件,汇总成可读的统计报告。
这是你以后 debug、复盘、写博客时的主力工具。
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def load_trace(path: Path) -> list[dict]:
    """读一个 jsonl 文件成 list of dict。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_run(trace: list[dict]) -> dict:
    """从一个 trace 里提取关键数据。"""
    run_start = next((e for e in trace if e["type"] == "run_start"), None)
    run_end = next((e for e in trace if e["type"] == "run_end"), None)
    plan_event = next((e for e in trace if e["type"] == "plan"), None)
    
    # researcher 维度的聚合
    researcher_steps: dict[str, int] = defaultdict(int)
    researcher_exit: dict[str, str] = {}
    forced_closures = []
    tool_calls = Counter()
    tool_warnings = []
    
    for e in trace:
        if e["type"] == "llm_call":
            researcher_steps[e["researcher_id"]] = max(
                researcher_steps[e["researcher_id"]], e["step"]
            )
        elif e["type"] == "researcher_end":
            researcher_exit[e["researcher_id"]] = e["exit_reason"]
        elif e["type"] == "forced_closure_triggered":
            forced_closures.append(e["researcher_id"])
        elif e["type"] == "tool_result":
            tool_calls[e["tool"]] += 1
            if e.get("is_warning"):
                tool_warnings.append((e["tool"], e["result_preview"][:80]))
    
    return {
        "run_id": run_start["run_id"] if run_start else "?",
        "query": run_start["query"] if run_start else "?",
        "duration_s": (run_end["total_duration_ms"] / 1000) if run_end else None,
        "n_sub_questions": plan_event["n_sub_questions"] if plan_event else 0,
        "researcher_steps": dict(researcher_steps),
        "researcher_exit": researcher_exit,
        "forced_closures": forced_closures,
        "tool_calls": dict(tool_calls),
        "tool_warnings": tool_warnings,
    }


def print_summary(s: dict):
    print(f"\n[{s['run_id']}] {s['query'][:60]}")
    print(f"  duration: {s['duration_s']:.1f}s | sub-questions: {s['n_sub_questions']}")
    print(f"  steps per researcher: {s['researcher_steps']}")
    print(f"  exit reasons: {s['researcher_exit']}")
    if s['forced_closures']:
        print(f"  ⚠ forced closure on: {s['forced_closures']}")
    print(f"  tool calls: {s['tool_calls']}")
    if s['tool_warnings']:
        print(f"  ⚠ {len(s['tool_warnings'])} tool warnings:")
        for tool, preview in s['tool_warnings']:
            print(f"     - {tool}: {preview}...")


if __name__ == "__main__":
    runs_dir = Path("runs")
    files = sorted(runs_dir.glob("*.jsonl"))
    
    if not files:
        print("No trace files in runs/")
        exit()
    
    print(f"Found {len(files)} trace files\n")
    
    summaries = [summarize_run(load_trace(f)) for f in files]
    for s in summaries:
        print_summary(s)
    
    # 跨 run 聚合
    print(f"\n{'='*60}\nAGGREGATE STATS\n{'='*60}")
    if summaries:
        durations = [s["duration_s"] for s in summaries if s["duration_s"]]
        forced = sum(1 for s in summaries if s["forced_closures"])
        warnings = sum(len(s["tool_warnings"]) for s in summaries)
        print(f"  Total runs: {len(summaries)}")
        print(f"  Avg duration: {sum(durations)/len(durations):.1f}s")
        print(f"  Worst duration: {max(durations):.1f}s")
        print(f"  Runs with forced closure: {forced}/{len(summaries)}")
        print(f"  Total tool warnings: {warnings}")