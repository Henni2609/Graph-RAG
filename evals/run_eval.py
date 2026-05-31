#!/usr/bin/env python
"""Ragas evaluation pipeline for the Graph RAG system.

Judge: claude CLI (nutzt dein Claude-Abo, kein API-Key nötig)
RAG:   LLM_API_KEY + NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD

Usage:
  python evals/run_eval.py [--limit N] [--judge-model MODEL]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Ragas evaluation against the Graph RAG system.")
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples.")
    p.add_argument(
        "--judge-model",
        default="claude-sonnet-4-6",
        help="Claude model for Ragas judge (default: claude-sonnet-4-6).",
    )
    p.add_argument(
        "--dataset",
        default=str(_ROOT / "evals" / "dataset" / "golden_dataset.jsonl"),
        help="Path to the golden dataset JSONL file.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

def _check_env() -> None:
    if not os.getenv("LLM_API_KEY"):
        sys.exit("Missing required env var: LLM_API_KEY")
    result = subprocess.run(["claude", "--version"], capture_output=True)
    if result.returncode != 0:
        sys.exit("claude CLI not found — install Claude Code or run `npm install -g @anthropic-ai/claude-code`")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(path: str, limit: int | None) -> list[dict[str, Any]]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if limit is not None:
        samples = samples[:limit]
    print(f"Loaded {len(samples)} samples from {path}")
    return samples


# ---------------------------------------------------------------------------
# RAG inference
# ---------------------------------------------------------------------------

def _run_rag_pipeline(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from kg_rag.config import RagConfig
    from kg_rag.compat import document_content
    from kg_rag.pipelines.query import QueryPipeline

    config = RagConfig.from_env()
    pipeline = QueryPipeline(config)
    results = []

    try:
        for i, sample in enumerate(samples):
            question = sample["question"]
            print(f"  [{i + 1}/{len(samples)}] {question[:80]}")
            try:
                result = pipeline.run(question, session_id="default")
                contexts = [
                    document_content(d)
                    for d in (result.vector_documents or []) + (result.graph_documents or [])
                    if document_content(d).strip()
                ]
                results.append({
                    "question": question,
                    "answer": result.answer or "",
                    "contexts": contexts,
                    "ground_truth_answer": sample["ground_truth_answer"],
                    "ground_truth_contexts": sample.get("ground_truth_contexts") or [],
                    "difficulty": sample.get("difficulty", ""),
                    "source_doc": sample.get("source_doc", ""),
                })
            except Exception as exc:
                print(f"    WARNING: RAG pipeline error: {exc}")
                results.append({
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth_answer": sample["ground_truth_answer"],
                    "ground_truth_contexts": sample.get("ground_truth_contexts") or [],
                    "difficulty": sample.get("difficulty", ""),
                    "source_doc": sample.get("source_doc", ""),
                })
    finally:
        try:
            pipeline.store.close()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Claude CLI judge (LangChain BaseChatModel wrapper)
# ---------------------------------------------------------------------------

def _build_judge(model: str) -> Any:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from ragas.llms import LangchainLLMWrapper

    class _ClaudeCLIChat(BaseChatModel):
        model_name: str = model

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            system_parts: list[str] = []
            user_parts: list[str] = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    system_parts.append(str(m.content))
                elif isinstance(m, HumanMessage):
                    user_parts.append(str(m.content))
                elif isinstance(m, AIMessage):
                    user_parts.append(f"[Assistant]: {m.content}")
                else:
                    user_parts.append(str(m.content))

            prompt = "\n\n".join(filter(None, ["\n\n".join(system_parts), "\n\n".join(user_parts)]))

            proc = subprocess.run(
                ["claude", "-p", prompt, "--model", self.model_name],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"claude CLI failed: {proc.stderr[:300]}")

            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=proc.stdout.strip()))])

        @property
        def _llm_type(self) -> str:
            return "claude-cli"

    return LangchainLLMWrapper(_ClaudeCLIChat())


# ---------------------------------------------------------------------------
# Ragas evaluation
# ---------------------------------------------------------------------------

def _build_embedder() -> Any:
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings

    return RagasHFEmbeddings(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def _run_ragas(rag_results: list[dict[str, Any]], judge_model: str) -> Any:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"] if r["contexts"] else [""],
            reference=r["ground_truth_answer"],
            reference_contexts=r["ground_truth_contexts"] if r["ground_truth_contexts"] else None,
        )
        for r in rag_results
    ]
    dataset = EvaluationDataset(samples=samples)

    judge = _build_judge(judge_model)
    embedder = _build_embedder()

    print(f"\nRunning Ragas evaluation — judge: {judge_model} (via claude CLI / dein Abo)")
    return evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=judge,
        embeddings=embedder,
        raise_exceptions=False,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _save_csv(rag_results: list[dict[str, Any]], eval_result: Any, output_path: Path) -> None:
    df = eval_result.to_pandas()

    rows = []
    for i, row in df.iterrows():
        r = rag_results[i]
        rows.append({
            "question": r["question"],
            "difficulty": r["difficulty"],
            "source_doc": r["source_doc"],
            "faithfulness": row.get("faithfulness", ""),
            "answer_relevancy": row.get("answer_relevancy", ""),
            "context_precision": row.get("context_precision", ""),
            "context_recall": row.get("context_recall", ""),
            "answer": r["answer"][:300].replace("\n", " "),
        })

    if not rows:
        print(f"WARNING: no eval rows — skipping CSV export to {output_path}")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved: {output_path}")


def _save_markdown(
    rag_results: list[dict[str, Any]], eval_result: Any, output_path: Path, judge_model: str
) -> None:
    df = eval_result.to_pandas()

    def _avg(col: str) -> str:
        if col in df.columns:
            vals = df[col].dropna()
            return f"{vals.mean():.3f}" if len(vals) else "N/A"
        return "N/A"

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    difficulty_rows: dict[str, dict[str, list[float]]] = {}
    for i, row in df.iterrows():
        diff = rag_results[i]["difficulty"]
        if diff not in difficulty_rows:
            difficulty_rows[diff] = {m: [] for m in metrics}
        for m in metrics:
            val = row.get(m)
            if val is not None and str(val) != "nan":
                difficulty_rows[diff][m].append(float(val))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Ragas Evaluation Report",
        "",
        f"**Date:** {now}  ",
        f"**Judge model:** {judge_model} (claude CLI)  ",
        f"**Samples evaluated:** {len(rag_results)}",
        "",
        "## Overall Scores",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for m in metrics:
        lines.append(f"| {m.replace('_', ' ').title()} | {_avg(m)} |")

    lines += [
        "",
        "## Scores by Difficulty",
        "",
        "| Difficulty | Faithfulness | Answer Relevancy | Context Precision | Context Recall | N |",
        "|------------|-------------|-----------------|------------------|----------------|---|",
    ]
    for diff in sorted(difficulty_rows):
        d = difficulty_rows[diff]
        n = max(len(v) for v in d.values()) if any(d.values()) else 0

        def _davg(vals: list[float]) -> str:
            return f"{sum(vals)/len(vals):.3f}" if vals else "N/A"

        lines.append(
            f"| {diff} | {_davg(d['faithfulness'])} | {_davg(d['answer_relevancy'])} | "
            f"{_davg(d['context_precision'])} | {_davg(d['context_recall'])} | {n} |"
        )

    lines += [
        "",
        "## Per-Question Scores",
        "",
        "| # | Question | Difficulty | Faithfulness | Ans.Rel | Ctx.Prec | Ctx.Rec |",
        "|---|----------|------------|-------------|---------|---------|---------|",
    ]
    for i, row in df.iterrows():
        r = rag_results[i]
        q = r["question"][:60].replace("|", "/")
        fa = f"{row.get('faithfulness', ''):.2f}" if str(row.get("faithfulness", "")) != "nan" else "-"
        ar = f"{row.get('answer_relevancy', ''):.2f}" if str(row.get("answer_relevancy", "")) != "nan" else "-"
        cp = f"{row.get('context_precision', ''):.2f}" if str(row.get("context_precision", "")) != "nan" else "-"
        cr = f"{row.get('context_recall', ''):.2f}" if str(row.get("context_recall", "")) != "nan" else "-"
        lines.append(f"| {i + 1} | {q} | {r['difficulty']} | {fa} | {ar} | {cp} | {cr} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _check_env()

    reports_dir = _ROOT / "evals" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading golden dataset...")
    samples = _load_dataset(args.dataset, args.limit)

    print("\nStep 2: Running RAG pipeline...")
    rag_results = _run_rag_pipeline(samples)

    print("\nStep 3: Running Ragas evaluation...")
    eval_result = _run_ragas(rag_results, args.judge_model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\nStep 4: Saving results...")
    _save_csv(rag_results, eval_result, reports_dir / f"eval_{timestamp}.csv")
    _save_markdown(rag_results, eval_result, reports_dir / f"eval_{timestamp}.md", args.judge_model)

    print("\nDone.")


if __name__ == "__main__":
    main()
