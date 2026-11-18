"""
chat.py — CLI Chat Interface for the AI Resume Job Matcher Agent
================================================================
Assignment-compliant conversational interface using LangGraph + Rich.

Usage:
    .\\venv\\Scripts\\python.exe chat.py

Commands: /new  /history  /state  /graph  /demo  /quit
"""

import sys
import os

# ── Encoding & warning suppression — MUST be first ──────────────────────────
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    for _p in [r"C:\Anaconda\Library\bin", r"C:\Anaconda\DLLs", r"C:\Anaconda"]:
        if os.path.exists(_p):
            try:
                os.add_dll_directory(_p)
            except Exception:
                pass

# ── Standard imports ─────────────────────────────────────────────────────────
import uuid
import threading
import time
from datetime import datetime
from typing import Optional

# ── Rich imports ─────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.rule import Rule
from rich.markup import escape   # <── KEY: escapes [ ] from LLM output
from rich import box

# ── Agent import ─────────────────────────────────────────────────────────────
from src.matching_agent import run_agent, get_graph_ascii_visualization

# ── Console (force_terminal=True prevents Windows crash on non-TTY) ───────────
console = Console(
    width=110,
    highlight=False,        # Don't parse Python/JSON syntax
    markup=True,            # We control markup explicitly via escape()
    force_terminal=True,    # Stable on Windows PowerShell
)

# ── Static text ───────────────────────────────────────────────────────────────
HELP_TEXT = """\
[bold yellow]Commands[/]
  [cyan]/new[/]      Start a fresh conversation
  [cyan]/history[/]  Show conversation history
  [cyan]/state[/]    Inspect current agent state
  [cyan]/graph[/]    Show LangGraph state machine
  [cyan]/demo[/]     Run 5 pre-scripted demo scenarios
  [cyan]/quit[/]     Exit

[bold yellow]Example Queries[/]
  [green]>[/] Find Python developers with 3+ years and FastAPI experience
  [green]>[/] Compare the top 3 candidates side by side
  [green]>[/] Why did the top candidate rank higher than the second?
  [green]>[/] Generate interview questions for the top candidate
  [green]>[/] Run full multi-round screening on Python candidates
  [green]>[/] Make Docker mandatory and re-rank"""

DEMO_SCENARIOS = [
    ("New Search",       "Find Python developers with 3+ years experience in FastAPI and PostgreSQL."),
    ("Refinement",       "Make Docker mandatory and prioritize candidates with CI/CD experience."),
    ("Comparison",       "Compare the top 3 candidates side by side."),
    ("Explainability",   "Why did the top candidate rank higher than the second candidate?"),
    ("Interview Prep",   "Generate tailored interview questions for the top candidate."),
]

# ── Session state ─────────────────────────────────────────────────────────────
class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.thread_id = str(uuid.uuid4())
        self.history: list[dict] = []
        self.latest_state: Optional[dict] = None
        self.turn = 0

session = Session()

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe(text: str) -> str:
    """Escape Rich markup characters in any LLM-generated string."""
    if not isinstance(text, str):
        text = str(text)
    return escape(text)


def print_banner():
    console.print()
    console.print("[bold cyan]" + "=" * 70 + "[/]")
    console.print("[bold white]    AI Resume Job Matcher — Recruiter Agent[/]")
    console.print("[dim]    LangGraph  ·  RAG  ·  Hybrid Search  ·  Multi-Round Screening[/]")
    console.print("[bold cyan]" + "=" * 70 + "[/]")
    console.print(
        f"\n  [dim]Session:[/] [cyan]{session.thread_id[:20]}...[/]  "
        f"[dim]Time:[/] [cyan]{datetime.now().strftime('%H:%M:%S')}[/]\n"
    )


def print_user_turn(text: str):
    console.print()
    console.print(Panel(
        safe(text),
        title="[bold green] You [/]",
        border_style="green",
        expand=False,
        padding=(0, 2),
    ))


def print_agent_response(state: dict):
    """Render the agent's response safely — all LLM text is escaped."""
    report   = state.get("report", "") or ""
    intent   = state.get("current_intent", "unknown") or "unknown"
    ranked   = state.get("ranked_candidates", []) or []

    intent_colors = {
        "new_search": "cyan", "refine_requirements": "yellow",
        "compare_candidates": "magenta", "explain_ranking": "blue",
        "generate_interview_questions": "green", "full_screening": "red",
        "general_question": "white", "error": "red", "timeout": "red",
    }
    color = intent_colors.get(intent, "white")

    # ── Report panel — report text is fully escaped ──────────────────────────
    console.print()
    try:
        console.print(Panel(
            Text(report, overflow="fold"),        # Text() never parses markup
            title=f"[bold cyan] Agent [/]  [bold {color}][intent: {safe(intent)}][/bold {color}]",
            border_style="cyan",
            padding=(1, 2),
        ))
    except Exception:
        # Absolute fallback
        print("\n--- Agent Response ---")
        print(report[:3000])
        print("--- End ---\n")

    # ── Candidate table ───────────────────────────────────────────────────────
    if ranked:
        _print_candidates_table(ranked)


def _print_candidates_table(ranked: list):
    table = Table(
        title="[bold]Candidate Shortlist[/]",
        box=box.ROUNDED,
        border_style="dim cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#",           style="dim",       width=3,  justify="center")
    table.add_column("Name",        style="bold white", min_width=18)
    table.add_column("Score",       justify="center",  width=8)
    table.add_column("Exp (yrs)",   justify="center",  width=9)
    table.add_column("Matched Skills", min_width=30)
    table.add_column("Missing Skills", min_width=20, style="dim red")

    for i, c in enumerate(ranked[:10], 1):
        score   = c.get("match_score", 0)
        scolor  = "bold green" if score >= 70 else ("bold yellow" if score >= 50 else "bold red")
        matched = safe(", ".join(str(s) for s in c.get("matched_skills", [])[:5]))
        missing = safe(", ".join(str(s) for s in c.get("missing_skills", [])[:3]))
        name    = safe(str(c.get("candidate_name", "Unknown")))
        exp     = safe(str(c.get("experience_years", "?")))

        table.add_row(
            str(i), name,
            f"[{scolor}]{score}[/]",
            exp,
            matched or "[dim]—[/]",
            missing or "[dim green]None[/]",
        )

    console.print()
    try:
        console.print(table)
    except Exception:
        for i, c in enumerate(ranked[:10], 1):
            print(f"  #{i}: {c.get('candidate_name')}  score={c.get('match_score')}")


def print_history():
    if not session.history:
        console.print("[dim]No conversation history yet.[/]")
        return
    console.print(Rule("[bold]Conversation History[/]"))
    for i, msg in enumerate(session.history, 1):
        role  = "You" if msg["role"] == "user" else "Agent"
        style = "bold green" if msg["role"] == "user" else "bold cyan"
        prev  = safe((msg["content"] or "")[:180])
        console.print(f"  [{style}]{i:02}. {role}:[/] {prev}")
        console.print()


def print_state(state: dict):
    if not state:
        console.print("[dim]No agent state yet. Submit a query first.[/]")
        return
    console.print(Rule("[bold]Agent State Inspector[/]"))
    fields = {
        "Intent":           safe(str(state.get("current_intent", "—"))),
        "JD (raw)":         safe((state.get("raw_jd") or "")[:80]) + "...",
        "Candidates Found": str(len(state.get("ranked_candidates", []))),
        "Screening Round":  str(state.get("screening_round", 0)),
        "Screening Status": safe(str(state.get("screening_status", "NOT_STARTED"))),
        "Session Thread":   session.thread_id[:20] + "...",
    }
    for k, v in fields.items():
        console.print(f"  [bold cyan]{k}:[/] {v}")

    parsed = state.get("parsed_jd", {})
    if parsed:
        console.print()
        console.print("  [bold cyan]Parsed JD:[/]")
        for pk, pv in parsed.items():
            console.print(f"    [dim]{safe(str(pk))}:[/] {safe(str(pv))}")


def print_graph():
    console.print(Rule("[bold]LangGraph State Machine[/]"))
    try:
        graph_txt = get_graph_ascii_visualization()
        console.print(Panel(safe(graph_txt), border_style="dim", expand=False))
    except Exception as e:
        console.print(f"[dim](Graph error: {e})[/]")


# ── Spinner (simple thread-safe, no Rich Live) ────────────────────────────────

def _run_agent_with_spinner(user_input: str) -> Optional[dict]:
    """
    Run the agent in a background thread.
    Uses a simple print-based spinner that won't crash Windows terminals.
    """
    result_holder: dict = {}
    error_holder:  dict = {}

    def _worker():
        try:
            result_holder["state"] = run_agent(
                user_input, thread_id=session.thread_id
            )
        except BaseException as exc:        # catch SystemExit too
            import traceback
            error_holder["error"] = f"{exc}\n{traceback.format_exc()}"

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # ── Simple spinner that prints to stdout directly ─────────────────────────
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    steps  = [
        "Classifying intent",
        "Parsing job requirements",
        "Searching resume vector store",
        "Ranking candidates",
        "Generating report",
    ]
    step_interval = 4
    start  = time.time()
    fi     = 0

    while thread.is_alive():
        elapsed = time.time() - start
        step    = steps[min(int(elapsed / step_interval), len(steps) - 1)]
        frame   = frames[fi % len(frames)]
        fi += 1
        # \r rewrites the same line — plain stdout, no Rich markup parsing
        sys.stdout.write(f"\r  {frame}  {step}...  [{elapsed:.1f}s]  ")
        sys.stdout.flush()
        time.sleep(0.12)

    sys.stdout.write("\r" + " " * 60 + "\r")   # clear spinner line
    sys.stdout.flush()

    elapsed_total = time.time() - start

    if error_holder:
        console.print(f"\n[bold red]Agent Error ({elapsed_total:.1f}s):[/]")
        console.print(safe(str(error_holder["error"])[:1000]))
        return None

    state = result_holder.get("state")
    if state is None:
        console.print(f"\n[bold red]Agent returned nothing after {elapsed_total:.1f}s.[/]")
        console.print("[dim]Check OPENROUTER_API_KEY in your .env file.[/]")
        return None

    console.print(f"  [dim green]Done in {elapsed_total:.1f}s[/]")
    return state


# ── Demo runner ───────────────────────────────────────────────────────────────

def run_demo():
    console.print()
    console.print(Rule("[bold yellow]DEMO MODE — 5 Assignment Scenarios[/]"))
    for idx, (label, query) in enumerate(DEMO_SCENARIOS, 1):
        console.print(Rule(f"[bold]Scenario {idx}/5: {label}[/]"))
        print_user_turn(query)
        result = _run_agent_with_spinner(query)
        session.history.append({"role": "user",  "content": query})
        if result:
            session.latest_state = result
            session.history.append({"role": "agent", "content": result.get("report", "")})
            print_agent_response(result)
        time.sleep(1)
    console.print(Rule("[bold green]Demo Complete[/]"))


# ── Main REPL ─────────────────────────────────────────────────────────────────

def main():
    print_banner()
    console.print(HELP_TEXT)
    console.print()
    console.print(Rule(style="dim"))

    while True:
        try:
            console.print()
            user_input = Prompt.ask("[bold green]You[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()

        if cmd in ("/quit", "/exit", "quit", "exit", "q"):
            console.print("[dim]Session ended. Goodbye![/]")
            break
        elif cmd == "/new":
            session.reset()
            console.print(f"[bold green]New session.[/] Thread: [cyan]{session.thread_id[:20]}...[/]")
        elif cmd == "/history":
            print_history()
        elif cmd == "/state":
            print_state(session.latest_state)
        elif cmd == "/graph":
            print_graph()
        elif cmd == "/demo":
            run_demo()
        elif cmd in ("/help", "help", "?"):
            console.print(HELP_TEXT)
        else:
            # ── Agent query ──────────────────────────────────────────────────
            session.turn += 1
            print_user_turn(user_input)
            session.history.append({"role": "user", "content": user_input})

            try:
                result = _run_agent_with_spinner(user_input)
                if result is not None:
                    session.latest_state = result
                    report = result.get("report", "")
                    session.history.append({"role": "agent", "content": report})
                    print_agent_response(result)
            except BaseException as err:
                import traceback as _tb
                console.print(f"\n[bold red]Error:[/] {safe(str(err))}")
                console.print(f"[dim]{safe(_tb.format_exc()[:600])}[/]")
                console.print("[dim]Continue with another query or type /quit[/]")


if __name__ == "__main__":
    try:
        main()
    except BaseException as fatal:
        import traceback as _tb
        print(f"\nFATAL: {fatal}")
        print(_tb.format_exc())
        input("\nPress Enter to exit...")
