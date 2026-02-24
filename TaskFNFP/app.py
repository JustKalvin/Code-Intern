import asyncio
import os
import json
import csv
from dotenv import load_dotenv
from gllm_inference.lm_invoker import OpenAILMInvoker
from gllm_inference.model import OpenAILM
from gllm_inference.prompt_builder import PromptBuilder
from gllm_inference.request_processor import LMRequestProcessor
from gllm_core.utils.retry import RetryConfig
from pydantic import BaseModel, Field

load_dotenv()


# ─────────────────────────────────────────────
# 1. PYDANTIC SCHEMA
# ─────────────────────────────────────────────
class Output(BaseModel):
    fp_risk: str = Field(description="Risk level: High / Medium / Low")
    fp_reason: str = Field(description="Analysis of why this task is vulnerable to False Positives")
    fp_solution: str = Field(description="Proposed solution to mitigate the FP vulnerability")
    fn_risk: str = Field(description="Risk level: High / Medium / Low")
    fn_reason: str = Field(description="Analysis of why this task is vulnerable to False Negatives")
    fn_solution: str = Field(description="Proposed solution to mitigate the FN vulnerability")


# ─────────────────────────────────────────────
# 2. COMPACT MESSAGE EXTRACTOR
# ─────────────────────────────────────────────
def extract_compact_messages(simulation_data: dict) -> str:
    """Extracts and formats messages from a simulation into a compact transcript string."""
    messages = simulation_data.get("messages", [])
    compact_lines = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "user":
            compact_lines.append(f"USER: {content}")

        elif role == "assistant":
            # Check for tool calls
            tool_calls = msg.get("tool_calls", None)
            if content:
                compact_lines.append(f"ASSISTANT: {content}")
            if tool_calls:
                tool_descs = []
                for tc in tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("arguments", {})
                    args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
                    tool_descs.append(f"{name}({args_str})")
                compact_lines.append(f"ASSISTANT (Action): {' | '.join(tool_descs)}")

        elif role == "tool":
            truncated = content[:150] + "..." if len(content) > 150 else content
            compact_lines.append(f"TOOL RESULT: {truncated}")

    return "\n".join(compact_lines)


# ─────────────────────────────────────────────
# 3. SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
### ROLE
You are a Senior AI Quality Assurance Engineer auditing the Tau-Bench evaluation dataset.

### OBJECTIVE
You are provided with a Task's details, its active Evaluator Group (e.g., ACTION + COMMUNICATE), and a compact Simulation Transcript.
Your goal is to identify specific vulnerabilities of this task in terms of False Positives (FP) and False Negatives (FN), heavily grounding your analysis in the provided Transcript and Task Scenario.

### DEFINITIONS
- **False Positive (FP):** The evaluator blindly gives a passing score (1.0) even though the agent fails the real intent, hallucinates info, gives incorrect data, or the user manipulates the interaction.
- **False Negative (FN):** The evaluator strictly gives a failing score (0.0) even though the agent successfully solves the problem in a natural, creative, or alternative valid way.

### LOGIC RULES PER EVALUATOR GROUP
Apply the vulnerability pattern that exactly matches the task's Evaluator Group:

1. **ACTION Only:** FP risk if the agent hallucinates incorrect details in its text *after* a technically correct API call — the evaluator only checks the call, not the spoken content. FN risk if the agent achieves the goal using a valid *alternative* API sequence that doesn't exactly match the ground-truth action list.

2. **ACTION + COMMUNICATE:** FP risk if the target substring (e.g., "10") appears incidentally in a wrong context (e.g., "10 days" instead of "$10"). FN risk if the agent gives a semantically correct answer but uses a different format or synonym that fails the strict substring check (e.g., "ten" instead of "10").

3. **ACTION + NL_ASSERTIONS:** FP risk if the agent is gradually manipulated mid-conversation into violating policy, but the final ACTION check still passes and the judge only scored the early polite refusal. FN risk if the agent enforces policy correctly but uses concise or unconventional phrasing that the LLM judge penalizes.

4. **ACTION + NL_ASSERTIONS + COMMUNICATE:** High compound risk. FP risk if the agent mixes hallucinated data with polite language, satisfying the NL and substring checks while failing in substance. FN risk from brittle multi-check failures — any single mismatch in phrasing, format, or synonym causes a full failing score.

5. **ACTION + ENV_ASSERTIONS:** FP risk if the environment state is altered by external/system artifacts unrelated to the agent's actions, causing a spurious pass. FN risk if the agent fixes the core task correctly but skips a secondary diagnostic API call, causing the env assertion to fail.

6. **COMMUNICATE Only:** FP risk if the agent produces a wrong answer but the expected substring appears incidentally in an unrelated sentence. FN risk if the agent gives a semantically correct answer using a valid synonym that fails the exact substring match.

7. **NL_ASSERTIONS Only:** FP risk if the agent is socially engineered into violating policy mid-conversation while still using the LLM judge's expected trigger words. FN risk if the agent correctly and concisely denies the request but the LLM judge expects a more elaborate or specifically worded justification.

8. **No Evaluator (NONE / Empty):** FP risk is high — any agent output, even incorrect, is undetected without an automated check. FN risk is high — a correct but unconventional solution has no rubric to validate it. Flag this as a coverage gap in the evaluation design.

### IMPORTANT: SOLUTION SCOPE
- `fp_solution` and `fn_solution` MUST propose fixes to the **evaluation rubric or evaluator logic** — NOT changes to the agent's behavior.
- Valid solution examples: "Add a context-aware substring check to COMMUNICATE", "Add an NL assertion for policy-adherence verification", "Add a secondary env assertion to check state X".

### INSTRUCTIONS
1. Read the TASK ID, DESCRIPTION, USER SCENARIO, and SIMULATION TRANSCRIPT carefully.
2. Identify the EVALUATOR GROUP from the input.
3. Match it to the relevant rule above.
4. Determine FP and FN risks (High / Medium / Low) based on how the agent and user ACTUALLY interacted in the transcript.
5. Ground your reasoning in the transcript — quote specific agent/user turns that trigger the risk.
6. Propose an evaluation-side fix (rubric or assertion change) for each identified gap.
7. Output strictly adhering to the provided JSON schema.\
"""


# ─────────────────────────────────────────────
# 4. EVALUATOR GROUP BUILDER
# ─────────────────────────────────────────────
EVALUATOR_KEY_MAP = {
    "actions": "ACTION",
    "communicate_info": "COMMUNICATE",
    "nl_assertions": "NL_ASSERTIONS",
    "env_assertions": "ENV_ASSERTIONS",
}


def build_evaluator_group(criteria: dict) -> str:
    """Builds a string like 'ACTION + COMMUNICATE' from non-empty evaluation_criteria keys."""
    parts = []
    for key, label in EVALUATOR_KEY_MAP.items():
        value = criteria.get(key)
        if value:  # not None, not empty list/dict/string
            parts.append(label)
    return " + ".join(parts) if parts else "NONE"


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
CSV_FILE = "TaskFNFP\evaluation_reports_v2.csv"
CSV_COLUMNS = [
    "id", "description", "user_scenario", "simulation_transcript",
    "evaluator_group", "actions", "communicate_info",
    "nl_assertions", "env_assertions",
    "fp_risk", "fp_reason", "fp_solution",
    "fn_risk", "fn_reason", "fn_solution",
]


async def main():
    # ── Load dataset ──
    file_path = "TaskFNFP\claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json"

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks_array = data.get("tasks", [])
    simulations_array = data.get("simulations", [])

    # ── Build index: task_id -> first matching simulation ──
    sim_index: dict = {}
    for sim in simulations_array:
        tid = sim.get("task_id")
        if tid and tid not in sim_index:
            sim_index[tid] = sim

    # ── Check existing IDs in CSV (for resume/append) ──
    existing_ids: set[str] = set()
    file_exists = os.path.isfile(CSV_FILE)

    if file_exists:
        with open(CSV_FILE, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if "id" in row:
                    existing_ids.add(str(row["id"]))
        print(f"Found {len(existing_ids)} existing IDs in '{CSV_FILE}'.")
    else:
        print(f"'{CSV_FILE}' does not exist yet. Will create on first write.")

    # ── Setup GLLM pipeline ──
    retry_config = RetryConfig(max_retries=3, timeout=100)
    lm_invoker = OpenAILMInvoker(
        OpenAILM.GPT_5_NANO,
        api_key=os.getenv("OPENAI_API_KEY"),
        response_schema=Output,
        retry_config=retry_config,
    )
    prompt_builder = PromptBuilder(
        system_template=SYSTEM_PROMPT,
        user_template="{query}",
    )
    lm_request_processor = LMRequestProcessor(
        prompt_builder=prompt_builder,
        lm_invoker=lm_invoker,
    )

    # ── Process each task ──
    def safe_serialize(obj) -> str:
        return json.dumps(obj) if isinstance(obj, (dict, list)) else str(obj)

    for task in tasks_array[:5]:
        task_id = task.get("id")
        print(f"Processing Task ID: {task_id}...")

        if str(task_id) in existing_ids:
            print(f"  ↳ Task ID '{task_id}' already in CSV. Skipped.")
            continue

        # ── Extract fields ──
        description = safe_serialize(task.get("description", ""))
        user_scenario = safe_serialize(task.get("user_scenario", ""))

        criteria = task.get("evaluation_criteria", {})
        evaluator_group = build_evaluator_group(criteria)

        actions_raw = criteria.get("actions")
        communicate_raw = criteria.get("communicate_info")
        nl_raw = criteria.get("nl_assertions")
        env_raw = criteria.get("env_assertions")

        actions_str = safe_serialize(actions_raw) if actions_raw else "N/A"
        communicate_str = safe_serialize(communicate_raw) if communicate_raw else "N/A"
        nl_str = safe_serialize(nl_raw) if nl_raw else "N/A"
        env_str = safe_serialize(env_raw) if env_raw else "N/A"

        # ── Find matching simulation ──
        sim = sim_index.get(task_id)
        simulation_transcript = extract_compact_messages(sim) if sim else "No simulation found."

        # ── Build user prompt ──
        query = (
            f"Task ID: {task_id}\n"
            f"Description: {description}\n"
            f"User Scenario: {user_scenario}\n"
            f"Evaluator Group: {evaluator_group}\n"
            f"Simulation Transcript:\n{simulation_transcript}"
        )

        # ── Call LLM ──
        try:
            result = await lm_request_processor.process(query=query)
            out: Output = result.outputs[0].output

            row = {
                "id": task_id,
                "description": description,
                "user_scenario": user_scenario,
                "simulation_transcript": simulation_transcript,
                "evaluator_group": evaluator_group,
                "actions": actions_str,
                "communicate_info": communicate_str,
                "nl_assertions": nl_str,
                "env_assertions": env_str,
                "fp_risk": out.fp_risk,
                "fp_reason": out.fp_reason,
                "fp_solution": out.fp_solution,
                "fn_risk": out.fn_risk,
                "fn_reason": out.fn_reason,
                "fn_solution": out.fn_solution,
            }

            # ── Append to CSV ──
            try:
                with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvf:
                    writer = csv.DictWriter(csvf, fieldnames=CSV_COLUMNS)
                    if not file_exists:
                        writer.writeheader()
                        file_exists = True  # only write header once
                    writer.writerow(row)
                print(f"  ✓ Appended to '{CSV_FILE}'")
            except Exception as e:
                print(f"  ✗ CSV write error: {e}")

        except Exception as e:
            print(f"  ✗ LLM processing failed for task '{task_id}': {e}")

    print("\n✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())
