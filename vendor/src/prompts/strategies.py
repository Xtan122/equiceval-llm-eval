"""
6 Prompt Templates từ bài báo gốc — Refai & Ahmed (arXiv:2510.16943).
Copy chính xác từ Figure 2 (Section 4.2) của paper.

Mỗi prompt được ghép với problem_text từ problems.py và JSON_SCHEMA_SUFFIX
trước khi gửi cho LLM.
"""

from typing import Dict

# ── Suffix thêm vào cuối mọi prompt ─────────────────────────────────────────
# Yêu cầu LLM output JSON chuẩn để output_parser có thể parse được.
JSON_SCHEMA_SUFFIX = """

Output ONLY a valid JSON object with exactly this structure (no explanation, no LaTeX):
{
  "variables": {
    "<var_name>": {"type": "binary|integer|continuous", "lb": <number>, "ub": <number or null>}
  },
  "objective": {
    "sense": "minimize|maximize",
    "coefficients": {"<var_name>": <number>}
  },
  "constraints": [
    {
      "name": "<constraint_name>",
      "coefficients": {"<var_name>": <number>},
      "sense": "<=|>=|==",
      "rhs": <number>
    }
  ]
}
"""

# ── 6 Prompt Templates (P1–P6) ───────────────────────────────────────────────
PROMPT_TEMPLATES: Dict[str, str] = {

    "P1": (
        "Solve the above optimization problem.\n"
        "- Step 1: Identify the decision variables, objective function, and constraints.\n"
        "- Step 2: Write the complete mathematical formulation. "
        "Do not include any explanation, only output the formulation."
        + JSON_SCHEMA_SUFFIX
    ),

    "P2": (
        "As an expert in optimization, solve the problem by:\n"
        "- Defining the decision variables.\n"
        "- Formulating the objective function.\n"
        "- Listing the constraints.\n"
        "Then, provide the complete formulation. "
        "Do not include any explanation, only output the formulation."
        + JSON_SCHEMA_SUFFIX
    ),

    "P3": (
        "Solve the problem by thinking step-by-step:\n"
        "- Identify the objective.\n"
        "- Define the decision variables.\n"
        "- Formulate the objective function.\n"
        "- List the constraints.\n"
        "Finally, present the full formulation. "
        "Do not include any explanation, only output the formulation."
        + JSON_SCHEMA_SUFFIX
    ),

    "P4": (
        "Use a logic-based reasoning process:\n"
        "- Define decision variables.\n"
        "- Specify the objective function.\n"
        "- Add constraints.\n"
        "- Identify the model type. "
        "Do not include any explanation, only output the formulation."
        + JSON_SCHEMA_SUFFIX
    ),

    "P5": (
        "Generate three formulations independently for the same optimization problem:\n"
        "- Each should include (decision variables, objective function, and constraints).\n"
        "Internally compare and select the best one.\n"
        "Output only the final selected formulation. "
        "Do not include any explanation or mention of other formulations."
        + JSON_SCHEMA_SUFFIX
    ),

    "P6": (
        "Step 1: Identify and define ONLY the decision variables. "
        "Do not solve or explain anything else.\n"
        "Step 2: Write ONLY the mathematical expression for the objective function. "
        "Do not explain or solve.\n"
        "Step 3: Write ONLY the mathematical constraints from the problem. "
        "Do not solve or interpret.\n"
        "Step 4: Combine the decision variables, objective function, and constraints "
        "into a complete optimization model. Do not explain or solve it."
        + JSON_SCHEMA_SUFFIX
    ),
}
