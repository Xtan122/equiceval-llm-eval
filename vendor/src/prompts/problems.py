"""
Natural language descriptions (problem texts) của 4 bài toán benchmark.
Copy từ Appendix của bài báo gốc (arXiv:2510.16943, Figures 3–6).

Đây là phần text được gửi cho LLM làm input trước prompt instruction.
"""

from typing import Dict

PROBLEM_DESCRIPTIONS: Dict[str, str] = {

    # ─── 1. Knapsack Problem ──────────────────────────────────────────────────
    "Knapsack": """\
Consider a knapsack problem. You have 3 items, each with a value and weight.
The knapsack has a maximum weight capacity of 50.

Items:
  - Item 1: value = 60, weight = 10
  - Item 2: value = 100, weight = 20
  - Item 3: value = 120, weight = 30

Decision: For each item, decide whether to include it (1) or not (0).
Objective: Maximize the total value of selected items.
Constraint: Total weight of selected items must not exceed the capacity.
Variable type: binary (0 or 1).
""",

    # ─── 2. Aircraft Assignment Problem ──────────────────────────────────────
    "AircraftAssignment": """\
Consider an aircraft assignment problem.
There are 3 aircraft (A1, A2, A3) and 2 routes (R1, R2).
Each aircraft can be assigned to each route (binary decision).

Availability (max assignments per aircraft):
  - A1: at most 2 total assignments
  - A2: at most 3 total assignments
  - A3: at most 1 total assignment

Demand (minimum capacity per route, in units):
  - Route R1 requires at least 100 units
  - Route R2 requires at least 150 units

Capacity provided (units per assignment):
  - A1 on R1: 50, A1 on R2: 70
  - A2 on R1: 60, A2 on R2: 80
  - A3 on R1: 70, A3 on R2: 90

Operational costs (minimize):
  - A1 on R1: 100, A1 on R2: 200
  - A2 on R1: 150, A2 on R2: 250
  - A3 on R1: 200, A3 on R2: 300

Decision variables: x_{i,j} = 1 if aircraft i assigned to route j, else 0 (binary).
Objective: Minimize total operational cost.
""",

    # ─── 3. Diet Optimization Problem ────────────────────────────────────────
    "Diet": """\
Consider a diet optimization problem.
Foods available: Apple (x_apple), Banana (x_banana).
Nutrients tracked: Vitamin C, Fiber.

Nutritional content per unit:
  - Apple:  Vitamin C = 10, Fiber = 5
  - Banana: Vitamin C = 5,  Fiber = 10

Nutrient requirements:
  - Vitamin C: at least 50, at most 100
  - Fiber:     at least 30, at most 60

Food purchase bounds:
  - x_apple:  between 0 and 10 (continuous)
  - x_banana: between 0 and 10 (continuous)

Food costs:
  - Apple:  2.0 per unit
  - Banana: 1.5 per unit

Objective: Minimize total food cost while satisfying all nutritional requirements.
""",

    # ─── 4. Aircraft Landing Problem ─────────────────────────────────────────
    "AircraftLanding": """\
Consider an aircraft landing scheduling problem.
There are 3 aircraft (A1, A2, A3) that must land within their time windows.

Time windows [Earliest, Latest] and Target times:
  - A1: [1, 10], Target = 4
  - A2: [3, 12], Target = 8
  - A3: [5, 15], Target = 14

Penalty costs (per unit time deviation):
  - A1: Early = 5,  Late = 10
  - A2: Early = 10, Late = 20
  - A3: Early = 15, Late = 30

Minimum separation times (in time units):
  - Between A1 and A2: 2
  - Between A1 and A3: 3
  - Between A2 and A3: 4
  (symmetric: same time required regardless of order)

Decision variables:
  - x_i: actual landing time of aircraft i (continuous)
  - e_i: earliness of aircraft i (continuous, >= 0)
  - l_i: lateness  of aircraft i (continuous, >= 0)
  - z_{i,j}: 1 if aircraft i lands before j, 0 otherwise (binary)

Use Big-M formulation (M = 1000) for separation constraints.

Objective: Minimize total penalty = sum of (early_penalty * e_i + late_penalty * l_i).
""",
}
