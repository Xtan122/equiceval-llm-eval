"""
Mock LLM Client — Hardcode các output LLM theo hành vi đã biết từ bài báo gốc.

Không gọi bất kỳ API nào. Dùng khi:
  - Không có API key
  - Test offline
  - Reproducing paper results mà không tốn API cost

Behavior được mô hình hóa dựa trên Refai & Ahmed (2025):
  - GPT-5 + P5/P6  → Perfect formulation
  - DeepSeek + P2  → Scale coefficient ×2
  - LLaMA + P1/P2  → Thêm non-negativity constraints dư (FP)
  - LLaMA/DeepSeek + AircraftLanding → Bỏ Big-M trong separation
"""

import json
import time

from .base import LLMClient, LLMResponse

# ─── Mock output JSON cho từng (model, prompt, problem) ─────────────────────
# Format: CanonicalIR JSON (được định nghĩa bởi output_parser.FormulationParser)

_KNAPSACK_PERFECT = {
    "variables": {"x_1": {"type": "binary"}, "x_2": {"type": "binary"}, "x_3": {"type": "binary"}},
    "objective": {"sense": "maximize", "coefficients": {"x_1": 60.0, "x_2": 100.0, "x_3": 120.0}},
    "constraints": [
        {"name": "capacity", "coefficients": {"x_1": 10.0, "x_2": 20.0, "x_3": 30.0}, "sense": "<=", "rhs": 50.0}
    ],
}

_KNAPSACK_LLAMA_FP = {  # LLaMA P1/P2: thêm non_neg_x1 dư (False Positive)
    "variables": {"x_1": {"type": "binary"}, "x_2": {"type": "binary"}, "x_3": {"type": "binary"}},
    "objective": {"sense": "maximize", "coefficients": {"x_1": 60.0, "x_2": 100.0, "x_3": 120.0}},
    "constraints": [
        {"name": "capacity",    "coefficients": {"x_1": 10.0, "x_2": 20.0, "x_3": 30.0}, "sense": "<=", "rhs": 50.0},
        {"name": "non_neg_x1", "coefficients": {"x_1": -1.0},                              "sense": "<=", "rhs": 0.0},
    ],
}

_KNAPSACK_DEEPSEEK_SCALED = {  # DeepSeek P2: scale ×2 (toán học tương đương nhưng RMSE cao)
    "variables": {"x_1": {"type": "binary"}, "x_2": {"type": "binary"}, "x_3": {"type": "binary"}},
    "objective": {"sense": "maximize", "coefficients": {"x_1": 60.0, "x_2": 100.0, "x_3": 120.0}},
    "constraints": [
        {"name": "scaled_cap", "coefficients": {"x_1": 20.0, "x_2": 40.0, "x_3": 60.0}, "sense": "<=", "rhs": 100.0}
    ],
}

_AA_PERFECT = {
    "variables": {f"x_{a}_{r}": {"type": "binary"} for a in ["1","2","3"] for r in ["1","2"]},
    "objective": {"sense": "minimize", "coefficients": {
        "x_1_1": 100.0, "x_1_2": 200.0, "x_2_1": 150.0,
        "x_2_2": 250.0, "x_3_1": 200.0, "x_3_2": 300.0,
    }},
    "constraints": [
        {"name": "avail_A1", "coefficients": {"x_1_1": 1.0, "x_1_2": 1.0}, "sense": "<=", "rhs": 2.0},
        {"name": "avail_A2", "coefficients": {"x_2_1": 1.0, "x_2_2": 1.0}, "sense": "<=", "rhs": 3.0},
        {"name": "avail_A3", "coefficients": {"x_3_1": 1.0, "x_3_2": 1.0}, "sense": "<=", "rhs": 1.0},
        {"name": "demand_R1", "coefficients": {"x_1_1": 50.0, "x_2_1": 60.0, "x_3_1": 70.0}, "sense": ">=", "rhs": 100.0},
        {"name": "demand_R2", "coefficients": {"x_1_2": 70.0, "x_2_2": 80.0, "x_3_2": 90.0}, "sense": ">=", "rhs": 150.0},
    ],
}

_AA_LLAMA_MISSING_DEMAND = {  # LLaMA P1/P2/P4: bỏ demand_R2
    **{k: v for k, v in _AA_PERFECT.items() if k != "constraints"},
    "constraints": _AA_PERFECT["constraints"][:4],  # thiếu demand_R2
}

_DIET_PERFECT = {
    "variables": {"x_apple": {"type": "continuous", "lb": 0.0, "ub": 10.0},
                  "x_banana": {"type": "continuous", "lb": 0.0, "ub": 10.0}},
    "objective": {"sense": "minimize", "coefficients": {"x_apple": 2.0, "x_banana": 1.5}},
    "constraints": [
        {"name": "min_vit_c", "coefficients": {"x_apple": 10.0, "x_banana": 5.0}, "sense": ">=", "rhs": 50.0},
        {"name": "min_fiber", "coefficients": {"x_apple": 5.0,  "x_banana": 10.0}, "sense": ">=", "rhs": 30.0},
        {"name": "max_vit_c", "coefficients": {"x_apple": 10.0, "x_banana": 5.0}, "sense": "<=", "rhs": 100.0},
        {"name": "max_fiber", "coefficients": {"x_apple": 5.0,  "x_banana": 10.0}, "sense": "<=", "rhs": 60.0},
    ],
}

_DIET_DEEPSEEK_MISSING_MAX = {  # DeepSeek P2/P4: bỏ max nutrient constraints
    **{k: v for k, v in _DIET_PERFECT.items() if k != "constraints"},
    "constraints": _DIET_PERFECT["constraints"][:2],
}

_ALP_PERFECT = {
    "variables": {
        "x_1": {"type": "continuous", "lb": 1.0, "ub": 10.0},
        "x_2": {"type": "continuous", "lb": 3.0, "ub": 12.0},
        "x_3": {"type": "continuous", "lb": 5.0, "ub": 15.0},
        "e_1": {"type": "continuous", "lb": 0.0}, "e_2": {"type": "continuous", "lb": 0.0},
        "e_3": {"type": "continuous", "lb": 0.0}, "l_1": {"type": "continuous", "lb": 0.0},
        "l_2": {"type": "continuous", "lb": 0.0}, "l_3": {"type": "continuous", "lb": 0.0},
        "z_1_2": {"type": "binary"}, "z_2_1": {"type": "binary"},
        "z_1_3": {"type": "binary"}, "z_3_1": {"type": "binary"},
        "z_2_3": {"type": "binary"}, "z_3_2": {"type": "binary"},
    },
    "objective": {"sense": "minimize", "coefficients": {
        "e_1": 5.0, "l_1": 10.0, "e_2": 10.0, "l_2": 20.0, "e_3": 15.0, "l_3": 30.0
    }},
    "constraints": [
        {"name": "earliness_A1", "coefficients": {"x_1": -1.0, "e_1": -1.0}, "sense": "<=", "rhs": -4.0},
        {"name": "lateness_A1",  "coefficients": {"x_1":  1.0, "l_1": -1.0}, "sense": "<=", "rhs": 4.0},
        {"name": "sep_1_2", "coefficients": {"x_1": 1.0, "x_2": -1.0, "z_1_2": 1000.0}, "sense": "<=", "rhs": 998.0},
    ],
}

_ALP_BROKEN_BIGM = {  # LLaMA/DeepSeek P1/P2: Big-M bị xóa
    **{k: v for k, v in _ALP_PERFECT.items() if k != "constraints"},
    "constraints": [
        {"name": "earliness_A1", "coefficients": {"x_1": -1.0, "e_1": -1.0}, "sense": "<=", "rhs": -4.0},
        {"name": "lateness_A1",  "coefficients": {"x_1":  1.0, "l_1": -1.0}, "sense": "<=", "rhs": 4.0},
        {"name": "sep_1_2_bad",  "coefficients": {"x_1":  1.0, "x_2": -1.0}, "sense": "<=", "rhs": 2.0},  # missing Big-M
    ],
}

# ─── Lookup table: (model_key, prompt, problem) → (json_dict, has_error, error_type)
_MOCK_BEHAVIOR = {
    # Knapsack
    ("llama",     "Knapsack", "P1"): (_KNAPSACK_LLAMA_FP,       True,  "extra_non_neg"),
    ("llama",     "Knapsack", "P2"): (_KNAPSACK_LLAMA_FP,       True,  "extra_non_neg"),
    ("deepseek",  "Knapsack", "P2"): (_KNAPSACK_DEEPSEEK_SCALED, True, "scaled_x2"),
    # AircraftAssignment
    ("llama",     "AircraftAssignment", "P1"): (_AA_LLAMA_MISSING_DEMAND, True, "omitted_demand_R2"),
    ("llama",     "AircraftAssignment", "P2"): (_AA_LLAMA_MISSING_DEMAND, True, "omitted_demand_R2"),
    ("llama",     "AircraftAssignment", "P4"): (_AA_LLAMA_MISSING_DEMAND, True, "omitted_demand_R2"),
    # Diet
    ("deepseek",  "Diet", "P2"): (_DIET_DEEPSEEK_MISSING_MAX, True, "omitted_max_nutrients"),
    ("deepseek",  "Diet", "P4"): (_DIET_DEEPSEEK_MISSING_MAX, True, "omitted_max_nutrients"),
    # AircraftLanding
    ("llama",     "AircraftLanding", "P1"): (_ALP_BROKEN_BIGM, True, "broken_big_M"),
    ("llama",     "AircraftLanding", "P2"): (_ALP_BROKEN_BIGM, True, "broken_big_M"),
    ("deepseek",  "AircraftLanding", "P1"): (_ALP_BROKEN_BIGM, True, "broken_big_M"),
    ("deepseek",  "AircraftLanding", "P2"): (_ALP_BROKEN_BIGM, True, "broken_big_M"),
}

# Latency/token stats từ paper (Table 4-7 approximate)
_MOCK_STATS = {
    "gpt":      {"latency": 2.0,  "in_tok": 137, "out_tok": 45},
    "gemini":   {"latency": 1.5,  "in_tok": 137, "out_tok": 48},
    "llama":    {"latency": 7.97, "in_tok": 137, "out_tok": 251},
    "deepseek": {"latency": 5.79, "in_tok": 137, "out_tok": 205},
}


def _model_key(model_name: str) -> str:
    name = model_name.lower()
    if "gpt" in name:     return "gpt"
    if "gemini" in name:  return "gemini"
    if "llama" in name:   return "llama"
    if "deepseek" in name: return "deepseek"
    return "gpt"


class MockClient(LLMClient):
    """
    Không gọi API. Trả về JSON hardcode mô phỏng behavior LLM.
    Dùng cho test offline và reproducing paper results.
    """

    provider = "mock"

    def __init__(self, model_name: str = "GPT-5", problem_name: str = "Knapsack", prompt_id: str = "P1"):
        self.model_name   = model_name
        self.problem_name = problem_name
        self.prompt_id    = prompt_id

    def call(self, user_message: str = "") -> LLMResponse:
        key = _model_key(self.model_name)
        lookup = (key, self.problem_name, self.prompt_id)

        # Lấy formulation json từ bảng, fallback về perfect
        _perfect_map = {
            "Knapsack": _KNAPSACK_PERFECT,
            "AircraftAssignment": _AA_PERFECT,
            "Diet": _DIET_PERFECT,
            "AircraftLanding": _ALP_PERFECT,
        }
        json_dict, has_error, error_type = _MOCK_BEHAVIOR.get(
            lookup, (_perfect_map.get(self.problem_name, _KNAPSACK_PERFECT), False, "none")
        )

        stats = _MOCK_STATS.get(key, _MOCK_STATS["gpt"])
        time.sleep(0)   # No actual wait

        return LLMResponse(
            raw_text=json.dumps(json_dict),
            latency_s=stats["latency"],
            input_tokens=stats["in_tok"] + len(self.prompt_id),
            output_tokens=stats["out_tok"],
            model_name=self.model_name,
            provider=self.provider,
        )
