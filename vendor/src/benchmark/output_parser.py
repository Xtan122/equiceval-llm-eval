"""
Output Parser — Chuyển đổi LLM raw text → các thành phần formulation.

LLM được yêu cầu output JSON (qua JSON_SCHEMA_SUFFIX trong strategies.py).
Parser tìm JSON block trong text và extract ra:
  - candidate_variables (set of str)
  - candidate_constraints (list of (name, lambda))
  - objective_coeffs (dict)
  - objective_sense (str)

Nếu parse thất bại → trả về None và log lý do.
"""

import json
import re
import logging
from typing import Optional, Tuple, Set, List, Dict, Any

logger = logging.getLogger(__name__)


class ParsedFormulation:
    """Kết quả parse từ LLM output."""

    def __init__(self, raw: Dict[str, Any]):
        self._raw = raw

    @property
    def variables(self) -> Set[str]:
        return set(self._raw.get("variables", {}).keys())

    @property
    def objective_sense(self) -> str:
        return self._raw.get("objective", {}).get("sense", "minimize")

    @property
    def objective_coeffs(self) -> Dict[str, float]:
        return {k: _to_float(v, 0.0) for k, v in
                self._raw.get("objective", {}).get("coefficients", {}).items()}

    @property
    def constraints_as_funcs(self) -> List[Tuple[str, Any]]:
        """
        Trả về list of (name, expression_func) theo chuẩn BaselineEvaluator.
        expression_func(x) = sum(coeff_i * x_i) - rhs   (≤ 0 khi thỏa mãn)
        """
        result = []
        for c in self._raw.get("constraints", []):
            name   = c.get("name", "unnamed")
            coeffs = {k: _to_float(v, 0.0) for k, v in c.get("coefficients", {}).items()}
            sense  = c.get("sense", "<=")
            # A non-numeric rhs (e.g. the raw string "x_A1 - 1") is malformed
            # output: coerce to 0.0 rather than crashing the whole run. Such a
            # record is still counted (worst case) and surfaced via the metrics.
            rhs    = _to_float(c.get("rhs"), 0.0)

            # Chuẩn hóa về dạng LHS - RHS ≤ 0
            if sense == "<=":
                func = _make_leq_func(coeffs, rhs)
            elif sense == ">=":
                # a >= b  ↔  -a ≤ -b  ↔  (-a) - (-b) ≤ 0
                func = _make_leq_func({k: -v for k, v in coeffs.items()}, -rhs)
            else:  # "=" treated as two constraints, take <= side
                func = _make_leq_func(coeffs, rhs)

            result.append((name, func))
        return result

    @property
    def variable_specs(self) -> Dict[str, Tuple[str, float, float]]:
        """Returns {var_name: (type, lb, ub)} for solver."""
        specs = {}
        for vname, vspec in self._raw.get("variables", {}).items():
            if isinstance(vspec, str):
                vtype = vspec.lower()
                lb, ub = 0.0, float("inf")
            else:
                vtype = vspec.get("type", "continuous").lower()
                # LLMs often emit `lb: null` / `ub: null` to mean "unbounded".
                lb = _to_float(vspec.get("lb"), 0.0)
                ub = _to_float(vspec.get("ub"), float("inf"))

            if vtype in ("binary", "bin"):
                specs[vname] = ("binary", 0.0, 1.0)
            elif vtype in ("integer", "int"):
                specs[vname] = ("integer", lb, ub)
            else:
                specs[vname] = ("continuous", lb, ub)
        return specs

    def constraints_for_solver(self) -> List[Tuple[Dict[str, float], str, float]]:
        """Returns list of (coeffs, sense, rhs) for UnifiedSolver."""
        result = []
        for c in self._raw.get("constraints", []):
            coeffs = {k: _to_float(v, 0.0) for k, v in c.get("coefficients", {}).items()}
            sense  = c.get("sense", "<=")
            if sense == "=":
                sense = "=="
            rhs    = _to_float(c.get("rhs"), 0.0)
            result.append((coeffs, sense, rhs))
        return result


def _to_float(value: Any, default: float) -> float:
    """Coerce a JSON value (number, numeric string, or None) to a float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _make_leq_func(coeffs: Dict[str, float], rhs: float):
    """Tạo lambda: f(x) = sum(c_i * x_i) - rhs. Thỏa mãn khi ≤ 0."""
    def func(x: Dict[str, float]) -> float:
        return sum(coeffs.get(v, 0.0) * x.get(v, 0.0) for v in coeffs) - rhs
    return func


class FormulationParser:
    """Parse LLM raw text → ParsedFormulation."""

    @staticmethod
    def parse(raw_text: str) -> Optional[ParsedFormulation]:
        """
        Tìm JSON block trong raw_text và parse.
        Trả về None nếu không tìm thấy hoặc JSON sai format.
        """
        json_dict = FormulationParser._extract_json(raw_text)
        if json_dict is None:
            logger.warning("FormulationParser: Không tìm thấy JSON trong output LLM.")
            logger.debug("Raw text: %s", raw_text[:300])
            return None

        # Validate các trường bắt buộc
        if "variables" not in json_dict:
            logger.warning("FormulationParser: Thiếu trường 'variables'.")
            return None
        if "objective" not in json_dict:
            logger.warning("FormulationParser: Thiếu trường 'objective'.")
            return None
        if "constraints" not in json_dict:
            logger.warning("FormulationParser: Thiếu trường 'constraints'.")
            return None

        return ParsedFormulation(json_dict)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Tìm JSON object đầu tiên trong text."""
        # Thử parse thẳng
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Tìm block ```json ... ``` (markdown code block)
        md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if md_match:
            try:
                return json.loads(md_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Tìm JSON object đầu tiên với regex { ... }
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        return None
