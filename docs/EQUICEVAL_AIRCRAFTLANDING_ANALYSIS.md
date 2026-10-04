# Vì sao EquiCEval trả `unresolved` cho 5 cặp AircraftLanding

> **Cập nhật:** case này đã được xử lý ở M1 bằng "phép chiếu có lượng hóa tồn tại"
> (xem `EquiCEval/docs/PROJECTION_EXISTENTIAL_PROPOSAL.md`). Sau khi mở rộng M1,
> 4/5 cặp chuyển `unresolved` → `disproved`, Recall EquiCEval 0.167 → 0.833, FPR
> giữ 0. Phần dưới giữ nguyên làm hồ sơ chẩn đoán gốc.

Ghi chú phân tích (Task 1, model `gpt-oss-120B`). Đây là giải thích cơ chế, kèm
bằng chứng.

## Hiện tượng

5/6 cặp AircraftLanding (P1, P2, P4, P5, P6) bị EquiCEval chấm `unresolved` với
`MapCoverage = 0.8` (12/15 biến), trong khi oracle thủ công kết luận **False**
(không tương đương). P3 (15 biến, đủ `z_ij`+`order_link`) thì EquiCEval chấm
`disproved` đúng.

## Chuỗi quyết định trong engine

1. `EquiCEvalEvaluator.evaluate` (`vendor/src/equiceval/evaluator.py:247`) gọi
   `build_projection_certificate`.
2. Certificate map 12 biến candidate → reference (`x_A1`→`x1`, `z_12`→`z12`, …).
   Reference còn 3 biến chưa map: `ref_only = {z21, z31, z32}`.
3. `_resolve_ref_eliminations` (`canonical_ir.py:814`) đề xuất `z_ji = 1 - z_ij`
   (complement) dựa trên chữ ký chỉ số đảo (`z21` ← reversed-sig → `z_12`).
4. `_verify_ref_eliminations` (`canonical_ir.py:794`) yêu cầu **mọi** ràng buộc
   reference dùng biến bị loại phải khớp một ràng buộc candidate.
5. Reference có `order_link_ij: z_ij + z_ji = 1`. Sau khi thay `z_ji = 1 - z_ij`,
   nó thành **tautology** `0 = 0` (`coeffs={}, constant=0`). Candidate **không có**
   ràng buộc tương ứng → `_constraints_equivalent` trả False.
6. `_resolve_ref_eliminations` trả `{}` → `build_projection_certificate` gọi
   `failed("Map does not cover reference variables")` (`canonical_ir.py:585,589`).
7. `cert.is_verified == False` → evaluator đặt `verdict = "unresolved"` và
   **return sớm** (`evaluator.py:256-259`), không chạy M2–M5.

`MapCoverage = 0.8` là do `variable_map` giữ lại 12 cặp map cục bộ (partial
coverage là lower bound), không phải 12/15 = verified.

## Đây là thiết kế cố ý, không phải bug

Test `../EquiCEval/tests/test_mapping_signatures.py:111`
(`test_linked_reference_rejects_candidate_missing_order_link`) khẳng định đúng
hành vi này: khi reference **có** `order_link`, candidate thiếu nó **phải** bị
reject. Complement-elimination chỉ được verify khi reference **không** có
`order_link` (`_unlinked_aircraft_landing_reference` ở `:22`).

Lý do bảo thủ: nếu không có bằng chứng ràng buộc tường minh ràng buộc hai chiều,
engine không được phép "đoán" quan hệ `z_ji = 1 - z_ij` chỉ từ tên. Nó đổi lại là
`unresolved` thay vì `certified`.

## Phân rã 5 cặp (khi vá tạm tautology để xem engine sẽ ra sao)

Tôi monkeypatch `_constraints_equivalent` để coi tautology là match vacuously, rồi
chạy lại (chỉ để điều tra, không commit):

| Cặp | Sau khi vá tautology | Nguyên nhân |
|---|---|---|
| P1 | `disproved` (đúng) | mapping pass; feasible-set witness |
| P4 | `disproved` (đúng) | mapping pass |
| P6 | `disproved` (đúng) | mapping pass |
| P2 | `unresolved` | fail `sep12_bwd`: dấu big-M khác thật |
| P5 | `unresolved` | mapping pass nhưng **solver timeout** (`bound_timeout=True`) |

Vậy có **3 nguyên nhân độc lập** khiến EquiCEval bỏ sót:

1. **Chủ ý bảo thủ** với `order_link` (P1, P2, P4, P5, P6 đều dừng ở đây).
2. **Khác dấu big-M thật** (P2) — đây là khác biệt ngữ nghĩa, engine đúng khi không
   certify, nhưng đáng ra feasibility check phải bắt được nếu mapping qua.
3. **Budget/timeout** (P5) — `solver_time_limit=5s`.

## Kết luận

- EquiCEval trả `unresolved` **không phải vì chấm sai lô-gic**, mà vì (a) nó chủ ý
  từ chối candidate thiếu `order_link`, rồi (b) khi thiếu mapping thì không chạy
  các tầng phát hiện lỗi. Đây là "abstain", không phải "pass".
- Về FPR/Recall: `unresolved` **không** được tính là đúng. Khi đối chiếu ground
  truth, nó là **miss** → Recall tụt còn 1/6.
- Muốn EquiCEval bắt được nhóm này cần thay đổi engine (thuộc repo `EquiCEval`,
  không phải oracle), ví dụ: cho phép eliminate binary khi `order_link` là
  tautology, hoặc project candidate sang không gian reference. Đây là thay đổi
  thiết kế có ảnh hưởng rộng → cần bàn riêng, không tự ý sửa.
