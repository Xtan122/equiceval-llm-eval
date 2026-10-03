# Task 1 — So sánh RA vs EquiCEval trên output LLM

> Handoff cho session mới. Đọc hết file này trước khi chạy.
> Repo: `equiceval-llm-eval` (`git@github.com:Xtan122/equiceval-llm-eval.git`).
> Tài liệu liên quan: `README.md`, `VENDOR.lock`, các repo method `EquiCEval`, `RA`.

## 1. Mục tiêu

Chấm **cùng một tập output LLM** bằng hai phương pháp và so sánh bằng FPR / Recall:

- **RA** — baseline Refai & Ahmed (component metrics: Cons-P/R, DV-P/R, gap, obj-RMSE, cons-RMSE).
- **EquiCEval** — engine M0–M5 (verdict theo contract).

Output LLM **không có nhãn vàng**, nên phải tạo nhãn độc lập trước rồi mới tính được FPR/Recall.
FPR/Recall là chỉ số **theo lớp**, không phải "khớp nhãn bao nhiêu".

## 2. Dữ liệu

| Thành phần | Nội dung |
|---|---|
| Reference | 4 bài ComplexOR: `Knapsack`, `AircraftAssignment`, `Diet`, `AircraftLanding` |
| Candidate | Output LLM = 4 mô tả × 6 prompt (`P1`..`P6`) × N model |
| Nhãn | Máy dùng oracle độc lập, mỗi cặp `(reference, candidate)` → `True` / `False` / `None` |
| Contract | `feasible_set_and_objective_affine` (chốt; đổi contract = đổi nhãn) |

`None` = oracle không chứng minh được → **giữ nguyên**, loại khỏi mẫu số FPR/Recall nhưng phải đếm.

## 3. Quyết định đã chốt

1. Sân chính là **output LLM**.
2. Nhãn tạo bằng **máy** (oracle chính xác), không chấm tay.
3. RA không có verdict → phải **ngưỡng hóa** chỉ số thành quyết định nhị phân.
4. Vendor thủ công + `VENDOR.lock` (không sửa code trong `vendor/`; sửa ở repo method rồi copy lại).
5. Mỗi repo có dataset + docs riêng; OptMATH60 đã bỏ nhưng giữ LLM client (kể cả vLLM).

## 4. Cài đặt & kiểm tra

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests -q     # kỳ vọng 7 passed
```

Bốn repo method/experiment (đã push):
`EquiCEval`, `RA`, `equiceval-llm-eval`, `equiceval-meta-eval` (account `Xtan122`).

## 5. Quy trình thực thi

### Bước 1 — Sinh output LLM (mất phí API)

```bash
PYTHONPATH=. .venv/bin/python -m expeval.generate_llm_outputs \
  --model bedrock-gpt \
  --problem Knapsack AircraftAssignment Diet AircraftLanding \
  --prompt P1 P2 P3 P4 P5 P6 \
  --output data/llm_outputs_raw.json
```

- Model hợp lệ do `vendor/src/benchmark/llm_runner.py` định tuyến: `bedrock-gpt`,
  `bedrock-gpt-20b`, `bedrock-llama`, `bedrock-deepseek`, `bedrock:<id>`,
  `Gemini`, `GPT-5`, `llama3.1:8b` (Ollama), `vllm:<hf-id>`.
- `--model` **không** nhận `ALL`.
- Output raw ghi mọi call; chỉ cặp parse thành công mới vào `pairs` (`label: null`).

### Bước 2 — Gán nhãn độc lập (máy, không gọi API)

```bash
PYTHONPATH=. .venv/bin/python -m expeval.label_oracle \
  --pairs data/llm_outputs_raw.json \
  --contract feasible_set_and_objective_affine \
  --output data/llm_outputs_labeled.json
```

- Dùng `src.benchmark.independent_oracle.independent_label` (liệt kê nguyên + đổi tên + bao hàm LP).
- In ra tỉ lệ gán nhãn được. **Không** ép `None` thành đúng/sai.

### Bước 3 — So sánh

```bash
PYTHONPATH=. .venv/bin/python -m expeval.run_comparison \
  --pairs data/llm_outputs_labeled.json \
  --contract feasible_set_and_objective_affine \
  --mode certificates --seconds 5 --ra-rule primary \
  --output output/llm_eval_report.json
```

Chạy thêm 2 độ nhạy cho RA: `--ra-rule cons-precision` và `--ra-rule solver-only`.

## 6. Ngưỡng RA (quan trọng)

RA không có verdict. Repo dùng 3 quy tắc (báo cả 3):

| `--ra-rule` | Quyết định "faulty" khi |
|---|---|
| `primary` | `Cons-R < 1` **hoặc** `Cons-RMSE > τ` **hoặc** `gap > τ` (`τ = 1e-6`) |
| `cons-precision` | thêm `Cons-P < 1` (đo tác động ràng buộc thừa) |
| `solver-only` | chỉ `gap > τ` (RA ở dạng dễ tính nhất) |

`primary` cố ý không tính `Cons-P` vì RA's own kết luận: ràng buộc dư vô hại không làm hỏng nghiệm.

## 7. Chỉ số & cách đọc

`output/llm_eval_report.json` gồm `methods.{equiceval,ra}.metrics`:

- `false_alarm_rate` (FPR) = cặp nhãn `True` bị báo `faulty` / số cặp `True`.
- `error_recall` = cặp nhãn `False` bị báo `faulty` / số cặp `False`.
- `equivalent_confirmation_rate`.
- `unresolved_rate` / `unsupported_rate` vẫn ở mẫu số.
- `n_unverified` = số nhãn `None` (không vào FPR/Recall).

## 8. Tái lập

- Ghi SHA vendor (`VENDOR.lock`) + `config` trong report.
- Chốt cấu hình trước khi chạy: model, prompt, temperature, max tokens, seed (nếu có).
- Muốn so với bảng công bố của RA: dùng đúng 3 model của họ (DeepSeek Math 7B,
  LLaMA 3.1 8B, GPT-5) và 6 prompt; khác model → ghi là **cấu hình thích nghi**.

## 9. Vấn đề đã biết — cần quyết trước khi chạy RA

1. **Lệch tên biến — ĐÃ xử lý.** `RAAdapter` tự chuẩn hóa tên biến candidate về tên
   ground truth (`x1` → `x_1`, `x11` → `x_1_1`, `z_1_2` → `z12`, …) trước khi chấm,
   nên `Cons-RMSE` lấy mẫu đúng miền GT. Test:
   `tests/test_adapters.py::test_ra_normalizes_compact_variable_names`.
2. **AircraftLanding** khó oracle nhất (MILP big-M) → nhiều nhãn `None`; nếu cần độ phủ
   cao phải bổ sung quy trình gán nhãn thủ công (2 người + Cohen κ).
3. `run_comparison` hiện gọi API tuần tự; số call = `N_model × 6 × 4`.

## 10. Tiêu chí nghiệm thu

- [ ] Sinh đủ output và `parse_ok` được ghi lại.
- [ ] Có `data/llm_outputs_labeled.json` + báo **độ phủ nhãn**.
- [ ] Cả 2 phương pháp chấm trên **cùng tập đóng băng**.
- [ ] Report có FPR/Recall (+ unresolved/unsupported) + chi phí; nhãn `None` báo riêng.
- [ ] `config` ghi SHA vendor + cấu hình model/prompt.

## 11. Liên hệ các task khác

- (2) 1200 cặp có nhãn → repo `EquiCEval` (`experiments.evaluate_equiceval`).
- (3)(4) EquiCEval vs EquivaMap → repo `equiceval-meta-eval`
  (`expeval.build_pairs` → `expeval.relabel` → `expeval.run_comparison`).
