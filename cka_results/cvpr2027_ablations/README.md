# VSP controlled stage-2 ablations

Chạy từ repository root. [audit.md](audit.md) ghi nguồn checkpoint, công thức loss và các khác biệt của run cũ. Không tái sử dụng checkpoint stage-2 lịch sử vào factorial: chưa xác minh được cùng initialization/reference/protocol.

## Kiểm tra môi trường và code

```bash
python scripts/cvpr2027/verify.py --verbose
python -c 'import torch, transformers, accelerate, flash_attn; print(torch.__version__, transformers.__version__, accelerate.__version__)'
```

Runner yêu cầu môi trường training của repository: `transformers==4.51.3`, CUDA và FlashAttention tương thích PyTorch/CUDA, cùng DeepSpeed. Môi trường hiện tại có Transformers 4.43.1, PyTorch 2.9.1+cu130 và FlashAttention đòi `libcudart.so.12`; chưa phù hợp để chạy GPU. Không tự thay packages hay chiếm GPU đang sử dụng. `--python /path/to/env/bin/python` chọn đúng interpreter cho cả launcher và training.

`verify.py` chạy smoke synthetic trên CPU, tắt FlashAttention chỉ trong process kiểm tra; không sửa môi trường. Test kiểm tra bốn loss placements, gradient đến projector/decoder, frozen vision, final trước norm, CKA row-normalized, save/resume model và AdamW. Đây không thay thế smoke bằng model/dataset thật trên môi trường CUDA đích.

## Initialization chung

Vicuna7 và Qwen3B chưa có stage-1 structural được xác minh trong workspace. Cung cấp `--stage1-adapter /path/to/mm_projector.bin` có `config.json` đi kèm. Runner đọc projector type từ checkpoint, kiểm tra model/type và `cka_loss=True`, lưu SHA256; metadata này không tự chứng minh toàn bộ provenance, xem audit. CE-only adapter chỉ chạy khi chủ động dùng `--allow-unverified-stage1`; đó là thí nghiệm initialization khác và phải báo cáo riêng.

Có thể tạo một stage-1 chung mới cho Vicuna (plain CE + projector CKA 0.1 theo hệ số hiện tại), rồi dùng đúng adapter đó cho cả bốn biến thể:

```bash
python scripts/cvpr2027/run.py stage1 --backbone vicuna-7b \
  --gpu-include localhost:0 --execute
export STAGE1_ADAPTER="$PWD/checkpoints/cvpr2027_ablations/controlled-v1/vicuna-7b/structural_stage1/seed-42/stage1/mm_projector.bin"
```

Stage-1 helper dùng recipe pretrain hiện tại: 558K, một epoch, LR 1e-3, batch 16/accumulation 16; checkpoint mới dưới `checkpoints/cvpr2027_ablations/`. Đó là initialization chung mới, không phải phục dựng run historical có PCGrad. Sửa `configs/ablation.json` để đặt chính xác recipe stage-1 của paper nếu khác. `--pretrain-data-path`/`--pretrain-image-folder` đổi vị trí input. Stage-1 seed cố định dùng chung; seeds 42/43/44 trong factorial là các lần training stage-2 độc lập, không phải ba lần pretrain độc lập.

Qwen7 mặc định dùng adapter structural đã audit: `checkpoints/qwen-7b-cka-grad/llava-pretrain/mm_projector.bin` (`mlp2x_gelu`). Khi chuyển máy phải copy đúng adapter/config hoặc truyền đường dẫn mới.

## Smoke và P0

```bash
# Kiểm tra load, hai optimizer steps trên 8 mẫu có ảnh, finite losses, save checkpoint.
python scripts/cvpr2027/run.py smoke --backbone vicuna-7b --variant final \
  --stage1-adapter "$STAGE1_ADAPTER" --gpu-include localhost:0 --execute

# Kiểm tra resume từ checkpoint-1 và thực hiện lại step 2.
python scripts/cvpr2027/run.py smoke --backbone vicuna-7b --variant final \
  --stage1-adapter "$STAGE1_ADAPTER" --gpu-include localhost:0 \
  --resume checkpoints/cvpr2027_ablations/controlled-v1/vicuna-7b/final/seed-42/smoke/checkpoint-1 --execute

# Ưu tiên training Final-only.
python scripts/cvpr2027/run.py train --backbone vicuna-7b --variant final --seed 42 \
  --stage1-adapter "$STAGE1_ADAPTER" --gpu-include localhost:0,1 --execute

# Factorial: Final, Baseline, Proj, Full; mỗi biến thể ba seeds.
python scripts/cvpr2027/run.py suite --backbone vicuna-7b --seeds 42 43 44 \
  --stage1-adapter "$STAGE1_ADAPTER" --gpu-include localhost:0,1 --execute
```

Không có `--execute`: chỉ ghi config/command/manifest pending. `plan` ghi toàn bộ factorial. Suite chạy tuần tự; từng lệnh có status/log riêng. Chạy Final-only rồi suite vào cùng campaign sẽ từ chối ghi đè Final đã có; các run còn lại vẫn chạy. Dùng `--variants baseline proj full` để chỉ chạy phần còn thiếu. Tiếp tục một run bị ngắt bằng cùng lệnh `train ... --resume latest --execute`; source/recipe/data/adapter phải trùng với identity đã lưu. Đổi `--campaign` khi cần một lần chạy mới; không dùng checkpoint smoke cho evaluation.

Recipe stage-2 chung: mix665K, một epoch, CE + weighted CKA, LR 2e-5, AdamW, weight decay 0, warmup .03, cosine, BF16/TF32, max length 2048, CLIP -2 patches, pad ảnh, toàn bộ projector và language backbone trainable. PCGrad/norm cap/protected refinement/subset selection/anchor overrides/head phụ đều tắt. Checkpoint được lưu mỗi 500 steps để resume (thay đổi tần suất lưu so với launcher cũ, không đổi loss/update). Hai GPU × batch4 × accumulation16 = global batch128. Nếu đổi số GPU, giữ global batch/hyperparameters nhất quán trong toàn bộ nhóm và ghi rõ protocol mới. `--gradient-checkpointing` đặt `use_reentrant=False` cho cả Baseline lẫn CKA. Không hardcode số decoder layers.

## Matched-strength control

```bash
python scripts/cvpr2027/run.py calibrate --backbone vicuna-7b --seed 42 \
  --stage1-adapter "$STAGE1_ADAPTER" --gpu-include localhost:0 \
  --calibration-batches 8 --calibration-batch-size 1 --execute
python scripts/cvpr2027/run.py train --backbone vicuna-7b --variant proj_strong --seed 42 \
  --stage1-adapter "$STAGE1_ADAPTER" --gpu-include localhost:0,1 \
  --strength-match cka_results/cvpr2027_ablations/training_logs/controlled-v1-vicuna-7b-full-seed42-calibration/strength_match.json --execute
```

Calibration chỉ forward/autograd trên các mẫu có ảnh được chọn uniform không lặp từ training data, không optimizer update và không lưu trained checkpoint. Ghi loss, norm của từng term đến projector, norm vector tổng/cosine, final-block norm và GPU memory trong `gradient_norms.csv`. Chọn `median(||0.1*g_proj + 0.1*g_final|| / ||g_proj||)`, không mặc định nhân đôi hệ số. Nếu vector tổng yếu hơn do cancellation, kết quả ghi rõ; không gọi hệ số nhỏ hơn là stronger control. Calibration không khớp decoder gradients hay bảo đảm khớp mọi bước training; image-only probe chưa bao gồm dummy image regularization của mẫu text-only. Để cố định một hệ số đã chọn cho nhiều seeds, đọc số từ calibration và dùng cùng `--strong-projector-weight VALUE`, ghi protocol lựa chọn trước evaluation. Có thể bật `log_gradient_norms`/`gradient_log_steps` trong recipe để khảo sát về sau; overhead diagnostics phải dùng đồng nhất khi so cost.

## P1: geometry và hệ số

```bash
python scripts/cvpr2027/run.py suite --backbone vicuna-7b --seeds 42 \
  --stage1-adapter "$STAGE1_ADAPTER" --gpu-include localhost:0,1 \
  --variants full_row_normalized full_proj_low full_proj_high full_final_low full_final_high --execute
```

`configs/variants.json` liệt kê hệ số: original .1/.1, các sweep .05 và .2 cho đúng một term. `linear` giữ nguyên định nghĩa hiện tại; `row_normalized` L2 mỗi token trong FP32 trước centering tại cả hai loss locations. Không thêm normalized-Gram MSE, masking hay inference module. Giữ patch padding/dummy image behavior như implementation hiện tại và báo cáo hạn chế trong audit.

Nếu lựa chọn hệ số, dùng validation đã định trước, không dùng benchmark test scores. Ví dụ tạo holdout theo nhóm ảnh, seed cố định:

```bash
python scripts/cvpr2027/prepare_validation.py --data-path playground/data/llava_v1_5_mix665k.json \
  --output-dir cka_results/cvpr2027_ablations/validation --fraction 0.01 --seed 2027
```

Dùng `--data-path .../validation/train.json --validation-data-path .../validation/validation.json` cho **tất cả** biến thể trong campaign mới; validation CE được ghi vào trainer_state/telemetry mỗi 500 steps, không thêm vào update. Cố định split và tiêu chí (ví dụ validation CE) trước khi xem test metrics. Holdout thay training set nên không so trực tiếp với factorial dùng toàn bộ mix665K. Runner từ chối exact training-conversation duplicates trong validation.

## Evaluation và báo cáo

```bash
# Dùng task IDs và adapter args của evaluator đã cài, theo protocol benchmark chính thức.
python scripts/cvpr2027/evaluate.py --run-id controlled-v1-vicuna-7b-final-seed42 \
  --tasks "$TASK_IDS" --model "$LMMS_BACKEND" --model-args "$LMMS_MODEL_ARGS" --execute
python scripts/cvpr2027/evaluate.py --run-id controlled-v1-vicuna-7b-final-seed42 \
  --import-results /path/to/full_results.json
python scripts/cvpr2027/summarize.py --root cka_results/cvpr2027_ablations --plots
```

Model args hỗ trợ `{checkpoint}` và `{model_name_or_path}` để dùng đúng manifest checkpoint. Có thể `--import-csv official_scores.csv` với `benchmark,metric,score` từ evaluator chính thức. Không đoán task mappings: dùng `python -m lmms_eval --help` và task list của môi trường evaluator. Script chỉ nhận completed `mode=train`, giữ nguyên results gốc và không ghi đè score đã nhập. Không import smoke/limited evaluation. Thêm `metric_directions.json` với `{"benchmark":{"metric":"higher"}}` hoặc `lower` để diễn giải cải thiện/suy giảm chính xác.

Artifacts: `training_manifest.csv`, `training_logs/`, `benchmark_results.csv`, `ablation_summary.csv`, `figures/`, `ablation_table.tex`, `summary.md`. Summary tính mean/sample SD theo independent training seeds (SD trống nếu một seed), paired deltas khi provenance trùng, và interaction `Full-Proj-Final+Baseline` chỉ từ quartet đầy đủ cùng seed. Không average các benchmark khác đơn vị. Báo cáo Q1–Q6 dựa trên measurements; trước training/evaluation tất cả kết luận hiệu quả đều chưa xác định.
