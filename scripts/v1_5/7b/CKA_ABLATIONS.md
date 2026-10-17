# Ablation vị trí CKA: projector và final state

## Runner mới: ba setting, hai lượt pretrain + finetune độc lập

Dùng [train_cka_two_stage.sh](train_cka_two_stage.sh) từ repository root.
Mặc định chạy **2 repeat × 3 setting = 6 pretrain + 6 finetune**, tuần tự theo
`repeat → setting → pretrain → finetune`. Mỗi setting ở mỗi repeat có adapter
Stage 1 riêng; finetune chỉ nạp adapter của chính run đó, không chia sẻ adapter.

Bảng loss này áp dụng cho **cả Stage 1 và Stage 2**:

| `SETTINGS` | Projector CKA | Final-state CKA | Decoder layers |
| --- | ---: | ---: | --- |
| `baseline` | 0 | 0 | tắt |
| `cka_proj` | 0.1 | 0 | tắt |
| `cka_proj_final` | 0.1 | 0.1 | `final` |

Loss là CE cộng trực tiếp các term được bật, dùng centered linear CKA, tau 0,
không gradient controller, token selection hay auxiliary head. Stage 1 chỉ
train projector; final CKA vẫn backpropagate qua decoder frozen tới projector.
Stage 2 train projector và LLM. `final` là output block cuối trước final norm.

Mặc định dùng `lmsys/vicuna-7b-v1.5`, `mlp2x_gelu`, GPU
`localhost:1,2`. Stage 1 microbatch **32**, accumulation **4** → global batch
**256**; Stage 2 microbatch **8**, accumulation **8** → global batch **128**
trên hai GPU. Cả hai stage truyền `--seed 42` cho HF Trainer; không seed riêng
trước khởi tạo projector. `repeat-1/2` là hai lượt chạy với cùng Trainer seed
42; projector khởi tạo độc lập, không được bảo đảm giống nhau hay tái lập hoàn
toàn chỉ nhờ seed này.

Xem đủ 12 lệnh mà không chạy train:

```bash
DRY_RUN=1 bash scripts/v1_5/7b/train_cka_two_stage.sh
```

Chạy trong environment `llava1`:

```bash
conda activate llava1
GPU_INCLUDE=localhost:1,2 bash scripts/v1_5/7b/train_cka_two_stage.sh
```

`STAGE=all` là mặc định; `STAGE=pretrain` hoặc `STAGE=finetune` chỉ chạy stage
tương ứng. `REPEATS` (mặc định 2) đổi số lượt chạy; `SETTINGS="baseline cka_proj"` chọn
subset cách nhau bằng dấu cách. `CKA_WEIGHT=0.1` là weight chung cho mọi term
được bật ở cả hai stage; override phải hữu hạn và dương. Override resource bằng
`PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE`,
`PRETRAIN_GRADIENT_ACCUMULATION_STEPS`, `PER_DEVICE_TRAIN_BATCH_SIZE` và
`GRADIENT_ACCUMULATION_STEPS`; giữ cùng recipe giữa các setting so sánh.

Output mặc định (có thể đổi `EXPERIMENT_ROOT`):

```text
checkpoints/cka-two-stage/vicuna-mlp2x_gelu-lambda0.1-seed42/
  repeat-1/{baseline,cka_proj,cka_proj_final}/
    llava-pretrain/mm_projector.bin
    llava-finetune/
  repeat-2/{baseline,cka_proj,cka_proj_final}/
    llava-pretrain/mm_projector.bin
    llava-finetune/
```

Stage 1 export adapter và config; Stage 2 export model cuối. Không ghi đè run
cũ. `STAGE=finetune` yêu cầu adapter Stage 1 tương ứng đã tồn tại.

## Runner cũ: ablation Stage 2 với shared/paired initialization

Phần dưới chỉ áp dụng cho `train_cka_ablations.sh`; runner cũ vẫn giữ weight
Stage 2 mặc định **0.01**, khác runner mới có weight chung **0.1** ở cả hai stage.

Chạy từ repository root bằng `bash scripts/v1_5/7b/train_cka_ablations.sh`.
Launcher gọi trực tiếp [pretrain_qwen2_5.sh](pretrain_qwen2_5.sh) và
[finetune_qwen2_5.sh](finetune_qwen2_5.sh), mặc định
`lmsys/vicuna-7b-v1.5`, `RUN_NAME=vicuna` và projector `mlp2x_gelu`.
Hai script con giữ recipe train; launcher chỉ điều phối stage, variant, seed,
adapter và output. Tên file vẫn giữ `qwen2_5`, nhưng backbone được chọn qua
`MODEL_NAME_OR_PATH`; launcher cũ truyền Vicuna tường minh cho cả hai stage.
Không gọi runner `scripts/cvpr2027/`.

## Bốn biến thể Stage 2

| `VARIANTS` | Projector CKA | Final-state CKA | Decoder layers |
| --- | ---: | ---: | --- |
| `baseline` | 0 | 0 | tắt |
| `proj` | 0.01 | 0 | tắt |
| `final` | 0 | 0.01 | `final` |
| `full` | 0.01 | 0.01 | `final` |

Đây là hệ số mặc định. Khi override weight, `proj/full` cùng dùng
`CKA_PROJECTOR_WEIGHT`, `final/full` cùng dùng `CKA_FINAL_HIDDEN_WEIGHT`.
Term bị tắt theo variant luôn bằng 0.

Loss là CE cộng trực tiếp các CKA term được bật. Dùng centered linear CKA gốc,
tau bằng 0 và bật CKA từ đầu. `final` là output decoder block cuối trước final
normalization, không hardcode độ sâu. Final loss giữ gradient qua decoder và
projector; chỉ reference vision features bị detach.

Không bật PCGrad, VSP gradient controller/norm cap, gradient diagnostics,
protected refinement, anchor override, Otsu/attention token selection, hidden
channel filtering, intermediate-layer CKA, row-normalized CKA, auxiliary head
hay LoRA. Chỉ ba biến weight được liệt kê bên dưới được đọc; các CKA/VSP
environment variables khác không bật lại các kỹ thuật phụ. Không nhận CLI
passthrough để tránh vô tình đổi objective. Launcher xóa các option cũ về
gradient controller, anchor, subset, auxiliary head và checkpoint/resume trước
khi gọi script con.

## Chỉnh CKA loss weight

| Environment variable | Áp dụng ở đâu | Mặc định |
| --- | --- | ---: |
| `PRETRAIN_CKA_PROJECTOR_WEIGHT` | Stage-1 structural adapter (`shared` hoặc nhánh `cka` của `paired`) | 0.1 |
| `CKA_PROJECTOR_WEIGHT` | Projector CKA ở Stage 2 (`proj`, `full`) | 0.01 |
| `CKA_FINAL_HIDDEN_WEIGHT` | Final-state CKA ở Stage 2 (`final`, `full`) | 0.01 |

Recipe paired đã chọn: train adapter baseline CE-only và adapter structural
với projector CKA weight 0.1, rồi finetune dùng projector weight 0.01 và
final-state weight 0.01. Có thể override bằng cách thay các giá trị trong lệnh:

```bash
INIT_MODE=paired STAGE=pretrain PRETRAIN_CKA_PROJECTOR_WEIGHT=0.1 \
PRETRAIN_GPU_INCLUDE=localhost:1,2 \
bash scripts/v1_5/7b/train_cka_ablations.sh

INIT_MODE=paired STAGE=finetune CKA_PROJECTOR_WEIGHT=0.01 \
CKA_FINAL_HIDDEN_WEIGHT=0.01 FINETUNE_GPU_INCLUDE=localhost:1,2 \
bash scripts/v1_5/7b/train_cka_ablations.sh
```

Trong ví dụ này, Stage 2 có các weight `(projector, final)` là baseline `(0,0)`,
proj `(0.01,0)`, final `(0,0.01)`, full `(0.01,0.01)`. Baseline luôn CE-only ở
stage tương ứng, bất kể các biến weight. Weight phải hữu hạn và không âm;
chấp nhận số dạng `5e-2`. Đặt weight bằng 0 tắt term đó, kể cả final hooks;
nếu cả hai term bằng 0 thì CKA được tắt hoàn toàn.

Ba biến độc lập: đổi weight finetune không đổi checkpoint pretrain có sẵn;
đổi `PRETRAIN_CKA_PROJECTOR_WEIGHT` ở `STAGE=finetune` cũng không sửa weight
đã dùng để train adapter. Đổi weight Stage 1 cần train adapter mới. Giữ cùng
adapter khi chỉ so sánh weight Stage 2. Để chạy thêm cấu hình weight sau khi
đã có kết quả, chọn `EXPERIMENT_ROOT` mới và trỏ tới adapter cũ bằng
`PRETRAIN_ADAPTER` (`shared`) hoặc `BASELINE_PRETRAIN_ADAPTER` cùng
`CKA_PRETRAIN_ADAPTER` (`paired`). Launcher không tự đổi tên output theo weight
và vẫn từ chối output đã tồn tại. Giá trị thực được truyền tường minh vào CLI
và lưu trong config/W&B của từng run. Dry-run cũng kiểm tra các weight bằng
Python 3 standard library, không load model hay CUDA.

## Initialization và recipe chung

`INIT_MODE=shared` (mặc định) train **một projector dùng chung** ở Stage 1 với
CE + projector CKA (weight mặc định 0.1). Cả bốn Stage-2 variants cùng nạp
adapter này và cùng base LLM, nên phép so sánh chỉ tách vị trí loss ở Stage 2.
`baseline` ở chế độ này là **CE-only ở Stage 2**, vẫn dùng structural
initialization của Stage 1.

`INIT_MODE=paired` train đúng **hai projector Stage 1**, tuần tự: `baseline`
dùng CE-only; `cka` dùng CE + projector CKA (weight mặc định 0.1). Hai nhánh
cùng `STAGE1_SEED` (mặc định 42) và cùng recipe; Trainer seed không bảo đảm
projector khởi tạo giống nhau vì không có seed trước khởi tạo. Stage 2 nạp adapter
`baseline` cho variant `baseline`, và adapter `cka` cho `proj`, `final`, `full`.
Vì vậy baseline so với các structural variants là so sánh **cả Stage 1 + Stage
2**; riêng `proj/final/full` vẫn dùng chung structural initialization.

Cả hai chế độ đều không dùng final CKA ở Stage 1 hay gradient controller.
Vision encoder và LLM frozen ở Stage 1; checkpoint lưu `mm_projector.bin` và
config, không phải một LLM mới. Stage 2 nạp lại cùng base LLM rồi nạp adapter.

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Data | LLaVA-Pretrain 558K | LLaVA mix665K |
| GPU mặc định trong launcher | `localhost:1,2` | `localhost:1,2` |
| Batch mỗi GPU | 32 | 8 |
| Gradient accumulation | 4 | 8 |
| Learning rate | 1e-3 | 2e-5 |
| Epochs | 1 | 1 |
| Save strategy | `no` | `no` |
| Logging interval | 1 optimizer step | 10 optimizer steps |
| Trainable | projector | projector + language model |

`save_strategy=no` tắt checkpoint định kỳ; mỗi job vẫn export kết quả cuối.
GPU `localhost:1,2` ở cả hai stage là cấu hình launcher đã chọn. Khi chạy script
Qwen con trực tiếp, default GPU riêng hiện tại là `localhost:2,3` và backbone
`Qwen/Qwen2.5-7B-Instruct`; launcher truyền các giá trị riêng của nó.

Giữ CLIP ViT-L/14-336 frozen, vision layer -2, không chọn subset patch, DeepSpeed
ZeRO-2, AdamW, cosine scheduler, warmup .03, weight decay 0, BF16/TF32, context
2048, gradient checkpointing tắt, 16 workers. Stage 2 pad ảnh và group theo
modality length như hai script Qwen. Padding ảnh vẫn tham gia CKA; final-state mask
loại text và sequence padding, chỉ dùng visual tokens còn lại sau context
truncation. Projector CKA dùng mọi patch trước truncation và giữ hành vi dummy
image của mẫu text-only như implementation hiện tại.

Stage 1 dùng template `plain`; Stage 2 mặc định `auto`, resolve thành `v1`
cho Vicuna. Launcher chỉ thay vị trí/weight CKA theo bảng và dùng `final` khi
bật decoder CKA. Dùng cùng backbone, projector và recipe cho mọi variant trong
một nhóm so sánh.

## Cách chạy

Xem năm lệnh mặc định (một pretrain + bốn finetune), không cần GPU hoặc adapter.
Thêm `INIT_MODE=paired` để xem sáu lệnh (hai pretrain + bốn finetune):

```bash
DRY_RUN=1 bash scripts/v1_5/7b/train_cka_ablations.sh
```

Chế độ shared: train Stage 1 một lần rồi chạy bốn variants tuần tự:

```bash
PRETRAIN_GPU_INCLUDE=localhost:1,2 FINETUNE_GPU_INCLUDE=localhost:1,2 \
bash scripts/v1_5/7b/train_cka_ablations.sh
```

Chế độ paired, chạy riêng hai stage:

```bash
INIT_MODE=paired STAGE=pretrain PRETRAIN_GPU_INCLUDE=localhost:1,2 \
bash scripts/v1_5/7b/train_cka_ablations.sh

INIT_MODE=paired STAGE=finetune FINETUNE_GPU_INCLUDE=localhost:1,2 \
bash scripts/v1_5/7b/train_cka_ablations.sh
```

Hoặc `INIT_MODE=paired STAGE=all` để tự chạy cả hai nhánh Stage 1 rồi bốn
variants Stage 2. Mỗi lần paired `pretrain/all` luôn train cả hai nhánh Stage 1,
kể cả khi chỉ chọn một số `VARIANTS` cho Stage 2. `SEEDS="42 43 44"` vẫn chạy
nhiều seed Stage 2 với hai adapter Stage 1 cố định.

Với shared, có thể chia riêng pretrain và finetune; ưu tiên bản final-only trước:

```bash
STAGE=pretrain PRETRAIN_GPU_INCLUDE=localhost:1,2 \
bash scripts/v1_5/7b/train_cka_ablations.sh

STAGE=finetune VARIANTS=final FINETUNE_GPU_INCLUDE=localhost:1,2 \
bash scripts/v1_5/7b/train_cka_ablations.sh

STAGE=finetune VARIANTS="baseline proj full" FINETUNE_GPU_INCLUDE=localhost:1,2 \
bash scripts/v1_5/7b/train_cka_ablations.sh
```

Ba seed độc lập ở Stage 2, cùng Stage-1 initialization cố định:

```bash
STAGE=finetune SEEDS="42 43 44" \
bash scripts/v1_5/7b/train_cka_ablations.sh
```

Chọn thư mục mới nếu đã chạy seed 42; hoặc chỉ chạy `SEEDS="43 44"` cho phần
còn thiếu. Nếu đã có adapter phù hợp, chỉ chạy finetune:

```bash
STAGE=finetune PRETRAIN_ADAPTER=/path/to/mm_projector.bin \
bash scripts/v1_5/7b/train_cka_ablations.sh
```

Với paired, adapter ngoài được chỉ định riêng qua `BASELINE_PRETRAIN_ADAPTER`
và `CKA_PRETRAIN_ADAPTER`. Stage `finetune` chỉ yêu cầu adapter mà các
`VARIANTS` đã chọn cần dùng. Adapter ngoài phải khớp backbone/projector và
provenance Stage 1 tương ứng; launcher chỉ kiểm tra file tồn tại và không rỗng,
không chứng minh provenance. Giữ projector `mlp2x_gelu` tương ứng ở cả hai stage.

Đổi backbone (giữ cùng cấu hình cho mọi variant trong một nhóm):

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct RUN_NAME=qwen2.5-7b \
MM_PROJECTOR_TYPE=mlp2x_gelu PRETRAIN_GPU_INCLUDE=localhost:1,2 \
FINETUNE_GPU_INCLUDE=localhost:1,2 \
bash scripts/v1_5/7b/train_cka_ablations.sh
```

Các backbone dùng cho `final/full` phải hỗ trợ hidden CKA. Có thể gọi trực tiếp
hai script Qwen nếu chỉ cần một job; dùng launcher này để chạy đủ variants và
giữ đúng mapping adapter giữa các stage.

## Output và các biến được phép đổi

Output mặc định của `shared`:

```text
checkpoints/cka-location-ablations/vicuna-mlp2x_gelu/
  stage1-seed42/llava-pretrain/mm_projector.bin
  seed-42/baseline/llava-finetune/
  seed-42/proj/llava-finetune/
  seed-42/final/llava-finetune/
  seed-42/full/llava-finetune/
```

Output của `paired` nằm trong `${EXPERIMENT_ROOT}/paired`:

```text
paired/
  stage1-seed42/baseline/llava-pretrain/mm_projector.bin
  stage1-seed42/cka/llava-pretrain/mm_projector.bin
  seed-42/baseline/llava-finetune/
  seed-42/proj/llava-finetune/
  seed-42/final/llava-finetune/
  seed-42/full/llava-finetune/
```

`STAGE`: `all` (mặc định), `pretrain`, `finetune`. `VARIANTS` và `SEEDS` là
danh sách cách nhau bởi dấu cách; `SEED` là alias cho danh sách một seed.
`STAGE1_SEED` độc lập với seed Stage 2. Có thể đổi `EXPERIMENT_ROOT`,
`RUN_NAME`, `WANDB_RUN_GROUP`,
`CONVERSATION_VERSION`, `MODEL_NAME_OR_PATH`, `MM_PROJECTOR_TYPE`.

`shared` giữ `PRETRAIN_OUTPUT_DIR` và `PRETRAIN_ADAPTER` như cũ. `paired` dùng
`BASELINE_PRETRAIN_OUTPUT_DIR`, `CKA_PRETRAIN_OUTPUT_DIR`,
`BASELINE_PRETRAIN_ADAPTER`, `CKA_PRETRAIN_ADAPTER` và từ chối hai biến
`PRETRAIN_OUTPUT_DIR`/`PRETRAIN_ADAPTER` để tránh mơ hồ. Với `STAGE=all` ở cả
hai chế độ, mọi adapter chỉ định tường minh phải khớp `mm_projector.bin` trong
output Stage 1 tương ứng mà launcher sẽ tạo.

`GPU_INCLUDE` đổi GPU cho cả hai stage; `PRETRAIN_GPU_INCLUDE` và
`FINETUNE_GPU_INCLUDE` override riêng từng stage. Nếu đổi số GPU, giữ global
batch nhất quán giữa các run so sánh: batch/GPU × accumulation × số GPU.
Override resource bằng `PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE`,
`PRETRAIN_GRADIENT_ACCUMULATION_STEPS`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
`GRADIENT_ACCUMULATION_STEPS`, `DATALOADER_NUM_WORKERS`. Override data bằng
`PRETRAIN_DATA_PATH`, `PRETRAIN_IMAGE_FOLDER`, `FINETUNE_DATA_PATH`,
`FINETUNE_IMAGE_FOLDER`. `REPORT_TO=none` tắt W&B.

Chạy thật yêu cầu active conda env có DeepSpeed; script Qwen finetune vẫn
kiểm tra `transformers==4.51.3` và `accelerate>=1.6.0`. Dùng training environment
trong `pyproject.toml` và FlashAttention/CUDA tương thích. `DRY_RUN=1` hoạt động
qua các script con mà không cần conda, model, adapter hoặc CUDA. Dừng khi một
job lỗi hoặc thiếu adapter sau pretrain. Từ chối
output đã tồn tại, kể cả symlink, để không ghi đè hay tự resume run cũ. Đây là
launcher train mới, không có chế độ resume/evaluation/calibration/sweep thêm.
