"""Closed-form GPU gradient reference through real PRe Trainer and ZeRO2."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset
import deepspeed

from llava.train.train import TrainingArguments
from llava.train.llava_trainer import LLaVATrainer

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
args_cli = parser.parse_args()
root = Path('/home/jovyan/dowser-lora-datavol-1/phong/LLaVA')


class AnalyticPRe(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_weight = nn.Parameter(torch.tensor(2.0))
        self.pre_predictor_weight = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(cka_loss=False, pre_loss=True, use_pcgrad=False)

    def forward(self, input_ids=None, labels=None):
        logits = 2 * self.lm_weight
        ce = logits.square()
        pre = (self.pre_predictor_weight * self.lm_weight).square()
        return {'loss': ce + 0.5 * pre, 'logits': logits}


class RepeatedData(Dataset):
    def __len__(self): return 32
    def __getitem__(self, index):
        return {'input_ids': torch.tensor([1]), 'labels': torch.tensor([1])}


def collate(items):
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


gradient_records = []


class RecordingTrainer(LLaVATrainer):
    def create_optimizer(self):
        result = super().create_optimizer()
        def record(optimizer, *_):
            gradient_records.append([
                parameter.grad.detach().float().cpu().clone()
                for group in optimizer.param_groups for parameter in group['params']
                if parameter.grad is not None
            ])
        self.optimizer.register_step_pre_hook(record)
        return result


# dCE/dw=16; d(0.5*PRe)/dw=2; d(0.5*PRe)/dp=4.
# Correct global-average gradient is [18,4] for every GAS. With nested DS
# output scaling at GAS32 the broken gradient would instead be [2.5,4].
manual = AnalyticPRe().to(dtype=torch.bfloat16)
for _ in range(32):
    (manual()['loss'] / 32).backward()
expected = torch.cat([parameter.grad.float().reshape(-1) for parameter in manual.parameters()])
print('MANUAL_BF16_GAS32_REFERENCE '+str(expected.tolist()), flush=True)

ds_config = json.loads((root / 'scripts/zero2_b200.json').read_text())
ds_config['gradient_clipping'] = 0.0
training_args = TrainingArguments(
    output_dir='/tmp/pre-zero2-gas32-numerics', per_device_train_batch_size=1,
    gradient_accumulation_steps=32, max_steps=1, num_train_epochs=1,
    learning_rate=2e-5, max_grad_norm=0.0, weight_decay=0.0,
    bf16=True, fp16=False, tf32=True, gradient_checkpointing=False,
    optim='adamw_torch_fused', deepspeed=ds_config,
    report_to=[], logging_steps=1, save_strategy='no',
    dataloader_num_workers=0, remove_unused_columns=False,
    disable_tqdm=True, local_rank=args_cli.local_rank,
)
model = AnalyticPRe().to(dtype=torch.bfloat16)
trainer = RecordingTrainer(model=model, args=training_args,
                          train_dataset=RepeatedData(), data_collator=collate)
trainer.train()
nonzero = [record for record in gradient_records if any(bool(tensor.count_nonzero()) for tensor in record)]
assert nonzero, 'No real optimizer gradient was captured.'
actual = torch.cat([tensor.reshape(-1) for tensor in nonzero[-1]])[:2]
print('ACTUAL_VS_MANUAL '+str(actual.tolist())+' '+str(expected.tolist()), flush=True)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
report = {
    'zero_stage': trainer.deepspeed.zero_optimization_stage(), 'gas': 32,
    'optimizer_steps': trainer.state.global_step,
    'ideal_fp32_lm_gradient': 18.0, 'expected_bf16_accumulated_lm_gradient': float(expected[0]), 'actual_lm_gradient': float(actual[0]),
    'expected_predictor_gradient': float(expected[1]), 'actual_predictor_gradient': float(actual[1]),
    'max_absolute_gradient_error': float((actual - expected).abs().max()),
    'broken_nested_hook_lm_gradient_would_be': 2.5,
    'precision': 'bf16 model/accumulation, fp32 optimizer gradient; reference accumulates32microgradients inbf16',
    'actual_optimizer': type(trainer.deepspeed.optimizer.optimizer).__name__,
    'fused': trainer.deepspeed.optimizer.optimizer.param_groups[0]['fused'],
    'scale_wrt_gas': trainer.deepspeed._scale_wrt_gas,
}
assert report['zero_stage'] == 2 and report['scale_wrt_gas'] is False
Path('/tmp/pre_zero2_gas32_gradient.json').write_text(json.dumps(report, indent=2)+'\n')
print('PRE_ZERO2_GAS32_GRADIENT_PASS '+json.dumps(report), flush=True)
torch.distributed.destroy_process_group()
