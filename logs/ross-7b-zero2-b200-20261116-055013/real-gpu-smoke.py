"""Real FLUX VAE + official 1024x3 DiT through tiny LLaVA/ZeRO-2."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from unittest import mock

import deepspeed
import torch
from torch import nn
from torch.utils.data import Dataset, SequentialSampler
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from llava.model.ross import compute_ross_reconstruction_loss
from llava.train.train import TrainingArguments
from llava.train.llava_trainer import LLaVATrainer
from llava.train import llava_trainer as training

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
args_cli = parser.parse_args()
root = Path('/home/jovyan/dowser-lora-datavol-1/phong/LLaVA')
vae_path = root / 'checkpoints/auxiliary/ross-flux-vae'
torch.set_num_threads(8)


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('features', torch.randn(4, 6, generator=torch.Generator().manual_seed(937)))

    def forward(self, images):
        return self.features.unsqueeze(0).expand(images.shape[0], -1, -1)


def projector():
    return nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 16))


def config():
    value = LlavaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=4, max_position_embeddings=64)
    value.ross_enable = True
    value.ross_head_initialized = False
    value.ross_loss_weight = 1.0
    value.mm_pixel_decoder = str(vae_path)
    value.mm_inv_projector_type = 'denoiser_vit3x'
    value.mm_inv_hidden_size = 64
    value.image_embed_len = 4
    value.decode_image_size = 32
    value.pre_loss = value.cka_loss = False
    value.use_pcgrad = value.use_cache = False
    value.mm_patch_merge_type = 'flat'
    value.image_aspect_ratio = 'pad'
    value.tokenizer_padding_side = 'right'
    value.tokenizer_model_max_length = 64
    value.tune_mm_mlp_adapter = value.mm_use_im_start_end = False
    return value


def fingerprint(module):
    result = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class TinyData(Dataset):
    def __init__(self, gas): self.gas = gas
    def __len__(self): return 4 * self.gas
    def __getitem__(self, index):
        # Three visual optimizer updates let the deliberately zero-initialized
        # adaLN/output gates learn; the fourth whole update is text-only.
        visual = index // self.gas < 3
        ids = torch.tensor([1, IMAGE_TOKEN_INDEX if visual else 5, 2, 3])
        return dict(input_ids=ids, labels=ids.clone(), attention_mask=torch.ones_like(ids),
                    images=torch.zeros(3, 32, 32))


def collate(items):
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


class TinyTrainer(LLaVATrainer):
    def _get_train_sampler(self): return SequentialSampler(self.train_dataset)


reports = []
for gas in (2, 32):
    torch.manual_seed(941)
    model = LlavaLlamaForCausalLM(config())
    core = model.get_model()
    core.vision_tower = Vision()
    core.mm_projector = projector()
    assert not hasattr(core, 'mm_inv_projector')
    core.initialize_ross_modules(load_vae=True)
    head, vae = core.mm_inv_projector, core.pixel_decoder
    assert len(head.net.blocks) == 3
    assert head.net.x_embedder.proj.out_channels == 1024
    assert not head.net.pos_embed.requires_grad
    assert head.net.final_layer.linear.weight.count_nonzero() == 0
    original_vae_hash = fingerprint(vae)
    model.bfloat16()
    assert all(parameter.dtype == torch.float32 and not parameter.requires_grad for parameter in vae.parameters())
    assert fingerprint(vae) == original_vae_hash
    assert all(parameter.dtype == torch.bfloat16 for parameter in head.parameters())
    directory = f'/tmp/ross-real-gpu-gas{gas}'
    train_args = TrainingArguments(
        output_dir=directory, per_device_train_batch_size=1,
        gradient_accumulation_steps=gas, max_steps=4, num_train_epochs=1,
        learning_rate=1e-3, lr_scheduler_type='constant', weight_decay=0.0,
        bf16=True, fp16=False, tf32=True, gradient_checkpointing=False,
        optim='adamw_torch_fused', deepspeed=str(root / 'scripts/zero2_b200.json'),
        report_to=[], logging_steps=1, save_strategy='steps' if gas == 2 else 'no',
        save_steps=4, save_total_limit=1, dataloader_num_workers=0,
        remove_unused_columns=False, disable_tqdm=True, local_rank=args_cli.local_rank,
    )
    compute_observed = {'denoiser_storage': set(), 'denoiser_inputs': set(),
                        'functional_weights': set(), 'functional_inputs': set(),
                        'vae_inputs': set(), 'vae_weights': set()}
    def observe_head(module, inputs, kwargs):
        compute_observed['denoiser_storage'].add(str(next(module.parameters()).dtype))
        compute_observed['denoiser_inputs'].add(str(kwargs['z'].dtype))
    def observe_functional(module, inputs):
        compute_observed['functional_weights'].add(str(module.weight.dtype))
        compute_observed['functional_inputs'].add(str(inputs[0].dtype))
    def observe_vae(module, inputs):
        compute_observed['vae_weights'].add(str(module.weight.dtype))
        compute_observed['vae_inputs'].add(str(inputs[0].dtype))
    handles = [head.register_forward_pre_hook(observe_head, with_kwargs=True),
               head.net.x_embedder.proj.register_forward_pre_hook(observe_functional),
               vae.pixel_decoder.encoder.conv_in.register_forward_pre_hook(observe_vae)]
    trainer = TinyTrainer(model=model, args=train_args, train_dataset=TinyData(gas), data_collator=collate)
    with mock.patch.object(training, 'backward_pre_deepspeed_loss',
                           wraps=training.backward_pre_deepspeed_loss) as backward:
        result = trainer.train()
    for handle in handles: handle.remove()
    assert backward.call_count == 4 * gas
    assert trainer.deepspeed._scale_wrt_gas is False
    assert trainer.deepspeed.zero_optimization_stage() == 2
    assert all(group['fused'] for group in trainer.deepspeed.optimizer.optimizer.param_groups)
    assert not head.net.pos_embed.requires_grad
    assert all(parameter.dtype == torch.float32 and not parameter.requires_grad for parameter in vae.parameters())
    assert fingerprint(vae) == original_vae_hash, 'Frozen VAE weights were quantized or changed.'
    assert float(model.last_ross_loss) == 0.0 and int(model.last_ross_num_images) == 0
    last_text_loss = float(model.last_ross_loss)
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert float(head.net.final_layer.linear.weight.float().norm()) > 0
    assert compute_observed['denoiser_storage'] == {'torch.bfloat16'}
    for key in ('denoiser_inputs', 'functional_weights', 'functional_inputs', 'vae_inputs', 'vae_weights'):
        assert compute_observed[key] == {'torch.float32'}, (key, compute_observed[key])

    # Verify conditioning after the learned gates have opened. Freeze the head
    # only for this direct post-training input-gradient probe, avoiding DS's
    # parameter accumulation hooks while retaining the real DiT/VAE math.
    trainability = [parameter.requires_grad for parameter in head.parameters()]
    head.requires_grad_(False)
    hidden = torch.randn(1, 4, 16, device=model.device, dtype=torch.bfloat16, requires_grad=True)
    images = torch.zeros(1, 3, 32, 32, device=model.device, dtype=torch.bfloat16)
    auxiliary = compute_ross_reconstruction_loss(head, vae, hidden, images, config=model.config)
    auxiliary.backward()
    condition_grad = float(hidden.grad.float().norm())
    assert torch.isfinite(hidden.grad).all() and condition_grad > 0, condition_grad
    for parameter, enabled in zip(head.parameters(), trainability): parameter.requires_grad_(enabled)
    assert not head.net.pos_embed.requires_grad

    saved_head_reload = False
    eval_logit_delta = None
    if gas == 2:
        trainer.save_model(directory + '/final')
        head_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        projector_state = {name: value.detach().cpu().clone() for name, value in core.mm_projector.state_dict().items()}
        assert list(Path(directory, 'checkpoint-4').glob('**/*optim_states.pt'))
        model.eval()
        ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]], device=model.device)
        with torch.no_grad(), mock.patch.object(vae, 'encode', side_effect=AssertionError('Eval must skip VAE.')):
            expected_logits = model(input_ids=ids, images=images).logits.detach().cpu()
        unset_hf_deepspeed_config()
        restored_config = LlavaConfig.from_pretrained(directory + '/final')
        restored_config.mm_pixel_decoder = '/tmp/intentionally-missing-ross-vae'
        with mock.patch('llava.model.ross.FrozenFluxAutoencoder', side_effect=AssertionError('Loading eval must not load VAE.')):
            restored = LlavaLlamaForCausalLM.from_pretrained(
                directory + '/final', config=restored_config,
                torch_dtype=torch.bfloat16, local_files_only=True,
            )
        assert not hasattr(restored.get_model(), 'pixel_decoder')
        for name, expected in head_state.items():
            torch.testing.assert_close(restored.get_model().mm_inv_projector.state_dict()[name], expected, rtol=0, atol=0)
        saved_head_reload = True
        restored.get_model().vision_tower = Vision()
        restored.get_model().mm_projector = projector()
        restored.get_model().mm_projector.load_state_dict(projector_state)
        restored.to(device=model.device, dtype=torch.bfloat16).eval()
        with torch.no_grad(): actual_logits = restored(input_ids=ids, images=images).logits.cpu()
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        eval_logit_delta = float((actual_logits.float() - expected_logits.float()).abs().max())
        del restored, head_state, projector_state
    report = {
        'gas': gas, 'optimizer_steps': trainer.state.global_step, 'zero_stage': 2,
        'real_vae_sha256': json.loads((vae_path / 'provenance.json').read_text())['weights_sha256'],
        'vae_state_unchanged': True, 'vae_dtype': 'float32', 'vae_frozen': True,
        'dit_depth': 3, 'dit_width': 1024, 'head_storage_dtype': 'bfloat16',
        'reconstruction_compute_dtype': 'float32',
        'pos_embed_requires_grad': head.net.pos_embed.requires_grad,
        'head_updated': True, 'condition_gradient_norm_after_updates': condition_grad,
        'last_text_loss': last_text_loss, 'normalization_helper_calls': backward.call_count,
        'engine_scale_wrt_gas': trainer.deepspeed._scale_wrt_gas,
        'head_exact_checkpoint_reload': saved_head_reload, 'eval_logit_delta': eval_logit_delta,
        'train_loss': result.training_loss,
        'observed_dtypes': {name: sorted(values) for name, values in compute_observed.items()},
    }
    reports.append(report)
    Path('/tmp/ross_real_gpu_smoke.json').write_text(json.dumps(reports, indent=2)+'\n')
    print('ROSS_REAL_GPU_SMOKE_PASS '+json.dumps(report), flush=True)
    del trainer, model, core, head, vae, auxiliary, hidden, images
    gc.collect()
    torch.cuda.empty_cache()
print('ALL_ROSS_REAL_GPU_SMOKES_PASSED', flush=True)
torch.distributed.destroy_process_group()
