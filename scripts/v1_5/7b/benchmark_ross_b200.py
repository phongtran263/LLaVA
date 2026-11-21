#!/usr/bin/env python
"""Synthetic, real-model ROSS throughput/memory probe; never saves trained weights.

Launch with DeepSpeed on an idle GPU. Each candidate uses the real CLIP/image
insertion path and a 2048-token expanded sequence. Adam states are allocated
before measurements. Microbatch and checkpointing are varied on one loaded
model; the global batch remains 128. Synthetic optimizer updates are discarded.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import deepspeed
from transformers import AutoConfig, set_seed

from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.model.backbones import as_llava_config
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.train import ModelArguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--model', default='lmsys/vicuna-7b-v1.5')
    parser.add_argument('--projector', default='checkpoints/full-ft/baseline-7b/llava-pretrain/mm_projector.bin')
    parser.add_argument('--cases', default='4:off')
    parser.add_argument('--vae', default='checkpoints/auxiliary/ross-flux-vae')
    parser.add_argument('--sequence-length', type=int, default=2048)
    parser.add_argument('--measure-microsteps', type=int, default=3)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(42)
    local_rank = max(args.local_rank, 0)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    deepspeed.init_distributed(dist_backend='nccl')
    if torch.distributed.get_world_size() != 1:
        raise ValueError('This probe is for one otherwise idle B200.')
    config = as_llava_config(AutoConfig.from_pretrained(args.model, local_files_only=True))
    config.pre_loss = False
    config.ross_enable = True
    config.ross_head_initialized = False
    config.ross_loss_weight = 1.0
    config.mm_pixel_decoder = str(Path(args.vae).resolve())
    config.mm_inv_projector_type = 'denoiser_vit3x'
    config.mm_inv_hidden_size = 64
    config.image_embed_len = 576
    config.decode_image_size = 384
    config.cka_loss = False
    config.use_pcgrad = False
    config.use_cache = False
    config.tokenizer_model_max_length = args.sequence_length
    config.tokenizer_padding_side = 'right'
    config.image_aspect_ratio = 'pad'
    config.mm_use_im_start_end = False
    config.mm_use_im_patch_token = False
    config.tune_mm_mlp_adapter = False
    print('ROSS_BENCH_LOADING_MODEL', flush=True)
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model_args = ModelArguments(
        vision_tower='openai/clip-vit-large-patch14-336',
        mm_projector_type='mlp2x_gelu', mm_vision_select_layer=-2,
        pretrain_mm_mlp_adapter=str(Path(args.projector).resolve()),
        mm_use_im_start_end=False, mm_use_im_patch_token=False,
    )
    model.get_model().initialize_vision_modules(model_args)
    model.get_vision_tower().to(device=device, dtype=torch.bfloat16)
    model.get_model().mm_projector.to(device=device, dtype=torch.bfloat16)
    vision_tower = model.get_vision_tower()
    model.config.image_mean = vision_tower.image_processor.image_mean
    model.config.image_std = vision_tower.image_processor.image_std
    # Fresh ROSS head creation must occur after HF's no_init_weights context.
    head = model.get_model().initialize_ross_modules(load_vae=True)
    head.to(device=device, dtype=torch.bfloat16)
    vae = model.get_model().pixel_decoder
    assert all(not parameter.requires_grad for parameter in vae.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in vae.parameters())
    assert not head.net.pos_embed.requires_grad
    assert not head.net.final_layer.linear.weight.count_nonzero()
    assert all(not block.adaLN_modulation[-1].weight.count_nonzero() for block in head.net.blocks)
    model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True,
    )
    ds_config = {
        'bf16': {'enabled': True}, 'fp16': {'enabled': False},
        'train_micro_batch_size_per_gpu': 4,
        'gradient_accumulation_steps': 32, 'train_batch_size': 128,
        'gradient_clipping': 1.0,
        'zero_optimization': {
            'stage': 2, 'overlap_comm': False, 'contiguous_gradients': True,
            'reduce_bucket_size': 16777216,
            'allgather_bucket_size': 16777216,
        },
        'steps_per_print': 1000000, 'wall_clock_breakdown': False,
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config,
    )
    optimizer = None
    assert all(parameter.dtype == torch.float32 for parameter in vae.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in head.parameters())
    assert not head.net.pos_embed.requires_grad
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    print('ROSS_BENCH_READY ' + json.dumps({
        'gpu': torch.cuda.get_device_name(device),
        'total_memory_gib': torch.cuda.get_device_properties(device).total_memory / 2**30,
        'zero_stage': engine.zero_optimization_stage(),
        'torch': torch.__version__, 'cpu_threads': torch.get_num_threads(),
        'sequence_length': args.sequence_length,
        'global_batch': 128,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'seed': 42,
        'optimizer': 'torch.optim.AdamW(fused=True)',
        'learning_rate': 2e-5,
        'adam_betas': [0.9, 0.999],
        'adam_epsilon': 1e-8,
        'weight_decay': 0.0,
        'model': args.model,
        'vision_tower': 'openai/clip-vit-large-patch14-336',
        'projector_checkpoint': str(Path(args.projector).resolve()),
        'vae_path': model.config.mm_pixel_decoder,
        'decode_image_size': model.config.decode_image_size,
        'ross_weight': model.config.ross_loss_weight,
        'ross_denoiser': model.config.mm_inv_projector_type,
        'ross_head_dtype': str(next(head.parameters()).dtype),
        'vae_dtype': str(next(vae.parameters()).dtype),
        'vae_frozen': all(not parameter.requires_grad for parameter in vae.parameters()),
        'head_pos_frozen': not head.net.pos_embed.requires_grad,
        'head_official_initial_output_zero': bool(not head.net.final_layer.linear.weight.count_nonzero()),
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'total_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'allocator': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
    }), flush=True)

    for case in args.cases.split(','):
        micro_text, checkpoint_text = case.split(':')
        micro = int(micro_text)
        checkpoint = checkpoint_text == 'on'
        if micro <= 0 or 128 % micro:
            raise ValueError('Microbatch must divide 128.')
        accum = 128 // micro
        if checkpoint:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            model.gradient_checkpointing_disable()
        engine.set_train_micro_batch_size(micro)
        engine.set_train_batch_size(128)
        input_length = args.sequence_length - 575
        if input_length < 3:
            raise ValueError('Expanded sequence must fit 576 visual patches.')
        batch = None
        try:
            ids = torch.randint(4, 32000, (micro, input_length), device=device)
            ids[:, 0] = 1
            ids[:, 1] = IMAGE_TOKEN_INDEX
            labels = ids.clone()
            labels[:, :2] = IGNORE_INDEX
            batch = dict(input_ids=ids, labels=labels,
                         attention_mask=torch.ones_like(ids),
                         images=torch.zeros(micro, 3, 336, 336, device=device, dtype=torch.bfloat16))
            print('ROSS_BENCH_CASE_START ' + json.dumps({'microbatch': micro, 'checkpointing': checkpoint}), flush=True)

            # One complete update warms kernels and touches every Adam moment.
            engine.set_gradient_accumulation_boundary(True)
            warm_output = engine(**batch)
            if not torch.isfinite(warm_output.loss):
                raise RuntimeError('ROSS warmup loss is nonfinite.')
            engine.backward(warm_output.loss / accum, scale_wrt_gas=False)
            engine.step()
            del warm_output
            torch.cuda.synchronize()
            warmup_peak_allocated = torch.cuda.max_memory_allocated() / 2**30
            head_output_updated = bool(head.net.final_layer.linear.weight.count_nonzero())
            if not head_output_updated:
                raise RuntimeError('The fresh ROSS output layer did not update during warmup.')
            state_elements = sum(
                tensor.numel() for state in engine.optimizer.optimizer.state.values()
                for tensor in state.values() if torch.is_tensor(tensor)
            )
            trainable_elements = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            if state_elements < 2 * trainable_elements:
                raise RuntimeError('Full Adam first/second moments were not allocated before timing.')
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            forward_backward = []
            cuda_forward_backward = []
            optimizer_seconds = None
            last_loss = None
            for index in range(args.measure_microsteps):
                boundary = index == args.measure_microsteps - 1
                engine.set_gradient_accumulation_boundary(boundary)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                begin.record()
                result = engine(**batch)
                if not torch.isfinite(result.loss):
                    raise RuntimeError('ROSS measured loss is nonfinite.')
                engine.backward(result.loss / accum, scale_wrt_gas=False)
                end.record()
                torch.cuda.synchronize()
                forward_backward.append(time.perf_counter() - start)
                cuda_forward_backward.append(begin.elapsed_time(end) / 1000)
                start = time.perf_counter()
                engine.step()
                torch.cuda.synchronize()
                if boundary:
                    optimizer_seconds = time.perf_counter() - start
                    last_loss = float(result.loss.detach())
                del result
            mean_micro = statistics.mean(forward_backward)
            accum = 128 // micro
            estimated_update = mean_micro * accum + optimizer_seconds
            report = {
                'microbatch': micro, 'checkpointing': checkpoint,
                'accumulation': accum, 'expanded_length': args.sequence_length,
                'forward_backward_seconds': forward_backward,
                'cuda_event_seconds': cuda_forward_backward,
                'optimizer_seconds': optimizer_seconds,
                'estimated_global128_update_seconds': estimated_update,
                'estimated_samples_per_second': 128 / estimated_update,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                'last_loss': last_loss,
                'last_ross_loss': float(model.last_ross_loss),
                'last_text_loss': float(model.last_ross_text_loss),
                'last_ross_images': int(model.last_ross_num_images),
                'warmup_peak_allocated_gib': warmup_peak_allocated,
                'adam_state_elements': state_elements,
                'full_adam_state_verified': state_elements >= 2 * trainable_elements,
                'head_output_updated_after_warmup': head_output_updated,
                'vae_dtype_after_engine': str(next(vae.parameters()).dtype),
                'status': 'ok',
            }
        except torch.OutOfMemoryError as error:
            report = {'microbatch': micro, 'checkpointing': checkpoint, 'status': 'oom', 'error': str(error)}
            reports.append(report)
            output_path.write_text(json.dumps(reports, indent=2) + '\n')
            print('ROSS_BENCH_RESULT ' + json.dumps(report), flush=True)
            print('ROSS_BENCH_STOPPED_AFTER_OOM; restart process for further trials.', flush=True)
            return 2
        reports.append(report)
        output_path.write_text(json.dumps(reports, indent=2) + '\n')
        print('ROSS_BENCH_RESULT ' + json.dumps(report), flush=True)
        del batch, ids, labels
        gc.collect()
        torch.cuda.empty_cache()
    print('ROSS_BENCH_COMPLETE ' + json.dumps(reports), flush=True)
    torch.distributed.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
