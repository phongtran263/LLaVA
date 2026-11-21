"""Exercise projector-only PCGrad launchers without training or downloads."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SCRIPT = 'train_llama3_2_3b_cka_pcgrad.sh'
BASELINE_WRAPPER = 'finetune_llama3_2_3b_cka_from_baseline.sh'
STAGES = ('pretrain', 'finetune')


def value(argv, flag):
    return argv[argv.index(flag) + 1]


class LlamaCKAPCGradScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / 'scripts/v1_5/7b'
        scripts.mkdir(parents=True)
        self.script = scripts / SCRIPT
        for source, destination in (
            (REPO / 'scripts/v1_5/7b' / SCRIPT, self.script),
            (REPO / 'scripts/v1_5/7b' / BASELINE_WRAPPER, scripts / BASELINE_WRAPPER),
            (REPO / 'scripts/v1_5/pretrain_backbone.sh', scripts.parent / 'pretrain_backbone.sh'),
            (REPO / 'scripts/v1_5/7b/finetune_qwen2_5.sh', scripts / 'finetune_qwen2_5.sh'),
        ):
            shutil.copyfile(source, destination)
        conda_bin = self.root / 'fake-env/bin'
        conda_bin.mkdir(parents=True)
        # Skip the dependency preflight, but retain both actual shell launchers.
        python = conda_bin / 'python'
        python.write_text('#!/bin/bash\ncat >/dev/null\n')
        python.chmod(0o755)
        deepspeed = conda_bin / 'deepspeed'
        deepspeed.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
stage = 'pretrain' if '--tune_mm_mlp_adapter' in argv else 'finetune'
record = {'argv': argv, 'stage': stage,
          'env': {k: v for k, v in os.environ.items()
                  if k.startswith(('CKA_', 'VSP_', 'WANDB_'))
                  or k in ('RESUME_FROM_CHECKPOINT', 'STOP_AFTER_STEP_RATIO', 'PRETRAIN_ADAPTER')}}
if stage == 'finetune':
    record['adapter'] = Path(argv[argv.index('--pretrain_mm_mlp_adapter') + 1]).read_text()
with open(os.environ['CALL_LOG'], 'a') as handle:
    handle.write(json.dumps(record) + '\\n')
if os.environ.get('FAIL_STAGE') == stage:
    raise SystemExit(7)
if stage == 'pretrain' and os.environ.get('ARTIFACT_MODE') != 'missing':
    output = Path(argv[argv.index('--output_dir') + 1])
    output.mkdir(parents=True)
    (output / 'mm_projector.bin').write_bytes(
        b'' if os.environ.get('ARTIFACT_MODE') == 'empty' else b'new-stage1-adapter')
''')
        deepspeed.chmod(0o755)
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith(('CKA_', 'VSP_', 'WANDB_')) or key in (
                'RUN_NAME', 'GPU_INCLUDE', 'SEED', 'PER_DEVICE_TRAIN_BATCH_SIZE',
                'GRADIENT_ACCUMULATION_STEPS', 'PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE',
                'PRETRAIN_GRADIENT_ACCUMULATION_STEPS', 'FAIL_STAGE', 'ARTIFACT_MODE',
                'FINETUNE_PCGRAD',
            ):
                self.env.pop(key)
        self.env.update(
            CONDA_PREFIX=str(conda_bin.parent),
            PATH=str(conda_bin) + os.pathsep + self.env.get('PATH', ''),
            DRY_RUN='0', STAGE='all', MODEL_NAME_OR_PATH='wrong-model', MM_PROJECTOR_TYPE='wrong-projector',
            PRETRAIN_ADAPTER=str(self.root / 'inherited-missing-adapter.bin'),
            OUTPUT_DIR='wrong-output', CKA_LOSS='False', CKA_LOSS_ENABLED='False',
            CKA_LAYERS='3', CKA_LOSS_TAU='0.2', CKA_LOSS_START_RATIO='0.8',
            CKA_ANCHOR_LAYER='2', CKA_VISION_ANCHOR_LAYER='20',
            CKA_PROJECTOR_VISION_ANCHOR_LAYER='24', CKA_FINAL_VISION_ANCHOR_LAYER='24',
            VSP_PCGRAD='False', VSP_APPLY_TO_PROJECTOR_ONLY='False',
            VSP_DIAGNOSTICS='True', VSP_NORM_CAP='True', VSP_PCGRAD_THRESHOLD='0.9',
            RESUME_FROM_CHECKPOINT='wrong-checkpoint', STOP_AFTER_STEP_RATIO='0.8',
            WANDB_RUN_ID='old-run', WANDB_RESUME='must')
        self.use_scenario('default')

    def use_scenario(self, name):
        self.call_log = self.root / (name + '.jsonl')
        self.env.update(CALL_LOG=str(self.call_log),
                        OUTPUT_ROOT=str(self.root / name / 'outputs with spaces'))

    def run_script(self):
        return subprocess.run(['bash', str(self.script)], env=self.env,
                              capture_output=True, text=True, timeout=15)

    def calls(self):
        return [json.loads(line) for line in self.call_log.read_text().splitlines()]

    def output_for(self, stage):
        return Path(self.env['OUTPUT_ROOT']) / ('llava-' + stage)

    def use_baseline_wrapper(self):
        self.script = self.script.with_name(BASELINE_WRAPPER)
        self.env.pop('PRETRAIN_ADAPTER', None)
        self.env.pop('OUTPUT_ROOT', None)

    def baseline_adapter(self, run_name='llama-3.2-3b', seed='42'):
        return (f'./checkpoints/finetune-cka/{run_name}/'
                f'baseline-vs-cka-stage1-stage2-seed{seed}/baseline/llava-pretrain/mm_projector.bin')

    def baseline_output_root(self, run_name='llama-3.2-3b', seed='42'):
        return (f'./checkpoints/finetune-cka/{run_name}/'
                f'baseline-stage1-cka-proj-final-stage2-seed{seed}')

    def test_real_launchers_defaults_order_and_environment_isolation(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = self.calls()
        self.assertEqual([run['stage'] for run in runs], list(STAGES))
        self.assertEqual(len({value(run['argv'], '--run_name') for run in runs}), 2)
        self.assertEqual(len({run['env']['WANDB_RUN_GROUP'] for run in runs}), 1)
        for run, stage in zip(runs, STAGES):
            with self.subTest(stage=stage):
                argv, env = run['argv'], run['env']
                stage1 = stage == 'pretrain'
                expected = {
                    '--model_name_or_path': 'meta-llama/Llama-3.2-3B-Instruct',
                    '--mm_projector_type': 'mlp2x_gelu', '--include': 'localhost:3',
                    '--seed': '42', '--version': 'plain' if stage1 else 'auto',
                    '--output_dir': str(self.output_for(stage)),
                    '--per_device_train_batch_size': '32' if stage1 else '8',
                    '--gradient_accumulation_steps': '8' if stage1 else '16',
                    '--cka_loss': 'True', '--cka_loss_layers': '-1' if stage1 else 'final',
                    '--cka_loss_projector_weight': '0.1',
                    '--cka_loss_final_hidden_weight': '0.0' if stage1 else '0.1',
                    '--cka_loss_tau': '0.0', '--cka_loss_start_ratio': '0.0',
                    '--vsp_asymmetric_pcgrad': 'True',
                    '--vsp_apply_to_projector_only': 'True',
                    '--vsp_norm_cap': 'False', '--vsp_gradient_diagnostics': 'False',
                    '--vsp_pcgrad_threshold': '0.05', '--mm_vision_select_layer': '-2',
                    '--save_strategy': 'no', '--eval_strategy': 'no',
                }
                for flag, expected_value in expected.items():
                    self.assertEqual(value(argv, flag), expected_value, flag)
                for flag in ('--evaluation_strategy', '--resume_from_checkpoint', '--stop_after_step_ratio',
                             '--cka_loss_anchor_layer', '--cka_loss_vision_anchor_layer',
                             '--cka_loss_projector_vision_anchor_layer', '--cka_loss_final_vision_anchor_layer'):
                    self.assertNotIn(flag, argv)
                for key in ('CKA_ANCHOR_LAYER', 'CKA_VISION_ANCHOR_LAYER',
                            'CKA_PROJECTOR_VISION_ANCHOR_LAYER', 'CKA_FINAL_VISION_ANCHOR_LAYER',
                            'RESUME_FROM_CHECKPOINT', 'STOP_AFTER_STEP_RATIO'):
                    self.assertEqual(env[key], '')
                self.assertNotIn('WANDB_RUN_ID', env)
                self.assertNotIn('WANDB_RESUME', env)
                if stage1:
                    self.assertEqual(value(argv, '--tune_mm_mlp_adapter'), 'True')
                    self.assertNotIn('--pretrain_mm_mlp_adapter', argv)
                    self.assertEqual(env['PRETRAIN_ADAPTER'], '')
                else:
                    self.assertNotIn('--tune_mm_mlp_adapter', argv)
                    self.assertEqual(value(argv, '--pretrain_mm_mlp_adapter'),
                                     str(self.output_for('pretrain') / 'mm_projector.bin'))
                    self.assertEqual(run['adapter'], 'new-stage1-adapter')

    def test_batch_seed_gpu_and_cka_weight_overrides(self):
        self.env.update(PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE='3',
                        PRETRAIN_GRADIENT_ACCUMULATION_STEPS='5',
                        PER_DEVICE_TRAIN_BATCH_SIZE='7', GRADIENT_ACCUMULATION_STEPS='11',
                        CKA_PROJECTOR_WEIGHT='0.25', CKA_FINAL_HIDDEN_WEIGHT='0.4',
                        SEED='123', GPU_INCLUDE='localhost:1', RUN_NAME='custom-llama')
        self.env.pop('PRETRAIN_ADAPTER')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        for run in self.calls():
            argv, stage1 = run['argv'], run['stage'] == 'pretrain'
            self.assertEqual(value(argv, '--per_device_train_batch_size'), '3' if stage1 else '7')
            self.assertEqual(value(argv, '--gradient_accumulation_steps'), '5' if stage1 else '11')
            self.assertEqual(value(argv, '--cka_loss_projector_weight'), '0.25')
            self.assertEqual(value(argv, '--cka_loss_final_hidden_weight'), '0.0' if stage1 else '0.4')
            self.assertEqual(value(argv, '--seed'), '123')
            self.assertEqual(value(argv, '--include'), 'localhost:1')
            self.assertTrue(value(argv, '--run_name').startswith('custom-llama-'))

    def test_real_cli_parses_with_training_dataclasses(self):
        from transformers import HfArgumentParser
        from llava.train.train import DataArguments, ModelArguments, TrainingArguments

        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        for run in self.calls():
            argv = run['argv']
            training_argv = argv[argv.index('llava/train/train_mem.py') + 1:]
            # Skip hardware/DeepSpeed initialization, not CLI parsing or types.
            with mock.patch.object(TrainingArguments, '__post_init__', return_value=None):
                model_args, _, args = parser.parse_args_into_dataclasses(training_argv, look_for_args_file=False)
            self.assertTrue(model_args.vsp_asymmetric_pcgrad)
            self.assertTrue(model_args.vsp_apply_to_projector_only)
            self.assertFalse(model_args.use_pcgrad)
            self.assertFalse(model_args.vsp_norm_cap)
            self.assertFalse(model_args.vsp_gradient_diagnostics)
            self.assertTrue(model_args.cka_loss)
            self.assertEqual(args.eval_strategy, 'no')

    def test_each_stage_failure_stops_pipeline(self):
        for index, stage in enumerate(STAGES):
            with self.subTest(stage=stage):
                self.use_scenario('failure-' + stage)
                self.env['FAIL_STAGE'] = stage
                result = self.run_script()
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual([run['stage'] for run in self.calls()], list(STAGES[:index + 1]))

    def test_finetune_only_reuses_existing_adapter_without_pretraining(self):
        output = self.output_for('pretrain')
        output.mkdir(parents=True)
        adapter = output / 'mm_projector.bin'
        adapter.write_text('existing-stage1-adapter')
        self.env.update(STAGE='finetune', PRETRAIN_ADAPTER=str(adapter),
                        WANDB_RUN_NAME='intentional-finetune-name')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = self.calls()
        self.assertEqual([run['stage'] for run in runs], ['finetune'])
        self.assertEqual(runs[0]['adapter'], 'existing-stage1-adapter')
        self.assertEqual(adapter.read_text(), 'existing-stage1-adapter')
        self.assertEqual(value(runs[0]['argv'], '--vsp_asymmetric_pcgrad'), 'True')
        self.assertEqual(value(runs[0]['argv'], '--vsp_apply_to_projector_only'), 'True')
        self.assertEqual(value(runs[0]['argv'], '--cka_loss_final_hidden_weight'), '0.1')
        self.assertEqual(value(runs[0]['argv'], '--run_name'), 'intentional-finetune-name')
        self.assertEqual(runs[0]['env']['WANDB_RUN_NAME'], 'intentional-finetune-name')

    def test_baseline_wrapper_defaults_reuse_baseline_adapter_and_only_finetune(self):
        self.use_baseline_wrapper()
        adapter = self.root / self.baseline_adapter()
        adapter.parent.mkdir(parents=True)
        adapter.write_text('baseline-stage1-adapter')
        self.assertEqual(self.env['STAGE'], 'all')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = self.calls()
        self.assertEqual([run['stage'] for run in runs], ['finetune'])
        run = runs[0]
        expected = {
            '--pretrain_mm_mlp_adapter': self.baseline_adapter(),
            '--output_dir': self.baseline_output_root() + '/llava-finetune',
            '--run_name': 'llama-3.2-3b-baseline-stage1-cka-proj-final-finetune-seed42',
            '--seed': '42', '--cka_loss': 'True', '--cka_loss_layers': 'final',
            '--cka_loss_projector_weight': '0.1', '--cka_loss_final_hidden_weight': '0.1',
            '--vsp_asymmetric_pcgrad': 'False', '--vsp_apply_to_projector_only': 'False',
            '--vsp_norm_cap': 'False', '--vsp_gradient_diagnostics': 'False',
            '--mm_vision_select_layer': '-2',
        }
        for flag, expected_value in expected.items():
            self.assertEqual(value(run['argv'], flag), expected_value, flag)
        for flag in ('--tune_mm_mlp_adapter', '--cka_loss_anchor_layer',
                     '--cka_loss_vision_anchor_layer', '--cka_loss_projector_vision_anchor_layer',
                     '--cka_loss_final_vision_anchor_layer'):
            self.assertNotIn(flag, run['argv'])
        self.assertEqual(run['env']['WANDB_RUN_GROUP'],
                         'llama-3.2-3b-baseline-stage1-cka-proj-final-stage2-seed42')
        self.assertEqual(run['env']['WANDB_RUN_NAME'], expected['--run_name'])
        self.assertEqual(run['adapter'], 'baseline-stage1-adapter')
        self.assertEqual(adapter.read_text(), 'baseline-stage1-adapter')
        self.assertFalse((self.root / self.baseline_output_root() / 'llava-pretrain').exists())

    def test_baseline_wrapper_cli_disables_inherited_pcgrad_without_disabling_cka(self):
        from transformers import HfArgumentParser
        from llava.train.train import DataArguments, ModelArguments, TrainingArguments

        self.use_baseline_wrapper()
        adapter = self.root / 'baseline-adapter.bin'
        adapter.write_text('baseline-stage1-adapter')
        self.env.update(PRETRAIN_ADAPTER=str(adapter), FINETUNE_PCGRAD='True',
                        VSP_PCGRAD='True', VSP_APPLY_TO_PROJECTOR_ONLY='True')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = self.calls()
        self.assertEqual([run['stage'] for run in runs], ['finetune'])
        argv = runs[0]['argv']
        training_argv = argv[argv.index('llava/train/train_mem.py') + 1:]
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        with mock.patch.object(TrainingArguments, '__post_init__', return_value=None):
            model_args, _, _ = parser.parse_args_into_dataclasses(training_argv, look_for_args_file=False)
        self.assertFalse(model_args.vsp_asymmetric_pcgrad)
        self.assertFalse(model_args.vsp_apply_to_projector_only)
        self.assertFalse(model_args.use_pcgrad)
        self.assertFalse(model_args.vsp_norm_cap)
        self.assertFalse(model_args.vsp_gradient_diagnostics)
        self.assertTrue(model_args.cka_loss)
        self.assertEqual(model_args.cka_loss_layers, 'final')
        self.assertEqual(model_args.cka_loss_projector_weight, 0.1)
        self.assertEqual(model_args.cka_loss_final_hidden_weight, 0.1)

    def test_baseline_wrapper_honors_adapter_output_and_label_overrides(self):
        self.script = self.script.with_name(BASELINE_WRAPPER)
        adapter = self.root / 'custom adapter.bin'
        adapter.write_text('custom-baseline-adapter')
        self.env.update(PRETRAIN_ADAPTER=str(adapter), RUN_NAME='custom-llama', SEED='123',
                        WANDB_RUN_GROUP='custom-group', WANDB_RUN_NAME='custom-finetune')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = self.calls()
        self.assertEqual([run['stage'] for run in runs], ['finetune'])
        self.assertEqual(value(runs[0]['argv'], '--pretrain_mm_mlp_adapter'), str(adapter))
        self.assertEqual(value(runs[0]['argv'], '--output_dir'), str(self.output_for('finetune')))
        self.assertEqual(value(runs[0]['argv'], '--run_name'), 'custom-finetune')
        self.assertEqual(value(runs[0]['argv'], '--seed'), '123')
        self.assertEqual(runs[0]['env']['WANDB_RUN_GROUP'], 'custom-group')
        self.assertEqual(runs[0]['adapter'], 'custom-baseline-adapter')
        self.assertEqual(adapter.read_text(), 'custom-baseline-adapter')

    def test_baseline_wrapper_rejects_missing_or_empty_adapter_without_pretraining(self):
        self.use_baseline_wrapper()
        adapter = self.root / self.baseline_adapter()
        for mode in ('missing', 'empty'):
            with self.subTest(mode=mode):
                if mode == 'empty':
                    adapter.parent.mkdir(parents=True)
                    adapter.touch()
                result = self.run_script()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('nonempty adapter', result.stderr)
                self.assertIn(self.baseline_adapter(), result.stderr)
                self.assertFalse(self.call_log.exists())
                self.assertFalse((self.root / self.baseline_output_root()).exists())

    def test_baseline_wrapper_dry_run_prints_only_finetune_with_derived_defaults(self):
        self.use_baseline_wrapper()
        self.env.update(DRY_RUN='1')
        self.env.pop('CONDA_PREFIX')
        for run_name, seed in (('llama-3.2-3b', '42'), ('custom-llama', '123')):
            with self.subTest(run_name=run_name, seed=seed):
                if seed != '42':
                    self.env.update(RUN_NAME=run_name, SEED=seed)
                result = self.run_script()
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [shlex.split(line) for line in result.stdout.splitlines()
                            if line.startswith('env ')]
                self.assertEqual(len(commands), 1)
                command = commands[0]
                for assignment in (
                    'PRETRAIN_ADAPTER=' + self.baseline_adapter(run_name, seed),
                    'OUTPUT_DIR=' + self.baseline_output_root(run_name, seed) + '/llava-finetune',
                    f'WANDB_RUN_GROUP={run_name}-baseline-stage1-cka-proj-final-stage2-seed{seed}',
                    f'WANDB_RUN_NAME={run_name}-baseline-stage1-cka-proj-final-finetune-seed{seed}',
                    'CKA_LAYERS=final', 'CKA_PROJECTOR_WEIGHT=0.1', 'CKA_FINAL_HIDDEN_WEIGHT=0.1',
                    'VSP_PCGRAD=False', 'VSP_APPLY_TO_PROJECTOR_ONLY=False', 'SEED=' + seed,
                ):
                    self.assertIn(assignment, command)
                self.assertEqual(command[-2:], ['bash', str(self.script.with_name('finetune_qwen2_5.sh'))])
                self.assertFalse(self.call_log.exists())
                self.assertFalse((self.root / 'checkpoints').exists())

    def test_finetune_only_rejects_missing_adapter_before_launch(self):
        self.env['STAGE'] = 'finetune'
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('nonempty adapter', result.stderr)
        self.assertFalse(self.call_log.exists())

    def test_finetune_only_dry_run_and_invalid_stage(self):
        self.env.update(STAGE='finetune', DRY_RUN='1')
        self.env.pop('CONDA_PREFIX')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('env ')]
        self.assertEqual(len(commands), 1)
        self.assertIn('PRETRAIN_ADAPTER=' + self.env['PRETRAIN_ADAPTER'], commands[0])
        self.assertTrue(commands[0][-1].endswith('finetune_qwen2_5.sh'))
        self.assertFalse(self.call_log.exists())
        self.env['STAGE'] = 'invalid'
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('STAGE must be all or finetune', result.stderr)

    def test_missing_or_empty_stage1_adapter_prevents_stage2(self):
        for mode in ('missing', 'empty'):
            with self.subTest(mode=mode):
                self.use_scenario(mode)
                self.env['ARTIFACT_MODE'] = mode
                result = self.run_script()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Stage 1 did not save a nonempty adapter', result.stderr)
                self.assertIn(str(self.output_for('pretrain') / 'mm_projector.bin'), result.stderr)
                self.assertEqual([run['stage'] for run in self.calls()], ['pretrain'])

    def test_any_existing_output_prevents_all_launches(self):
        for stage in STAGES:
            for kind in ('directory', 'file', 'dangling-symlink'):
                with self.subTest(stage=stage, kind=kind):
                    self.use_scenario(stage + '-' + kind)
                    output = self.output_for(stage)
                    output.parent.mkdir(parents=True)
                    if kind == 'directory':
                        output.mkdir()
                    elif kind == 'file':
                        output.write_text('do not overwrite')
                    else:
                        output.symlink_to(self.root / 'nonexistent-target')
                    result = self.run_script()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Output already exists', result.stderr)
                    self.assertIn(str(output), result.stderr)
                    self.assertFalse(self.call_log.exists())

    def test_dry_run_prints_two_commands_without_environment_or_artifacts(self):
        self.env['DRY_RUN'] = '1'
        self.env.pop('CONDA_PREFIX')
        self.env.pop('PRETRAIN_ADAPTER')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('env ')]
        self.assertEqual(len(commands), 2)
        for command, stage in zip(commands, STAGES):
            self.assertIn('OUTPUT_DIR=' + str(self.output_for(stage)), command)
            self.assertIn('CKA_LOSS=True', command)
            self.assertIn('CKA_LOSS_ENABLED=True', command)
            self.assertIn('VSP_PCGRAD=True', command)
            self.assertIn('VSP_APPLY_TO_PROJECTOR_ONLY=True', command)
            self.assertTrue(command[-1].endswith(
                'pretrain_backbone.sh' if stage == 'pretrain' else 'finetune_qwen2_5.sh'))
        self.assertIn('PRETRAIN_ADAPTER=' + str(self.output_for('pretrain') / 'mm_projector.bin'), commands[1])
        self.assertFalse(self.call_log.exists())
        self.assertFalse(Path(self.env['OUTPUT_ROOT']).exists())

    def test_default_output_root_and_seed(self):
        self.env.update(DRY_RUN='1')
        self.env.pop('OUTPUT_ROOT')
        self.env.pop('CONDA_PREFIX')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        root = './checkpoints/finetune-cka/llama-3.2-3b/cka-proj-final-pcgrad-stage1-stage2-seed42'
        for stage in STAGES:
            self.assertIn('OUTPUT_DIR=' + root + '/llava-' + stage, result.stdout)
        self.assertFalse(self.call_log.exists())
        self.assertFalse((self.root / 'checkpoints').exists())


if __name__ == '__main__':
    unittest.main()
