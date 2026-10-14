"""Exercise both stages of the sequential runner without training or downloads."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SCRIPT = 'finetune_llama3_2_3b_baseline_cka.sh'
JOBS = (
    ('baseline', 'stage1'),
    ('baseline', 'stage2'),
    ('cka-proj-final', 'stage1'),
    ('cka-proj-final', 'stage2'),
)
STAGE_OUTPUTS = {'stage1': 'llava-pretrain', 'stage2': 'llava-finetune'}


class LlamaBaselineCKAScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / 'scripts/v1_5/7b'
        scripts.mkdir(parents=True)
        self.script = scripts / SCRIPT
        shutil.copyfile(REPO / 'scripts/v1_5/7b' / SCRIPT, self.script)
        # Record the stage and environment, simulate failures, and produce each
        # variant's own adapter for its corresponding stage 2.
        fake_child = '''#!/bin/bash
FAKE_STAGE=STAGE python3 - <<'PY'
import json, os
from pathlib import Path
keys = [k for k in os.environ if k.startswith(('CKA_', 'WANDB_', 'VSP_'))]
keys += ['MODEL_NAME_OR_PATH', 'RUN_NAME', 'GPU_INCLUDE', 'SEED',
         'MM_PROJECTOR_TYPE', 'PRETRAIN_ADAPTER', 'OUTPUT_DIR',
         'PER_DEVICE_TRAIN_BATCH_SIZE', 'GRADIENT_ACCUMULATION_STEPS',
         'RESUME_FROM_CHECKPOINT', 'STOP_AFTER_STEP_RATIO']
record = {k: os.environ.get(k) for k in keys}
record['stage'] = os.environ['FAKE_STAGE']
output = Path(os.environ['OUTPUT_DIR'])
record['variant'] = output.parent.name
job = record['variant'] + '-' + record['stage']
with open(os.environ['CALL_LOG'], 'a') as handle:
    handle.write(json.dumps(record) + '\\n')
if os.environ.get('FAIL_JOB') == job:
    raise SystemExit(7)
if record['stage'] == 'stage1':
    mode = os.environ.get('ARTIFACT_MODE', 'normal') if os.environ.get('ARTIFACT_JOB') == job else 'normal'
    if mode != 'missing':
        output.mkdir(parents=True, exist_ok=True)
        (output / 'mm_projector.bin').write_bytes(b'' if mode == 'empty' else job.encode())
PY
'''
        (scripts.parent / 'pretrain_backbone.sh').write_text(
            fake_child.replace('FAKE_STAGE=STAGE', 'FAKE_STAGE=stage1'))
        (scripts / 'finetune_qwen2_5.sh').write_text(
            fake_child.replace('FAKE_STAGE=STAGE', 'FAKE_STAGE=stage2'))
        conda_bin = self.root / 'fake-env/bin'
        conda_bin.mkdir(parents=True)
        fake_deepspeed = conda_bin / 'deepspeed'
        fake_deepspeed.write_text('#!/bin/bash\nexit 99\n')
        fake_deepspeed.chmod(0o755)
        self.call_log = self.root / 'calls.jsonl'
        self.env = dict(os.environ)
        for key in ('PER_DEVICE_TRAIN_BATCH_SIZE', 'GRADIENT_ACCUMULATION_STEPS',
                    'PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE',
                    'PRETRAIN_GRADIENT_ACCUMULATION_STEPS', 'WANDB_RUN_GROUP',
                    'FAIL_JOB', 'ARTIFACT_JOB', 'ARTIFACT_MODE'):
            self.env.pop(key, None)
        self.env.update(
            CONDA_PREFIX=str(conda_bin.parent),
            PRETRAIN_ADAPTER=str(self.root / 'inherited-missing-adapter.bin'),
            CALL_LOG=str(self.call_log), SWEEP_OUTPUT_ROOT=str(self.root / 'outputs'),
            DRY_RUN='0', SEED='123', RUN_NAME='llama-3.2-3b', GPU_INCLUDE='localhost:3',
            MODEL_NAME_OR_PATH='wrong-model', MM_PROJECTOR_TYPE='wrong-projector',
            CKA_LOSS='wrong', CKA_LOSS_ENABLED='wrong',
            CKA_PROJECTOR_WEIGHT='0.1', CKA_FINAL_HIDDEN_WEIGHT='0.1',
            CKA_LAYERS='3', CKA_LOSS_START_RATIO='0.8', CKA_LOSS_TAU='0.2',
            CKA_ANCHOR_LAYER='2', CKA_VISION_ANCHOR_LAYER='20',
            CKA_PROJECTOR_VISION_ANCHOR_LAYER='24', CKA_FINAL_VISION_ANCHOR_LAYER='24',
            VSP_DIAGNOSTICS='True', VSP_PCGRAD='True', VSP_NORM_CAP='True',
            RESUME_FROM_CHECKPOINT='wrong-checkpoint', STOP_AFTER_STEP_RATIO='0.8',
            WANDB_RUN_ID='old-run', WANDB_RESUME='must',
            OUTPUT_DIR='wrong-output', WANDB_RUN_NAME='wrong-name')

    def run_script(self):
        return subprocess.run(['bash', str(self.script)], env=self.env,
                              capture_output=True, text=True, timeout=15)

    def calls(self):
        return [json.loads(line) for line in self.call_log.read_text().splitlines()]

    def output_for(self, variant, stage):
        return Path(self.env['SWEEP_OUTPUT_ROOT']) / variant / STAGE_OUTPUTS[stage]

    def use_scenario(self, scenario):
        self.env['SWEEP_OUTPUT_ROOT'] = str(self.root / scenario / 'outputs')
        self.call_log = self.root / (scenario + '.jsonl')
        self.env['CALL_LOG'] = str(self.call_log)

    def test_order_independent_initialization_and_environment_isolation(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = self.calls()
        self.assertEqual([(run['variant'], run['stage']) for run in runs], list(JOBS))
        for run, (variant, stage) in zip(runs, JOBS):
            with self.subTest(variant=variant, stage=stage):
                enabled = 'True' if variant == 'cka-proj-final' else 'False'
                self.assertEqual(run['CKA_LOSS' if stage == 'stage1' else 'CKA_LOSS_ENABLED'], enabled)
                self.assertEqual(run['CKA_PROJECTOR_WEIGHT'], '0.1' if enabled == 'True' else '0.0')
                self.assertEqual(run['CKA_FINAL_HIDDEN_WEIGHT'],
                                 '0.1' if enabled == 'True' and stage == 'stage2' else '0.0')
                self.assertEqual(run['MODEL_NAME_OR_PATH'], 'meta-llama/Llama-3.2-3B-Instruct')
                self.assertEqual(run['MM_PROJECTOR_TYPE'], 'mlp2x_gelu')
                self.assertEqual(run['GPU_INCLUDE'], 'localhost:3')
                self.assertEqual(run['SEED'], '123')
                self.assertEqual(run['CKA_LAYERS'], '-1' if stage == 'stage1' else 'final')
                self.assertEqual(run['PER_DEVICE_TRAIN_BATCH_SIZE'], '32' if stage == 'stage1' else '8')
                self.assertEqual(run['GRADIENT_ACCUMULATION_STEPS'], '8' if stage == 'stage1' else '16')
                self.assertEqual(run['OUTPUT_DIR'], str(self.output_for(variant, stage)))
                if stage == 'stage2':
                    adapter = self.output_for(variant, 'stage1') / 'mm_projector.bin'
                    self.assertEqual(run['PRETRAIN_ADAPTER'], str(adapter))
                    self.assertEqual(adapter.read_bytes(), (variant + '-stage1').encode())
                self.assertEqual(run['CKA_LOSS_START_RATIO'], '0.0')
                self.assertEqual(run['CKA_LOSS_TAU'], '0.0')
                for key in ('VSP_DIAGNOSTICS', 'VSP_PCGRAD', 'VSP_NORM_CAP',
                            'VSP_APPLY_TO_PROJECTOR_ONLY'):
                    self.assertEqual(run[key], 'False')
                for key in ('CKA_ANCHOR_LAYER', 'CKA_VISION_ANCHOR_LAYER',
                            'CKA_PROJECTOR_VISION_ANCHOR_LAYER', 'CKA_FINAL_VISION_ANCHOR_LAYER',
                            'RESUME_FROM_CHECKPOINT', 'STOP_AFTER_STEP_RATIO'):
                    self.assertEqual(run[key], '')
                self.assertNotIn('WANDB_RUN_ID', run)
                self.assertNotIn('WANDB_RESUME', run)
        self.assertEqual(len({run['OUTPUT_DIR'] for run in runs}), 4)
        self.assertEqual(len({run['WANDB_RUN_NAME'] for run in runs}), 4)
        self.assertEqual(len({run['WANDB_RUN_GROUP'] for run in runs}), 1)

    def test_stage_batch_and_cka_weight_overrides(self):
        self.env.update(PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE='3',
                        PRETRAIN_GRADIENT_ACCUMULATION_STEPS='5',
                        PER_DEVICE_TRAIN_BATCH_SIZE='7', GRADIENT_ACCUMULATION_STEPS='11',
                        CKA_PROJECTOR_WEIGHT='0.25', CKA_FINAL_HIDDEN_WEIGHT='0.4')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        for run in self.calls():
            stage1 = run['stage'] == 'stage1'
            self.assertEqual(run['PER_DEVICE_TRAIN_BATCH_SIZE'], '3' if stage1 else '7')
            self.assertEqual(run['GRADIENT_ACCUMULATION_STEPS'], '5' if stage1 else '11')
            cka = run['variant'] == 'cka-proj-final'
            self.assertEqual(run['CKA_PROJECTOR_WEIGHT'], '0.25' if cka else '0.0')
            self.assertEqual(run['CKA_FINAL_HIDDEN_WEIGHT'], '0.4' if cka and not stage1 else '0.0')

    def test_each_job_failure_prevents_later_jobs(self):
        for index, (variant, stage) in enumerate(JOBS):
            with self.subTest(variant=variant, stage=stage):
                self.use_scenario('failure-' + variant + '-' + stage)
                self.env['FAIL_JOB'] = variant + '-' + stage
                result = self.run_script()
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual([(run['variant'], run['stage']) for run in self.calls()],
                                 list(JOBS[:index + 1]))

    def test_missing_or_empty_generated_adapter_prevents_its_stage2(self):
        for variant, expected_count in (('baseline', 1), ('cka-proj-final', 3)):
            for mode in ('missing', 'empty'):
                with self.subTest(variant=variant, mode=mode):
                    self.use_scenario(mode + '-' + variant)
                    self.env.update(ARTIFACT_JOB=variant + '-stage1', ARTIFACT_MODE=mode)
                    result = self.run_script()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Stage 1 did not save a nonempty adapter', result.stderr)
                    self.assertIn(str(self.output_for(variant, 'stage1') / 'mm_projector.bin'),
                                  result.stderr)
                    self.assertEqual([(run['variant'], run['stage']) for run in self.calls()],
                                     list(JOBS[:expected_count]))

    def test_real_launchers_receive_expected_arguments(self):
        from transformers import HfArgumentParser
        from llava.train.train import DataArguments, ModelArguments, TrainingArguments

        scripts = self.script.parent
        shutil.copyfile(REPO / 'scripts/v1_5/pretrain_backbone.sh',
                        scripts.parent / 'pretrain_backbone.sh')
        shutil.copyfile(REPO / 'scripts/v1_5/7b/finetune_qwen2_5.sh',
                        scripts / 'finetune_qwen2_5.sh')
        conda_bin = Path(self.env['CONDA_PREFIX']) / 'bin'
        # Skip only the dependency import check. Both real launchers expand every
        # flag; fake DeepSpeed records argv and emits fake stage-1 adapters.
        python = conda_bin / 'python'
        python.write_text('#!/bin/bash\ncat >/dev/null\n')
        python.chmod(0o755)
        (conda_bin / 'deepspeed').write_text('''#!/bin/bash
python3 - "$@" <<'PY'
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
with open(os.environ['CALL_LOG'], 'a') as handle:
    handle.write(json.dumps(argv) + '\\n')
if '--tune_mm_mlp_adapter' in argv and argv[argv.index('--tune_mm_mlp_adapter') + 1] == 'True':
    output = Path(argv[argv.index('--output_dir') + 1])
    output.mkdir(parents=True)
    (output / 'mm_projector.bin').write_bytes(b'fake-stage1-adapter')
PY
''')
        self.env['PATH'] = str(conda_bin) + os.pathsep + self.env.get('PATH', '')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = self.calls()
        self.assertEqual(len(runs), 4)

        # Validate the real CLI against installed dataclasses, not only strings.
        # Skip TrainingArguments hardware/DeepSpeed initialization, but keep the
        # actual HfArgumentParser and all four launchers' flags and values.
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        for argv in runs:
            training_argv = argv[argv.index('llava/train/train_mem.py') + 1:]
            with mock.patch.object(TrainingArguments, '__post_init__', return_value=None):
                _, _, args = parser.parse_args_into_dataclasses(
                    training_argv, look_for_args_file=False,
                )
            self.assertEqual(args.eval_strategy, 'no')

        def value(argv, flag):
            return argv[argv.index(flag) + 1]

        self.assertEqual(len({value(argv, '--run_name') for argv in runs}), 4)
        for argv, (variant, stage) in zip(runs, JOBS):
            with self.subTest(variant=variant, stage=stage):
                stage1 = stage == 'stage1'
                cka = variant == 'cka-proj-final'
                self.assertEqual(value(argv, '--model_name_or_path'), 'meta-llama/Llama-3.2-3B-Instruct')
                self.assertEqual(value(argv, '--mm_projector_type'), 'mlp2x_gelu')
                self.assertEqual(value(argv, '--version'), 'plain' if stage1 else 'auto')
                self.assertEqual(value(argv, '--include'), 'localhost:3')
                self.assertEqual(value(argv, '--seed'), '123')
                self.assertEqual(value(argv, '--output_dir'), str(self.output_for(variant, stage)))
                self.assertEqual(value(argv, '--per_device_train_batch_size'), '32' if stage1 else '8')
                self.assertEqual(value(argv, '--gradient_accumulation_steps'), '8' if stage1 else '16')
                self.assertEqual(value(argv, '--save_strategy'), 'no')
                self.assertEqual(value(argv, '--eval_strategy'), 'no')
                self.assertNotIn('--evaluation_strategy', argv)
                self.assertEqual(value(argv, '--cka_loss_layers'), '-1' if stage1 else 'final')
                self.assertEqual(value(argv, '--cka_loss'), 'True' if cka else 'False')
                self.assertEqual(value(argv, '--cka_loss_projector_weight'), '0.1' if cka else '0.0')
                self.assertEqual(value(argv, '--cka_loss_final_hidden_weight'),
                                 '0.1' if cka and not stage1 else '0.0')
                self.assertEqual(value(argv, '--cka_loss_tau'), '0.0')
                self.assertEqual(value(argv, '--cka_loss_start_ratio'), '0.0')
                self.assertEqual(value(argv, '--mm_vision_select_layer'), '-2')
                if stage1:
                    self.assertEqual(value(argv, '--tune_mm_mlp_adapter'), 'True')
                    self.assertNotIn('--pretrain_mm_mlp_adapter', argv)
                else:
                    self.assertNotIn('--tune_mm_mlp_adapter', argv)
                    self.assertEqual(value(argv, '--pretrain_mm_mlp_adapter'),
                                     str(self.output_for(variant, 'stage1') / 'mm_projector.bin'))
                for flag in ('--vsp_gradient_diagnostics', '--vsp_asymmetric_pcgrad', '--vsp_norm_cap',
                             '--vsp_apply_to_projector_only'):
                    self.assertEqual(value(argv, flag), 'False')
                for flag in ('--resume_from_checkpoint', '--stop_after_step_ratio',
                             '--cka_loss_anchor_layer', '--cka_loss_vision_anchor_layer',
                             '--cka_loss_projector_vision_anchor_layer', '--cka_loss_final_vision_anchor_layer'):
                    self.assertNotIn(flag, argv)

    def test_generic_backbone_launchers_use_current_eval_flag(self):
        for name in ('pretrain_backbone.sh', 'finetune_backbone.sh'):
            with self.subTest(script=name):
                script = (REPO / 'scripts/v1_5' / name).read_text()
                self.assertIn('--eval_strategy no', script)
                self.assertNotIn('--evaluation_strategy', script)

    def test_existing_any_output_prevents_all_jobs(self):
        for variant, stage in JOBS:
            with self.subTest(variant=variant, stage=stage):
                self.use_scenario('existing-' + variant + '-' + stage)
                output = self.output_for(variant, stage)
                output.mkdir(parents=True)
                result = self.run_script()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Output already exists', result.stderr)
                self.assertIn(str(output), result.stderr)
                self.assertFalse(self.call_log.exists())

    def test_dangling_output_symlink_prevents_all_jobs(self):
        output = self.output_for('cka-proj-final', 'stage2')
        output.parent.mkdir(parents=True)
        output.symlink_to(self.root / 'nonexistent-output')
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output already exists', result.stderr)
        self.assertFalse(self.call_log.exists())

    def test_dry_run_prints_four_commands_without_environment_or_adapters(self):
        self.env['DRY_RUN'] = '1'
        self.env.pop('CONDA_PREFIX', None)
        self.env.pop('PRETRAIN_ADAPTER', None)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if line.startswith('env ')]
        self.assertEqual(len(commands), 4)
        for command, (variant, stage) in zip(commands, JOBS):
            self.assertIn('OUTPUT_DIR=' + str(self.output_for(variant, stage)), command)
            launcher = 'pretrain_backbone.sh' if stage == 'stage1' else 'finetune_qwen2_5.sh'
            self.assertIn(launcher, command)
            enabled = 'True' if variant == 'cka-proj-final' else 'False'
            self.assertIn(('CKA_LOSS=' if stage == 'stage1' else 'CKA_LOSS_ENABLED=') + enabled,
                          command)
            if stage == 'stage2':
                self.assertIn('PRETRAIN_ADAPTER=' + str(self.output_for(variant, 'stage1') / 'mm_projector.bin'),
                              command)
        self.assertFalse(self.call_log.exists())
        self.assertFalse(Path(self.env['SWEEP_OUTPUT_ROOT']).exists())

    def test_default_output_root_includes_both_stages(self):
        self.env.update(DRY_RUN='1')
        self.env.pop('SWEEP_OUTPUT_ROOT')
        self.env.pop('CONDA_PREFIX', None)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        root = './checkpoints/finetune-cka/llama-3.2-3b/baseline-vs-cka-stage1-stage2-seed123'
        for variant, stage in JOBS:
            self.assertIn('OUTPUT_DIR=' + root + '/' + variant + '/' + STAGE_OUTPUTS[stage],
                          result.stdout)
        self.assertFalse(self.call_log.exists())

    def test_no_initial_adapter_is_required(self):
        self.env.pop('PRETRAIN_ADAPTER', None)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 4)


if __name__ == '__main__':
    unittest.main()
