import os, json, runpy, argparse, gc
import torch
import deepspeed
from torch.utils.data import Dataset, SequentialSampler
from safetensors.torch import load_file
from llava.constants import IMAGE_TOKEN_INDEX
from llava.train.train import TrainingArguments
from llava.train.llava_trainer import LLaVATrainer
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
a = parser.parse_args()
repo = '/home/jovyan/dowser-lora-datavol-1/phong/LLaVA'
helpers = runpy.run_path(repo + '/tests/test_predictive_regularization.py')

class TinyData(Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        visual = i % 2 == 0
        ids = torch.tensor([1, IMAGE_TOKEN_INDEX if visual else 5, 2, 3])
        return dict(input_ids=ids, labels=ids.clone(), attention_mask=torch.ones_like(ids), images=torch.zeros(3,2,2))

def collate(items):
    return {k: torch.stack([x[k] for x in items]) for k in items[0]}

class TinyTrainer(LLaVATrainer):
    def _get_train_sampler(self): return SequentialSampler(self.train_dataset)

reports=[]
for accum in (1,2):
    directory = f'/tmp/pre-zero2-smoke-accum{accum}'
    args = TrainingArguments(output_dir=directory, per_device_train_batch_size=1,
        gradient_accumulation_steps=accum, max_steps=2, num_train_epochs=1,
        learning_rate=2e-5, weight_decay=0.0, bf16=True, fp16=False,
        gradient_checkpointing=False, optim='adamw_torch_fused',
        deepspeed=repo+'/scripts/zero2_b200.json', report_to=[], logging_steps=1,
        save_strategy='steps', save_steps=2, save_total_limit=1,
        dataloader_num_workers=2, dataloader_prefetch_factor=4, dataloader_pin_memory=True, dataloader_persistent_workers=True, accelerator_config={'non_blocking':True}, remove_unused_columns=False, local_rank=a.local_rank,
        disable_tqdm=True, seed=733, tf32=True)
    torch.manual_seed(733)
    model=helpers['_model']().to(dtype=torch.bfloat16)
    expected_init={k:v.detach().float().cpu().clone() for k,v in model.get_model().pre_predictor.state_dict().items()}
    calls=[]
    def observe(module,inputs,output):
        calls.append({'shape':list(inputs[0].shape),'dtype':str(inputs[0].dtype), 'output_dtype':str(output.dtype)})
    handle=model.get_model().pre_predictor.register_forward_hook(observe)
    trainer=TinyTrainer(model=model,args=args,train_dataset=TinyData(),data_collator=collate)
    result=trainer.train()
    trainer.save_model(directory+'/final')
    base_optimizer=trainer.deepspeed.optimizer.optimizer
    assert type(base_optimizer).__name__=='AdamW'
    assert all(group['fused'] is True for group in base_optimizer.param_groups)
    assert trainer.deepspeed.zero_optimization_stage()==2
    assert not model.get_model().gradient_checkpointing
    from pathlib import Path
    checkpoint=Path(directory)/'checkpoint-2'
    assert list(checkpoint.glob('**/*optim_states.pt')), 'Missing ZeRO2 optimizer checkpoint'
    assert json.loads((checkpoint/'trainer_state.json').read_text())['global_step']==2
    handle.remove()
    params=list(model.get_model().pre_predictor.parameters())
    with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
        trained={k:v.detach().cpu().clone() for k,v in model.get_model().pre_predictor.state_dict().items()}
    assert any(not torch.equal(trained[k].float(),expected_init[k]) for k in trained), 'Predictor failed to update'
    assert len(calls)==2*accum, calls
    assert all(c['dtype']=='torch.bfloat16' and c['output_dtype']=='torch.bfloat16' for c in calls),calls
    assert any(c['shape'][0]==0 for c in calls) and any(c['shape'][0]>0 for c in calls),calls
    states=load_file(directory+'/final/model.safetensors')
    saved={k:v for k,v in states.items() if k.startswith('model.pre_predictor.')}
    assert len(saved)==4, list(saved)
    for k,v in trained.items(): torch.testing.assert_close(v,saved['model.pre_predictor.'+k],rtol=0,atol=0)
    unset_hf_deepspeed_config()
    restored=type(model).from_pretrained(directory+'/final',local_files_only=True,torch_dtype=torch.bfloat16)
    for k,v in trained.items(): torch.testing.assert_close(v,restored.get_model().pre_predictor.state_dict()[k],rtol=0,atol=0)
    report={'accumulation':accum,'steps':trainer.state.global_step,'calls':calls,'saved_predictor_keys':list(saved),'predictor_updated':True,'predictor_exact_reload':True,'last_text_only_pre_loss':float(model.last_pre_loss),'last_num_patches':int(model.last_pre_num_patches),'train_loss':result.training_loss,'zero_stage':trainer.deepspeed.zero_optimization_stage(),'optimizer':type(base_optimizer).__name__,'fused':True,'gradient_checkpointing':False,'loader_workers':2,'loader_prefetch':4,'non_blocking':trainer.accelerator.dataloader_config.non_blocking}
    assert report['last_text_only_pre_loss']==0 and report['last_num_patches']==0
    print('PRE_ZERO2_SMOKE_RESULT '+json.dumps(report),flush=True)
    reports.append(report)
    del trainer, model, restored, states, trained, params
    gc.collect(); torch.cuda.empty_cache()
print('ALL_PRE_ZERO2_SMOKES_PASSED '+json.dumps(reports),flush=True)
