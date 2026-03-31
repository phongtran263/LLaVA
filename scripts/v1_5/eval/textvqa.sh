#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/llava-v1.5-3b-lora-mul-merged \
    --question-file ./playground/data/eval/text_vqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-3b-lora-mul-merged.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/text_vqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-3b-lora-mul-merged.jsonl
