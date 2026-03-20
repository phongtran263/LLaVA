import torch
from llava.model.builder import load_pretrained_model
import argparse
import os

def main(args):
    print("🔹 Loading base LLaVA model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.base_model,
        None,
        model_name="llava-v1.5-7b"
    )

    print("🔹 Loading Stage-1 checkpoint...")
    ckpt_path = os.path.join(args.pretrain_ckpt, "mm_projector.bin")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("🔹 Extracting mm_projector weights...")
    mm_projector_state = {}
    for k, v in ckpt.items():
        if "mm_projector" in k:
            new_k = k.replace("model.mm_projector.", "")
            mm_projector_state[new_k] = v

    print(f"Found {len(mm_projector_state)} projector params")

    print("🔹 Loading into model...")
    model.model.mm_projector.load_state_dict(mm_projector_state, strict=True)

    if args.load_vision:
        print("🔹 Loading vision tower weights...")
        vision_state = {}
        for k, v in ckpt.items():
            if "vision_tower" in k:
                new_k = k.replace("model.vision_tower.", "")
                vision_state[new_k] = v

        model.model.vision_tower.load_state_dict(vision_state, strict=False)

    print("🔹 Saving new model...")
    os.makedirs(args.output_dir, exist_ok=True)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("✅ Done! Saved to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./llava-custom")
    parser.add_argument("--load_vision", action="store_true")

    args = parser.parse_args()
    main(args)