from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="sdpa") # sdpa or flash_attention_2
