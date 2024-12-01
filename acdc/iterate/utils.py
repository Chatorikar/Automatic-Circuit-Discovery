from transformer_lens import HookedTransformer
import torch

def get_iterate_model(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HookedTransformer.from_pretrained("gpt-neo-125M", device=device)
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    return model

