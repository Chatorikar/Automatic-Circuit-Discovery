from transformer_lens import HookedTransformer
import torch
from acdc.docstring.utils import AllDataThings
import itertools
from transformers import GPT2Tokenizer
import random
import json
from acdc.acdc_utils import negative_log_probs
from functools import partial


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def token_count(text):
    return len(tokenizer.encode(text))


def get_iterate_model(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HookedTransformer.from_pretrained("gpt-neo-125M", device=device)
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    return model

def generate_samples(num_samples):
    
    # Generate potential values for B
    characters = "abcdefghijklmnopqrstuvwxyz"
    valid_Bs = []

    # Generate values for B (exactly 1 token)
    for i in range(2, 4):  # Try lengths from 2 to 3 characters
        for combo in itertools.product(characters, repeat=i):
            B_candidate = "".join(combo)
            if token_count(B_candidate) == 1:
                valid_Bs.append(B_candidate)
            if len(valid_Bs) >= num_samples:
                break
        if len(valid_Bs) >= num_samples:
            break

    # Generate function definitions and save to file
    return [f'def iterate({B}):\n\t' for B in valid_Bs]

def generate_corrupted_samples(num_samples):
    # Generate potential values for A and B
    characters = "abcdefghijklmnopqrstuvwxyz"

    # Create lists to store 1000 valid values for A and B
    valid_As = []
    valid_Bs = []

    # Generate values for A (exactly 2 tokens)
    for i in range(2, 6):  # Try lengths from 2 to 5 characters
        for combo in itertools.product(characters, repeat=i):
            A_candidate = "".join(combo)
            if token_count(A_candidate) == 2:
                valid_As.append(A_candidate)
            if len(valid_As) >= num_samples:
                break
        if len(valid_As) >= num_samples:
            break

    # Generate values for B (exactly 1 token)
    for i in range(2, 4):  # Try lengths from 2 to 3 characters
        for combo in itertools.product(characters, repeat=i):
            B_candidate = "".join(combo)
            if token_count(B_candidate) == 1:
                valid_Bs.append(B_candidate)
            if len(valid_Bs) >= num_samples:
                break
        if len(valid_Bs) >= num_samples:
            break

    # Generate function definitions by pairing A and B
    samples = []
    i = 0
    while i < num_samples:
        A = random.choice(valid_As)
        B = random.choice(valid_Bs)
        sample = f'def {A}({B}):\n\t'
        if token_count(sample) == 8:
            samples.append(sample)
            i += 1

    return samples


KEYWORD = 'for'


def get_all_iterate_things(num_samples: int, device):
    random.seed(0)
    tl_model = get_iterate_model(device)
    data = generate_samples(num_samples * 2)
    patch_data = generate_corrupted_samples(num_samples * 2)
    data = torch.tensor(tokenizer([tokenizer.bos_token + d for d in data]).input_ids).to(device) # num_samples * context length
    p_d = [tokenizer.bos_token + d for d in patch_data]
    
    # for p in patch_data:
    #     if token_count(p) > 8:
    #         print('Pre Padding:',p, token_count(p))
    # for p in p_d:
    #     if token_count(p) > 9:
    #         print('Post Padding:',p, token_count(p))
    patch_data = torch.tensor(tokenizer(p_d).input_ids).to(device) # num_samples * context length
    
    val_data = data[:num_samples,:]
    test_data = data[num_samples:,:]
    val_patch_data = patch_data[:num_samples,:]
    test_patch_data = patch_data[num_samples:,:]

    # the correct label for each prompt is "for"
    '''
    def iterate(arr):
        [for]
    '''
    labels = torch.tensor(tokenizer.encode(KEYWORD) * data.shape[0]).to(device)
    val_labels = labels[:num_samples]
    test_labels = labels[num_samples:]
    #return data, patch_data

    val_metric = partial(
        negative_log_probs,
        labels = val_labels,
        last_seq_element_only=True
    )

    test_metric = partial(
        negative_log_probs,
        labels = test_labels,
        last_seq_element_only=True
    )

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=val_metric,
        validation_data=val_data,
        validation_labels=val_labels,
        validation_mask=None,
        validation_patch_data=val_patch_data,
        test_metrics=test_metric,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        test_patch_data=test_patch_data
    )




    # return AllDataThings(
    #     tl_model=tl_model,
    #     validation_metric=partial(validation_metric, correct=correct_answers),
    #     validation_data=data,
    #     validation_labels=None,
    #     validation_mask=None,
    #     validation_patch_data=data.clone(), # We're doing zero ablation so irrelevant
    #     test_metrics=test_metrics,
    #     test_data=data,
    #     test_labels=None,
    #     test_mask=None,
    #     test_patch_data=data.clone(),
    # )



if __name__ == '__main__':
    things = get_all_iterate_things(1000, 'cuda')
    
