import torch
import torch.nn as nn
from pathlib import Path

def batch_loss(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def loader_loss(dataloader, model, device, num_batch=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float('nan')
    elif num_batch is None:
        num_batch = len(dataloader)
    else:
        num_batch = min(num_batch, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batch:
            loss = batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batch

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = loader_loss(train_loader, model, device, num_batch=eval_iter)
        val_loss = loader_loss(val_loader, model, device, num_batch=eval_iter)
    model.train()
    return train_loss, val_loss

def text_to_token_ids(txt, tokenizer):
    token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)
    return token_ids_tensor

def token_ids_to_text(ids, tokenizer):
    text = ids.squeeze(0)
    return tokenizer.decode(text.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context, cfg):
    model.eval()
    context_size = model.pos_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model=model, idx=encoded, max_new_tokens=cfg["max_new_tokens"], context_size=context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
        model.train()


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature

            probas = torch.softmax(logits, dim=-1)   # (batch, context_length)

            idx_next = torch.multinomial(probas, num_samples=1)  # (batch_size, 1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # if idx_next == eos_id:
        #     break
        if eos_id is not None and (idx_next == eos_id).any():
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def get_weights_file_path(cfg, epoch: str):
    weight_folder = cfg["weight_folder"]
    weight_basename = cfg["weight_basename"]
    weight_filename = f"{weight_basename}{epoch}.pt"
    return str(Path(".")/weight_folder/weight_filename)