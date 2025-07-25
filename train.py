import torch
from tqdm import tqdm
from pathlib import Path
import tiktoken

from config import cfg
from utils import get_weights_file_path, loader_loss, batch_loss, evaluate_model, generate_and_print_sample
from model import Model
from dataset import train_dataloader, val_dataloader

torch.manual_seed(28)



def train(cfg, model, train_loader, val_loader, eval_freq, eval_iter, start_context):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    print(f"Getting BPE tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Using gpt2 tokenizer...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step, initial_epoch = 0, -1, 0

    Path(cfg["weight_folder"]).mkdir(parents=True, exist_ok=True)

    if cfg["preload"]:
        weight_filename = get_weights_file_path(cfg, cfg["preload"])
        print(f"Preloading model -> {weight_filename}")

        state = torch.load(weight_filename)
        model.load_state_dict(state["model_state_dict"])

        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    
    else:
        print("No weights found...")

    for epoch in range(initial_epoch, cfg["num_epochs"]):
        batch_iterator = tqdm(train_loader, desc=f"Processing batch {epoch: 02d}")

        for input_batch, target_batch in batch_iterator:
            model.train()
            optimizer.zero_grad()
            loss = batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model=model, train_loader=train_loader, val_loader=val_loader, device=device, eval_iter=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                print(f"ep {epoch+1} (Step {global_step:06d}): "
                      f"train loss {train_loss:.3f}, val loss {val_loss:.3f}")
        generate_and_print_sample(model=model, tokenizer=tokenizer, device=device, start_context=start_context, cfg=cfg)

        weight_filename = get_weights_file_path(cfg, f"{epoch: 02d}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "global_step": global_step
        }, weight_filename)

        print("weights and optimizer saved.")

    return train_losses, val_losses, track_token_seen


model = Model(cfg)

train_losses, val_losses, token_seen = train(
   cfg=cfg, model=model, train_loader=train_dataloader, val_loader=val_dataloader,
   eval_freq=cfg['eval_freq'], eval_iter=cfg['eval_iter'], 
   start_context="Every effort moves you"
)

