import torch
import torch.nn as nn
import wandb
from tqdm.auto import tqdm


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, captions, true_len in tqdm(loader, desc="Training"):
        images = images.to(device)
        captions = captions.to(device)
        true_len = true_len.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, captions, true_len)
        # outputs: (batch, seq_len, vocab_size)
        # captions: (batch, seq_len)

        # Calculem la loss:
        # outputs els aplanem a (batch*seq_len, vocab_size)
        # captions objectiu: saltem el <SOS> inicial → captions[:, 1:]

        batch_size, seq_len, vocab_size = outputs.shape
        target = captions[:, 1:seq_len+1]  # mateixa longitud que outputs
        loss = criterion(
            outputs.reshape(-1, vocab_size),
            target.reshape(-1)
        )

        # Backward pass
        loss.backward()

        # Gradient clipping (evita exploding gradients a la RNN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions, true_len in tqdm(loader, desc="Validation"):
            images = images.to(device)
            captions = captions.to(device)
            true_len = true_len.to(device)

            outputs = model(images, captions, true_len)
            batch_size, seq_len, vocab_size = outputs.shape
            target = captions[:, 1:seq_len+1]  # mateixa longitud que outputs
            loss = criterion(
                outputs.reshape(-1, vocab_size),
                target.reshape(-1)
            )
            total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, optimizer, criterion,
          num_epochs, device, idx2char):

    wandb.watch(model, criterion, log="all", log_freq=50)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer,
                                  criterion, device)
        val_loss = val_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Guardar el millor model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  → Millor model guardat!")

        # Log a WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # Cada 5 epochs, genera un exemple per veure com va
        if (epoch + 1) % 5 == 0:
            sample = next(iter(val_loader))[0]
            sample_img = sample[0]
            predicted = model.generate(sample_img, idx2char, device=device)
            print(f"\n  Exemple generat {model.max_len}:")
            print(f"  {predicted}")
            # print(f"  {sample[1]}")
