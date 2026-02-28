# Fill this file out


from torch.fx.experimental.optimization import optimize_for_inference
import clip
import argparse
from cfg import cfg

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.indian_birds import IndianBirdsDataset
from tqdm import tqdm
from models.contrastive_loss import ContrastiveLoss

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")


def create_dataloaders(root_dir,
                       preprocess,
                       batch_size=32,
                       shuffle=True,
                       num_workers=4,
                       template="This is a photo of"):
    train_dataset = IndianBirdsDataset(
        root=root_dir+"/"+"train",
        tokenizer_fn=clip.tokenize,
        template=template,
        transform=preprocess
    )

    valid_dataset = IndianBirdsDataset(
        root=root_dir+"/"+"valid",
        tokenizer_fn=clip.tokenize,
        template=template,
        transform=preprocess
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, valid_loader


def load_model(model_name="ViT-B/16"):
    model, preprocess = clip.load(model_name, device=device, jit=False)
    # Convert model to single precision to prevent NaN loss when training with Adam
    # model = model.float()
    return model, preprocess


def train(train_loader, model, epochs=1):

    # For mixed precision training (fp16) adam's default epsilon is too small, and
    # quickly gets out of bound for the range of fp16. I could have either usef fp32
    # for training or increase the epsilon. Without it the model quickly turn to Nans.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, eps=10e-4)

    for epoch in range(epochs):
        tqdm.write(f"Epoch {epoch+1}/{epochs}")
        with tqdm(train_loader, desc="Training") as pbar:
            model.train()
            train_losses = []

            for images, class_labels in pbar:
                optimizer.zero_grad()
                images = images.to(device)
                class_labels = class_labels.to(device)

                i_embed, t_embed = model(images, class_labels)

                labels = torch.arange(i_embed.shape[0], device=i_embed.device)

                loss_I = F.cross_entropy(i_embed, labels)
                loss_T = F.cross_entropy(t_embed, labels)

                loss = (loss_I + loss_T)/2.0

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_epoch_loss = sum(train_losses)/len(train_losses)
            tqdm.write(
                f"Epoch {epoch+1}/{epochs} Train Loss: {train_epoch_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/vit_b16_exp.yaml")
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)

    model, preprocess = load_model(cfg.run_name)
    train_loader, valid_loader = create_dataloaders(
        cfg.dataset.root,
        preprocess,
        cfg.train.batch_size,
        cfg.train.shuffle,
        cfg.num_workers)

    train(train_loader, model, cfg.train.epochs)


if __name__ == "__main__":
    main()
