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
from sklearn.metrics import confusion_matrix

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")


def create_val_dataloader(root_dir,
                          preprocess,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          template="This is a photo of"):
    valid_dataset = IndianBirdsDataset(
        root=root_dir+"/"+"valid",
        tokenizer_fn=clip.tokenize,
        template=template,
        transform=preprocess
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return valid_loader


def load_model(model_name="ViT-B/16"):
    model, preprocess = clip.load(model_name, device=device, jit=False)
    # Convert model to single precision to prevent NaN loss when training with Adam
    # model = model.float()
    return model, preprocess


def evaluate(val_loader, model):

    model.eval()
    true_labels = []
    pred_labels = []
    with tqdm(val_loader, desc="Validation") as pbar:
        accuracy = 0
        for images, class_labels in pbar:
            images = images.to(device)
            class_labels = class_labels.to(device)

            i_embed, _ = model(images, class_labels)

            labels = torch.arange(images.shape[0], device=device)
            probs = F.softmax(i_embed, dim=-1)
            predictions = torch.argmax(probs, dim=-1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

            accuracy += torch.eq(predictions, labels).sum().item()

        accuracy = accuracy/len(val_loader.dataset)
        print(f"Average Accuracy: {accuracy}")

        conf_matrix = confusion_matrix(true_labels, pred_labels)
        print(f"Confusion Matrix: {conf_matrix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/vit_b16_exp.yaml")
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)

    model, preprocess = load_model(cfg.run_name)
    valid_loader = create_val_dataloader(
        cfg.dataset.root,
        preprocess,
        cfg.train.batch_size,
        cfg.train.shuffle,
        cfg.num_workers)

    evaluate(valid_loader, model)


if __name__ == "__main__":
    main()
