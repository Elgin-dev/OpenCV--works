import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DeblurDataset
from models import UNet

def psnr(pred, target):
    mse = ((pred.clamp(0, 1) - target) ** 2).mean()
    return 10 * torch.log10(1 / mse)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    train_dl = DataLoader(DeblurDataset("data/train", train=True), batch_size=16,
                          shuffle=True, num_workers=0)
    val_dl = DataLoader(DeblurDataset("data/val", train=False), batch_size=16, num_workers=0)

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.L1Loss()
    best = 0

    for epoch in range(20):
        model.train()
        total = 0
        for blur, sharp in train_dl:
            blur, sharp = blur.to(device), sharp.to(device)
            loss = loss_fn(model(blur), sharp)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        model.eval()
        scores, base_scores = [], []
        with torch.no_grad():
            for blur, sharp in val_dl:
                blur, sharp = blur.to(device), sharp.to(device)
                scores.append(psnr(model(blur), sharp).item())
                base_scores.append(psnr(blur, sharp).item())
        val = sum(scores) / len(scores)
        base = sum(base_scores) / len(base_scores)
        print(f"Epoch {epoch+1}: loss {total/len(train_dl):.4f} | "
              f"val PSNR {val:.2f} dB (blurred input: {base:.2f} dB)")

        if val > best:
            best = val
            torch.save(model.state_dict(), "unet_best.pt")