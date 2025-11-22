from utils import *
from model import *
from loss import *
from mapping import *



class ForgeryDataset(Dataset):
    def __init__(self, image_files, mask_dir, augment=False, image_size=512):
        self.image_files = image_files
        self.mask_dir = mask_dir
        self.augment = augment
        self.image_size = image_size
        
        self.tf_img = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.tf_img(img) 

        filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, filename + ".npy")

        mask = np.load(mask_path).astype(np.float32)

        if mask.ndim == 3:
            if mask.shape[0] > 1:
                mask = np.any(mask, axis=0).astype(np.float32)  
            else:
                mask = mask.squeeze(0)

        elif mask.ndim == 2:
            pass  

        else:
            raise ValueError(f"Shape {mask.shape} in {mask_path}")

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  
        mask = TF.resize(mask, [self.image_size, self.image_size],
                        interpolation=TF.InterpolationMode.NEAREST)

        return img, mask




def dice_coef(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + 1) / (union + 1)

def iou(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = target.sum() + pred.sum() - inter
    return (inter + 1) / (union + 1)


DATASET_ROOT = "."

images_auth = sorted(glob(f"{DATASET_ROOT}/train_images/authentic/*.png"))
images_forg = sorted(glob(f"{DATASET_ROOT}/train_images/forged/*.png"))

all_images = images_auth + images_forg
random.shuffle(all_images)

train_img, val_img = train_test_split(all_images, test_size=0.15, random_state=42)

train_ds = ForgeryDataset(train_img, f"{DATASET_ROOT}/train_masks", augment=True, image_size=256)
val_ds   = ForgeryDataset(val_img, f"{DATASET_ROOT}/train_masks", image_size=256)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)



device = "cuda" if torch.cuda.is_available() else "cpu"

model = CmfdInstanceModel(num_queries=5).to(device)
criterion = CmfdLoss()
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


EPOCHS = 25
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]")
    for img, mask_gt in pbar:
        img, mask_gt = img.to(device), mask_gt.to(device)

        out = model(img)
        loss, _ = criterion(out, mask_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()

    model.eval()
    val_dice = 0
    val_iou = 0

    with torch.no_grad():
        for img, mask_gt in val_loader:
            img, mask_gt = img.to(device), mask_gt.to(device)
            out = model(img)
            mask_pred = instance_to_foreground(out)

            val_dice += dice_coef(mask_pred, mask_gt).item()
            val_iou  += iou(mask_pred, mask_gt).item()

    val_dice /= len(val_loader)
    val_iou  /= len(val_loader)

    print(f"\nEpoch {epoch} | Train Loss={train_loss/len(train_loader):.4f} | "
          f"Val Dice={val_dice:.4f} | Val IoU={val_iou:.4f}")

    torch.save(model.state_dict(), f"{SAVE_DIR}/cmfd_epoch_{epoch}.pth")
