from torch.utils.data import DataLoader
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
import pandas as pd
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import torch.nn.functional as F


class Affectnet(Dataset):
    def __init__(self, root, is_train, transform=None):
        image_path = os.path.join(root, "Manually_Annotated_Images")
        self.transform = transform

        if is_train:
            label_path = os.path.join(root, "training.csv")
        else:
            label_path = os.path.join(root, "validation.csv")

        list_label = pd.read_csv(label_path)
        valid_labels = list_label[list_label['expression'] < 8].copy()

        valid_labels['full_image_path'] = valid_labels['subDirectory_filePath'].apply(
            lambda x: os.path.join(image_path, x))

        valid_labels = valid_labels[valid_labels['full_image_path'].apply(os.path.isfile)]

        self.list_image = valid_labels['full_image_path'].tolist()
        self.list_label_expression = valid_labels['expression'].tolist()
        self.list_valence = valid_labels['valence'].tolist()
        self.list_arousal = valid_labels['arousal'].tolist()

    def __len__(self):
        return len(self.list_label_expression)

    def __getitem__(self, item):
        try:
            # print(self.list_image[item])
            image = Image.open(self.list_image[item]).convert("RGB")
            expr = self.list_label_expression[item]
            valence = self.list_valence[item]
            arousal = self.list_arousal[item]

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(expr), torch.tensor(valence, dtype=torch.float32), torch.tensor(arousal, dtype=torch.float32)

        except (FileNotFoundError, OSError) as e:
            print(f"Error loading image at {self.list_image[item]}: {e}")
            next_item = (item + 1) % len(self.list_image)
            return self.__getitem__(next_item)


def save_dataset_in_chunks(root, is_train=True, image_size=224, batch_size=1024, num_workers=0):
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5402, 0.4410, 0.3938], std=[0.2914, 0.2657, 0.2609]),
    ])

    dataset = Affectnet(root=root, is_train=is_train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    os.makedirs("chunks1024", exist_ok=True)
    prefix = "train" if is_train else "val"
    for idx, (imgs, exprs, vals, aros) in enumerate(tqdm(dataloader, desc=f"Saving {prefix} batches")):
        save_file = f"chunks1024/{prefix}_{idx:04d}.pt"
        torch.save({
            "images": imgs,
            "expressions": exprs,
            "valences": vals,
            "arousals": aros,
        }, save_file)

    print(f"✅ Saved {idx + 1} batches to folder 'chunks1024/' as {prefix}_*.pt")

class AffectnetPt(Dataset):
    def __init__(self, root, is_train, transform=None):
        root = "chunks1024"
        self.transform = transform
        if is_train:
            path = os.path.join(root, "train")
        else:
            path = os.path.join(root, "val")
        self.list_file = [os.path.join(path, x) for x in os.listdir(path)]
        self.data_index = []
        for file_path in self.list_file:
            data = torch.load(file_path, map_location="cpu")
            num_samples = data["images"].shape[0]
            for i in range(num_samples):
                self.data_index.append((file_path, i))

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, idx):
        file_path, inner_idx = self.data_index[idx]

        # ❌ Không dùng cache → load trực tiếp
        data = torch.load(file_path, map_location="cpu")
        img = data["images"][inner_idx]
        expr = data["expressions"][inner_idx]
        val = data["valences"][inner_idx]
        aro = data["arousals"][inner_idx]

        if self.transform:
            img = self.transform(img)
        img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        return img, expr, val, aro

def cout(a):
    print("****************************")
    print(a)
    print(type(a))
    print("****************************")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transform = Compose([
    #     Resize((224, 224)),
    #     ToTensor()
    # ])
    #
    # dataset = Affectnet(root="C:/Users/tam/Documents/data/Affectnet", is_train=True, transform=transform)
    # print(len(dataset))
    # img, expr, valence, arousal = dataset[0]
    # print("Batch:", img.shape)
    # print("Expressions:", expr)
    # print("Valence:", valence)
    # print("Arousal:", arousal)
    # # img = img.permute(1, 2, 0).numpy()  # (H, W, C)
    # # img = (img * 255).clip(0, 255).astype(np.uint8)
    # #
    # # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # # cv2.imshow(f"Expr:{expr.item()}, Val:{valence:.2f}, Aro:{arousal:.2f}", img_bgr)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # img_pil = to_pil_image(img)
    # img_pil.show(title="Affectnet")



    # root = "C:/Users/tam/Documents/data/Affectnet"
    #
    # save_dataset_in_chunks(
    #     root=root,
    #     is_train=True
    #     # save_path="affectnet_train.pt"
    # )
    #
    # save_dataset_in_chunks(
    #     root=root,
    #     is_train=False
    #     # save_path="affectnet_val.pt"
    # )

    transform = Normalize(mean=[0.5402, 0.4410, 0.3938], std=[0.2914, 0.2657, 0.2609])
    dataset = AffectnetPt(root="chunks1024", is_train=False, transform=transform)
    cout((dataset))
    # cout(len(dataset[1][3]))
