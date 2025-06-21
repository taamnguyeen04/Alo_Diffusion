import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose
import pandas as pd
from torchvision.transforms.functional import to_pil_image

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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    dataset = Affectnet(root="C:/Users/tam/Documents/data/Affectnet", is_train=True, transform=transform)
    img, expr, valence, arousal = dataset[0]
    print("Batch:", img.shape)
    print("Expressions:", expr)
    print("Valence:", valence)
    print("Arousal:", arousal)
    # img = img.permute(1, 2, 0).numpy()  # (H, W, C)
    # img = (img * 255).clip(0, 255).astype(np.uint8)
    #
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imshow(f"Expr:{expr.item()}, Val:{valence:.2f}, Aro:{arousal:.2f}", img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_pil = to_pil_image(img)
    img_pil.show(title="Affectnet")
