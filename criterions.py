import os
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
from torchvision.models import resnet50
from dataset import Affectnet
from torch.utils.data import Dataset, DataLoader

class PerceptualWaveletLoss(nn.Module):
    """Perceptual loss for wavelet domain with separate handling of LL and HF bands"""
    def __init__(self):
        super().__init__()
        # Use VGG16 for perceptual loss
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ])
        
        for param in self.vgg_blocks.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, generated, target, lambda_ll=0.3, lambda_hi=1.0):
        """
        Args:
            generated: Generated image in [-1, 1]
            target: Target image in [-1, 1]
            lambda_ll: Weight for perceptual loss on LL band
            lambda_hi: Weight for L1 loss on HF bands
        """
        from model import DWT
        dwt = DWT().to(generated.device)
        
        # Get wavelet decomposition
        gen_wavelet = dwt(generated)
        target_wavelet = dwt(target)
        
        # Split into LL and HF bands
        C = generated.shape[1]
        gen_ll = gen_wavelet[:, :C, :, :]
        target_ll = target_wavelet[:, :C, :, :]
        
        gen_hi = gen_wavelet[:, C:, :, :]
        target_hi = target_wavelet[:, C:, :, :]
        
        # Normalize LL band to [0, 1] for VGG
        gen_ll_norm = (gen_ll + 1) / 2
        target_ll_norm = (target_ll + 1) / 2
        
        # Resize LL to 224x224 for VGG
        gen_ll_resized = F.interpolate(gen_ll_norm, size=(224, 224), mode='bilinear')
        target_ll_resized = F.interpolate(target_ll_norm, size=(224, 224), mode='bilinear')
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)
        gen_ll_norm = (gen_ll_resized - mean) / std
        target_ll_norm = (target_ll_resized - mean) / std
        
        # Compute perceptual loss on LL band
        perceptual_loss = 0.0
        gen_feats = gen_ll_norm
        target_feats = target_ll_norm
        
        for block in self.vgg_blocks:
            gen_feats = block(gen_feats)
            target_feats = block(target_feats)
            perceptual_loss += F.l1_loss(gen_feats, target_feats)
        
        # L1 loss on HF bands
        hi_loss = F.l1_loss(gen_hi, target_hi)
        
        # Combine losses
        total_loss = lambda_ll * perceptual_loss + lambda_hi * hi_loss
        
        return total_loss


class AdaptiveLossWeighter(nn.Module):
    """Adaptive loss weighting to balance identity preservation vs emotion expression"""
    def __init__(self, initial_weights, warmup_steps=5000):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.initial_weights = initial_weights

        self.loss_ema = {}
        self.alpha = 0.99

    def update_weights(self, losses):
        """Update loss weights based on relative loss magnitudes"""
        self.step_count += 1

        for name, loss in losses.items():
            if name not in self.loss_ema:
                self.loss_ema[name] = loss.item()
            else:
                self.loss_ema[name] = self.alpha * self.loss_ema[name] + (1 - self.alpha) * loss.item()

        if self.step_count < self.warmup_steps:
            if 'aux_expr' in self.loss_ema and self.loss_ema['aux_expr'] < 0.05:
                warmup_progress = self.step_count / self.warmup_steps
                id_scale = max(0.05, 1.0 - warmup_progress * 0.9)
                expr_scale = min(5.0, 1.0 + warmup_progress * 4.0)
                return {
                    'lambda_id': self.initial_weights['lambda_id'] * id_scale,
                    'lambda_aux_expr': self.initial_weights['lambda_aux_expr'] * expr_scale
                }
            elif 'aux_expr' in self.loss_ema and self.loss_ema['aux_expr'] < 0.5:
                warmup_progress = self.step_count / self.warmup_steps
                expr_scale = 1.0 + warmup_progress * 2.0
                return {
                    'lambda_aux_expr': self.initial_weights['lambda_aux_expr'] * expr_scale
                }

        return {}

class DAN(nn.Module):
    def __init__(self, num_class=7, num_head=4, pretrained=True):
        super(DAN, self).__init__()

        resnet = models.resnet18(pretrained)

        if pretrained:
            checkpoint = torch.load('affecnet7_epoch6_acc0.6569.pth')
            resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, "cat_head%d" % i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))

        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)

        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)

        return out, x, heads


class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y

        return out


class ChannelAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()
        )

    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        out = sa * y

        return out


class Emotion_model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']

        self.model = DAN(num_head=4, num_class=7, pretrained=False)
        checkpoint = torch.load('affecnet7_epoch6_acc0.6569.pth',
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)

        return faces

    def fer(self, path):
        img0 = Image.open(path).convert('RGB')

        faces = self.detect(img0)

        if len(faces) == 0:
            return 'null'

        x, y, w, h = faces[0]

        img = img0.crop((x, y, x + w, y + h))

        img = self.data_transforms(img)
        img = img.view(1, 3, 224, 224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out, 1)
            index = int(pred)
            label = self.labels[index]

            return label


if __name__ == "__main__":
    model = Emotion_model()
    batch_size = 4
    # transform = Compose([
    #     Resize((image_size, image_size)),
    #     ToTensor(),
    #     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])
    # image = r"C:\Users\tam\Documents\data\FEG\Manually_Annotated_Images\Manually_Annotated_Images\12\4e73e4cfde7ded87a2d6274a5aa52aef4fac3a81c67f5fa169cf0c3e.jpeg"
    # assert os.path.exists(image), "Failed to load image file."
    # label = model.fer(image)
    # val_dataset = Affectnet(root="C:/Users/tam/Documents/data/FEG", is_train=False, transform=transform)
    # val_dataloader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=batch_size,
    #     num_workers=4,
    #     shuffle=False,
    #     drop_last=True
    # )
    # print(f'emotion label: {label}')
