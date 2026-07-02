"""
MedTrust-Net model definitions.
Shared between training notebooks and the FastAPI backend.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


PATHOLOGIES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion',
]
NUM_CLASSES = len(PATHOLOGIES)
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ConfidenceAwareAttentionBlock(nn.Module):
    """Stochastic spatial+channel attention with reparameterized Gaussian weights."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        self.channel_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_pool_max = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_channels // reduction, 8)
        self.channel_mlp_mu = nn.Sequential(
            nn.Linear(in_channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )
        self.channel_mlp_logvar = nn.Sequential(
            nn.Linear(in_channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )

        self.spatial_conv_mu = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_conv_logvar = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.last_spatial_mu = None
        self.last_spatial_logvar = None
        self.last_channel_mu = None
        self.last_channel_logvar = None

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        b, c, h, w = x.shape

        avg_c = self.channel_pool_avg(x).view(b, c)
        max_c = self.channel_pool_max(x).view(b, c)
        ch_in = torch.cat([avg_c, max_c], dim=1)
        ch_mu = self.channel_mlp_mu(ch_in)
        ch_logvar = torch.clamp(self.channel_mlp_logvar(ch_in), min=-10, max=2)
        ch_attn = torch.sigmoid(self.reparameterize(ch_mu, ch_logvar)).view(b, c, 1, 1)

        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s = torch.max(x, dim=1, keepdim=True)[0]
        sp_in = torch.cat([avg_s, max_s], dim=1)
        sp_mu = self.spatial_conv_mu(sp_in)
        sp_logvar = torch.clamp(self.spatial_conv_logvar(sp_in), min=-10, max=2)
        sp_attn = torch.sigmoid(self.reparameterize(sp_mu, sp_logvar))

        self.last_spatial_mu = sp_mu
        self.last_spatial_logvar = sp_logvar
        self.last_channel_mu = ch_mu
        self.last_channel_logvar = ch_logvar

        return x * ch_attn * sp_attn

    def kl_divergence(self):
        if self.last_spatial_mu is None:
            return torch.tensor(0.0)
        kl_sp = -0.5 * torch.mean(
            1 + self.last_spatial_logvar
              - self.last_spatial_mu.pow(2)
              - self.last_spatial_logvar.exp()
        )
        kl_ch = -0.5 * torch.mean(
            1 + self.last_channel_logvar
              - self.last_channel_mu.pow(2)
              - self.last_channel_logvar.exp()
        )
        return kl_sp + kl_ch


class MedTrustNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        bb = resnet50(weights=weights)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.cab1 = ConfidenceAwareAttentionBlock(256)
        self.cab2 = ConfidenceAwareAttentionBlock(512)
        self.cab3 = ConfidenceAwareAttentionBlock(1024)
        self.cab4 = ConfidenceAwareAttentionBlock(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, return_maps=False):
        x = self.stem(x)
        x = self.layer1(x); x = self.cab1(x)
        x = self.layer2(x); x = self.cab2(x)
        x = self.layer3(x); x = self.cab3(x)
        x = self.layer4(x); x = self.cab4(x)
        feat = self.avgpool(x).flatten(1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        if return_maps:
            dam = torch.sigmoid(self.cab4.last_spatial_mu)
            crm = torch.exp(0.5 * self.cab4.last_spatial_logvar)
            return logits, dam, crm
        return logits

    def total_kl_divergence(self):
        return (self.cab1.kl_divergence() + self.cab2.kl_divergence()
              + self.cab3.kl_divergence() + self.cab4.kl_divergence())


class PlainResNet50(nn.Module):
    """Baseline ResNet-50 (no CAB). Used for the side-by-side comparison panel."""

    def __init__(self, num_classes=5, pretrained=False, dropout=0.3):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        bb = resnet50(weights=weights)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); feat_map = self.layer4(x)
        feat = self.avgpool(feat_map).flatten(1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat_map
        return logits


def get_preprocess_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class GradCAM:
    """Grad-CAM on the last conv layer of the baseline ResNet-50."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx):
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        self.model.zero_grad()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear', align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
