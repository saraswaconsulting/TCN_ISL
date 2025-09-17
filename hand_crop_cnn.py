import cv2, numpy as np, torch, torch.nn as nn
from torchvision import models, transforms

def _bbox_from_landmarks(frame, landmarks, margin=0.2):
    if landmarks is None: 
        return None
    h, w = frame.shape[:2]
    xs = [int(lm.x * w) for lm in landmarks.landmark]
    ys = [int(lm.y * h) for lm in landmarks.landmark]
    x0, x1 = max(0, min(xs)), min(w-1, max(xs))
    y0, y1 = max(0, min(ys)), min(h-1, max(ys))
    # add margin
    dx, dy = int((x1-x0)*margin), int((y1-y0)*margin)
    x0, x1 = max(0, x0-dx), min(w-1, x1+dx)
    y0, y1 = max(0, y0-dy), min(h-1, y1+dy)
    if x1<=x0 or y1<=y0: return None
    return x0,y0,x1,y1

class HandCropEncoder(nn.Module):
    def __init__(self, embed_dim=128, freeze=True):
        super().__init__()
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_ch = 576  # last channel of mobilenet_v3_small features
        self.head = nn.Sequential(
            nn.Linear(in_ch, embed_dim),
            nn.ReLU(inplace=True)
        )
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad=False
        self.tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def forward(self, crops):
        # crops: list of HxWx3 np.uint8 images
        if len(crops)==0:
            # return zero embedding
            return torch.zeros(1, self.head[0].out_features, device=next(self.parameters()).device)
        xs = [self.tx(c) for c in crops]
        x = torch.stack(xs).to(next(self.parameters()).device)
        h = self.backbone(x)
        h = self.pool(h).flatten(1)
        z = self.head(h)
        # average embeddings over provided crops
        z = z.mean(dim=0, keepdim=True)
        return z
