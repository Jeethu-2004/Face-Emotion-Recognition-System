"""
Realtime FER with Grad-CAM overlays
Save as: realtime_fer_gradcam.py
Run: python realtime_fer_gradcam.py

Notes:
 - Update WEIGHTS path if needed.
 - This script auto-finds the last Conv2d layer for Grad-CAM.
 - For best fps use a GPU. On CPU it will run slower but works.
"""

import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# --------- Config ----------
WEIGHTS = r"D:\FRP_project\simple_fer_from_csv.pth"   # <-- update if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAM_INDEX = 0                 # camera index
CONF_THRESHOLD = 0.20         # min confidence to display label
EMOTIONS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ---------------------------

# --- Model (must match training) ---
class SimpleFER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- Load model weights ---
model = SimpleFER(num_classes=len(EMOTIONS)).to(DEVICE)
state = torch.load(WEIGHTS, map_location=DEVICE)
# handle a few checkpoint formats
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
elif isinstance(state, dict) and any(isinstance(v, dict) for v in state.values()):
    # some checkpoints store model under a nested dict
    # try common keys, otherwise assume it's already a state_dict
    found = False
    for k in ["model_state_dict", "state_dict", "model"]:
        if k in state:
            model.load_state_dict(state[k])
            found = True
            break
    if not found:
        try:
            model.load_state_dict(state)
        except Exception:
            # last resort: try first value that looks like a state_dict
            first = next(iter(state.values()))
            model.load_state_dict(first)
else:
    model.load_state_dict(state)
model.eval()

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.CenterCrop(48),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Helper: find last Conv2d layer in model.features ---
def find_last_conv(module):
    last_conv = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

target_conv = find_last_conv(model.features)
if target_conv is None:
    raise RuntimeError("Could not find a Conv2d layer for Grad-CAM target.")

# We'll capture forward activations and backward gradients on the target conv
activations = None
grads = None

def forward_hook(module, inp, out):
    global activations
    activations = out.detach().cpu()   # shape [B, C, H, W]

def backward_hook(module, grad_in, grad_out):
    global grads
    # grad_out is a tuple; take first
    grads = grad_out[0].detach().cpu()  # shape [B, C, H, W]

# register hooks
fwd_handle = target_conv.register_forward_hook(forward_hook)
bwd_handle = target_conv.register_full_backward_hook(backward_hook)  # use full hook for robustness

# --- Face detector: Haar cascade ---
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)
if face_cascade.empty():
    raise RuntimeError("Haar cascade failed to load.")

# --- Camera ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}.")

print("Press 'q' to quit, 's' to save snapshot. Grad-CAM overlay is shown on faces.")

with torch.no_grad():  # we'll still compute grads manually using requires_grad when needed
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48))

        # collect face crops and positions
        face_rois = []
        positions = []
        for (x, y, w, h) in faces:
            pad = int(0.1 * w)
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad); y2 = min(frame.shape[0], y + h + pad)
            roi = gray[y1:y2, x1:x2]
            # convert to PIL, resize 48x48
            pil = Image.fromarray(roi).resize((48,48), Image.BILINEAR)
            tensor = preprocess(pil)  # [1,48,48]
            face_rois.append(tensor)
            positions.append((x1,y1,x2,y2))

        if len(face_rois) > 0:
            batch = torch.stack(face_rois, dim=0).to(DEVICE)  # shape [B,1,48,48]
            batch.requires_grad_(True)  # allow gradients for Grad-CAM
            # forward
            out = model(batch)  # logits [B, C]
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            # compute Grad-CAM for each sample in batch
            # To get gradients we must perform backward on each predicted class score.
            # We'll do a loop per sample (batch sizes are small: faces per frame typically 1-3).
            cams = []
            for i in range(batch.shape[0]):
                model.zero_grad()
                # clear previous hooks' saved activations/grads
                # activations and grads are filled by hooks when forward/backward through target conv
                # forward already ran; activations holds last forward output for the whole batch.
                # We need gradient w.r.t. target class score for sample i.
                score = out[i, preds[i]]
                # backward for this single score
                score.backward(retain_graph=True)

                # grads and activations captured on CPU by hooks
                if activations is None or grads is None:
                    # fallback: compute activation by running features manually (rare)
                    act = model.features(batch).detach().cpu()
                    # approximate grads via autograd is not possible without backward; use zeros
                    g = torch.zeros_like(act)
                else:
                    act = activations[i]   # [C,H,W]
                    g = grads[i]           # [C,H,W]

                # global-average-pool the gradients to get weights
                weights = g.mean(dim=(1,2), keepdim=True)  # [C,1,1]
                # weighted combination of activations
                cam = (weights * act).sum(dim=0, keepdim=False)  # [H,W]
                cam = torch.relu(cam)
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()
                cam_np = cam.numpy()  # HxW in [0,1]
                cams.append(cam_np)

                # zero gradients for next iteration
                model.zero_grad()
                batch.grad = None

            # draw each face with overlay
            for idx, (x1,y1,x2,y2) in enumerate(positions):
                prob = float(probs[idx, preds[idx]])
                label = f"{EMOTIONS[preds[idx]]}: {prob:.2f}"
                color = (0,255,0) if prob >= CONF_THRESHOLD else (0,165,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                # prepare heatmap: resize cam to face box size
                cam = cams[idx]
                cam_resized = cv2.resize(cam, (x2 - x1, y2 - y1))
                heatmap = np.uint8(255 * cam_resized)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR

                # overlay heatmap on the corresponding region of the frame
                overlay = frame[y1:y2, x1:x2].astype(np.float32) / 255.0
                heat = heatmap.astype(np.float32) / 255.0
                alpha = 0.5  # heatmap transparency
                blended = cv2.addWeighted(overlay, 1 - alpha, heat, alpha, 0)
                frame[y1:y2, x1:x2] = np.uint8(blended * 255)

                # draw label background and text (top-left of box)
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, 0.6, (0,0,0), 1, cv2.LINE_AA)

        # Show frame
        cv2.imshow("FER Grad-CAM - press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            ts = int(time.time())
            fname = f"snapshot_{ts}.png"
            cv2.imwrite(fname, frame)
            print("Saved", fname)

# cleanup
cap.release()
cv2.destroyAllWindows()
fwd_handle.remove()
bwd_handle.remove()
