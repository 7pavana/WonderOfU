"""Synthetic software validation only; never reports ML performance."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
    from ml.inference import InferenceService
    from ml.losses import masked_multitask_loss
    from ml.models import WonderOfUModel
except ImportError as exc:
    raise SystemExit(f"SMOKE TEST BLOCKED: missing dependency ({exc}). Install requirements.txt first.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WonderOfUModel().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
face, signal = torch.randn(2, 3, 160, 160, device=device), torch.randn(2, 1, 32, device=device)
face_out = model(face=face, rppg=signal)
face_labels = {task: torch.zeros(2, dtype=torch.long, device=device) for task in face_out}
face_masks = {"spoof": torch.tensor([True, False], device=device), "forgery": torch.tensor([False, True], device=device), "document": torch.tensor([False, False], device=device)}
face_loss = masked_multitask_loss(face_out, face_labels, face_masks)
document_out = model(document=torch.randn(2, 3, 160, 160, device=device))
document_labels = {task: torch.zeros(2, dtype=torch.long, device=device) for task in document_out}
document_masks = {"spoof": torch.tensor([False, False], device=device), "forgery": torch.tensor([False, False], device=device), "document": torch.tensor([True, True], device=device)}
document_loss = masked_multitask_loss(document_out, document_labels, document_masks)
loss = face_loss + document_loss; loss.backward(); optimizer.step()
with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "smoke.pt"; torch.save(model.state_dict(), checkpoint); model.load_state_dict(torch.load(checkpoint, weights_only=True)); assert checkpoint.exists()
unavailable = InferenceService().analyze(None, None)
assert unavailable["unified_assessment"]["status"] == "unavailable"
print(f"SMOKE TEST PASSED on {device}; loss={loss.item():.4f}. It validates all three task heads, masked losses, checkpoints, and the safe untrained-inference response; synthetic inputs validate software only.")
