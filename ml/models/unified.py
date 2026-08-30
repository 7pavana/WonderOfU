"""Unified partial-label identity-fraud architecture."""
from __future__ import annotations
try:
    import torch
    from torch import Tensor, nn
except ImportError:  # imports remain informative before ML dependencies are installed
    torch = None
    Tensor = object
    nn = None


def _torch_required() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required. Install requirements.txt with a Python version supported by PyTorch.")


if nn:
    class ImageEncoder(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.network = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim))
        def forward(self, image: Tensor) -> Tensor: return self.network(image)

    class RPPGEncoder(nn.Module):
        def __init__(self, dim: int):
            super().__init__(); self.network = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, dim))
        def forward(self, signal: Tensor) -> Tensor: return self.network(signal)

    class WonderOfUModel(nn.Module):
        def __init__(self, embedding_dim: int = 128, shared_dim: int = 128, use_rppg: bool = True):
            super().__init__(); self.use_rppg = use_rppg
            self.visual_encoder, self.document_encoder = ImageEncoder(embedding_dim), ImageEncoder(embedding_dim)
            self.rppg_encoder = RPPGEncoder(embedding_dim) if use_rppg else None
            face_fusion_in = embedding_dim * (2 if use_rppg else 1)
            # Modality-specific projection avoids inventing a second document modality
            # merely to match the visual+rPPG fusion width.
            self.face_projection = nn.Linear(face_fusion_in, shared_dim)
            self.document_projection = nn.Linear(embedding_dim, shared_dim)
            self.shared_representation = nn.Sequential(nn.Linear(shared_dim, shared_dim), nn.ReLU(), nn.Dropout(0.1))
            self.spoof_head, self.forgery_head, self.document_head = nn.Linear(shared_dim, 2), nn.Linear(shared_dim, 2), nn.Linear(shared_dim, 2)
        def forward(self, face: Tensor | None = None, rppg: Tensor | None = None, document: Tensor | None = None) -> dict[str, Tensor]:
            if face is None and document is None: raise ValueError("A face or document input is required.")
            if face is not None:
                visual = self.visual_encoder(face)
                features = [visual]
                if self.use_rppg:
                    if rppg is None: raise ValueError("rPPG signal required when use_rppg=True.")
                    features.append(self.rppg_encoder(rppg))
                face_shared = self.shared_representation(self.face_projection(torch.cat(features, dim=1)))
            else: face_shared = None
            doc_shared = self.shared_representation(self.document_projection(self.document_encoder(document))) if document is not None else None
            return {"spoof": self.spoof_head(face_shared) if face_shared is not None else None, "forgery": self.forgery_head(face_shared) if face_shared is not None else None, "document": self.document_head(doc_shared) if doc_shared is not None else None}
else:
    ImageEncoder = RPPGEncoder = None
    class WonderOfUModel:
        def __init__(self, *args, **kwargs): _torch_required()
