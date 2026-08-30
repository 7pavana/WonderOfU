"""Independent baselines deliberately bypass the unified shared representation."""
from .unified import ImageEncoder, RPPGEncoder, nn, torch, _torch_required

if nn:
    class _ImageBaseline(nn.Module):
        def __init__(self, dim=128): super().__init__(); self.encoder = ImageEncoder(dim); self.head = nn.Linear(dim, 2)
        def forward(self, image): return self.head(self.encoder(image))
    class SpoofBaseline(_ImageBaseline): pass
    class ForgeryBaseline(_ImageBaseline): pass
    class DocumentBaseline(_ImageBaseline): pass
else:
    class SpoofBaseline:
        def __init__(self, *args, **kwargs): _torch_required()
    ForgeryBaseline = DocumentBaseline = SpoofBaseline
