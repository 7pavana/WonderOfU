import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.config import dataset_path, load_config
from ml.datasets import DFDCAdapter, DocumentRecaptureAdapter, FaceForensicsAdapter, IDNetAdapter, SiWAdapter

ADAPTERS = {"siw": SiWAdapter, "faceforensics": FaceForensicsAdapter, "idnet": IDNetAdapter, "document_recapture": DocumentRecaptureAdapter, "dfdc": DFDCAdapter}
if __name__ == "__main__":
    config = load_config()
    for name, adapter_cls in ADAPTERS.items(): print(f"{name}: {adapter_cls(dataset_path(config, name)).availability_message()}")
