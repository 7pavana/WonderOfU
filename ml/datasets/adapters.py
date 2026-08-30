"""Official dataset adapters. Each expects an explicit project-owned manifest.csv.

Manifest schema: path,label,source_id,split.  Splits must group related frames by
source_id/video/document to avoid leakage; this code intentionally will not make
a random frame-level split.
"""
from .base import ManifestDatasetAdapter


class SiWAdapter(ManifestDatasetAdapter):
    name, task = "SiW", "spoof"


class FaceForensicsAdapter(ManifestDatasetAdapter):
    name, task = "FaceForensics++", "forgery"


class IDNetAdapter(ManifestDatasetAdapter):
    name, task = "IDNet", "document"


class DocumentRecaptureAdapter(ManifestDatasetAdapter):
    name, task = "TIFS 2022 Document Recapture", "document"


class DFDCAdapter(ManifestDatasetAdapter):
    name, task = "DFDC Preview", "forgery"
