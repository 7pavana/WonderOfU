# WonderOfU

WonderOfU is a research foundation for **identity-fraud/authenticity detection**: face presentation attacks, face forgery, and document manipulation. It does not claim legal ownership or face-to-document identity matching.

## Architecture

Face video produces visual and CHROM rPPG embeddings; documents produce a document embedding. A fusion layer builds a shared representation for spoofing, forgery, and document heads. Independent baselines are included for the planned ablation comparison. Training uses task masks, so each sample activates only its known label.

## Setup and current state

Use a Python version supported by the selected PyTorch build, then install dependencies: `python -m pip install -r requirements.txt`. Configure the external data root through `WONDEROFU_DATA_ROOT` or [configs/default.json](C:/Projects/WonderOfU/configs/default.json). Large datasets are intentionally external and never downloaded by this repository.

Run `python scripts/check_datasets.py`, `python -m unittest tests.test_foundation`, and after dependencies are installed, `python scripts/smoke_test.py`. Start the API with `uvicorn backend.main:app --reload`; serve `frontend/index.html` with any local static server.

Dataset adapters are manifest-first: each official/curated dataset root needs `manifest.csv` with `path,label,source_id,split`. Labels are never guessed and splits must be source/video/document-disjoint.

## Limitations

No target dataset was detected during initial setup and no model has been trained. The API/UI therefore return unavailable results instead of predictions. rPPG is a lightweight CHROM initial implementation over a central aligned-face ROI; its performance must be validated with real facial video.
