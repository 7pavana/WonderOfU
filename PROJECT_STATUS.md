# WonderOfU Project Status

## Overall Progress

Foundation implementation started on 2026-08-27.

## Completed

- Project configuration, privacy-oriented ignore rules, manifest-first dataset abstraction, five dataset adapters, CHROM rPPG extraction, unified model design, independent baselines, masked loss, unavailable-safe inference contract, API/UI foundations, dependency-aware smoke tests, real-media ingestion/preprocessing components, a reusable training/evaluation engine, and a standardized team experiment wrapper/runbook.

## In Progress

- FaceForensics++ Deepfakes c23 pilot preprocessing validated. An expansion targeting 12 independent source groups (24 originals + 24 Deepfakes) was retried on 2026-08-30. The official EU2 downloader completed 8 originals, then stalled while transferring the ninth ordered original (`866.mp4`). Its verified 2,039,808-byte temporary transfer file was removed after confirming there were no active downloader processes. No manifest or training has been created from the incomplete expansion.

## Not Started

- Real dataset manifest curation, face detection/alignment, real training/evaluation, trained-checkpoint inference, and cross-domain experiments.

## Dataset Availability

- `C:\WonderOfU_Data` was not present during initial inspection. All five adapters are ready but unavailable.
- Audit on 2026-08-27 did not download datasets or create manifests/sample data.
- FaceForensics++ access is APPROVED. On 2026-08-28, the officially supplied downloader fetched only the approved pilot to `C:\WonderOfU_Data\FaceForensicsPP`: `original` videos `585.mp4`, `599.mp4` and `Deepfakes` videos `585_599.mp4`, `599_585.mp4`, all at `c23` through `EU2`. Total MP4 size is 7,470,541 bytes (~7.47 MB). No masks, models, c40/raw data, other manipulation types, or other datasets were downloaded.
- Official `filelist.json` inspection on 2026-08-30 verified 500 pairs, 1,000 unique source IDs, and no repeated source ID: each pair is an independent two-source group. The planned first experiment uses 12 groups split 8/2/2 for train/val/test (32/8/8 videos, balanced per group), but the download is incomplete: 8 valid originals (`585`, `599`, `469`, `481`, `183`, `253`, `672`, `720`) and 2 existing pilot Deepfakes are present. New Deepfakes were not started. EU2 stalled on the next ordered original, `866.mp4`; its sole incomplete temporary file was verified and removed. No completed data was deleted.

## Implemented Components

- Visual/document encoders, CHROM rPPG encoder path, modality-specific projections into a learned shared representation, three heads, baselines, masked loss, dataset availability detection, upload-session API contract, and frontend request flow.
- All adapters require the same explicit CSV header: `path,label,source_id,split`; valid splits are `train`, `val`, and `test`; all required values must be non-empty and `label` must be `0` or `1`. The semantic mapping must come from each official dataset's metadata.
- Pillow RGB document decoding; OpenCV video decoding with uniform frame sampling; configurable OpenCV Haar face detection; padded crop/resize geometric face normalization; RGB tensor conversion/normalization; CHROM rPPG extraction; PyTorch dataset/DataLoader and task-mask collation. `source_id`, split, input path, and manifest metadata are retained in every batch.
- Configurable unified/baseline model factory; AdamW or SGD; seed and CPU/auto-device selection; epoch-level training/validation; source-disjoint manifest validation; resumable checkpoints; timestamped experiment directories; metrics/config JSON; and standalone test-split evaluation.
- Standardized `scripts/run_experiment.py` interface for DFDC/IDNet/other existing adapters. It runs a fail-closed manifest/preprocessing/leakage/model-compatibility preflight, delegates to the existing engine, then saves `preflight.json`, `reproducibility.json`, and held-out `test_metrics.json` alongside the existing artifacts. It does not encode unverified DFDC or IDNet label mappings.

## Tests Passed

- `scripts/check_datasets.py` completed: all five adapters correctly reported unavailable, with no crash.
- `python -m unittest tests.test_foundation` completed: 3 tests passed (configuration, unavailable dataset reporting, and finite CHROM rPPG output).
- Python compilation completed for `ml`, `backend`, `scripts`, and `tests`.
- `scripts/smoke_test.py` PASSED on CPU: face visual+rPPG and document forward paths, all three task heads, masked partial-label losses, backward pass, optimizer step, checkpoint save/load, and the safe no-checkpoint inference response.
- API import/route check passed; inference with no checkpoint correctly returns `unavailable` rather than a prediction.
- `python -m unittest tests.test_foundation tests.test_data_pipeline` completed: 5 tests passed. Disposable test fixtures validated `manifest → image/video decode → preprocessing → tensor → DataLoader → unified-model-compatible batch` for document and face-video inputs. These are software checks only, not ML performance results.
- `scripts/smoke_training.py` completed: 3 training-engine tests passed. A disposable video fixture verified unified training, epoch validation, ROC-AUC calculation with both classes present, `last.pt` checkpoint creation, resume into the next epoch, test-split evaluation, and source-leakage rejection. It also constructed and forwarded all three independent baselines.
- Full local suite completed: 8 tests passed. No performance result or experiment artifact from disposable fixtures is retained.
- Team-standardization test suite completed: 10 tests passed. Disposable fixtures verified valid preflight, missing-column failure, missing-file failure, source-leakage failure, incompatible-baseline failure, standardized test-metric output, and reproducibility-record output. No real dataset was used.

## Experiments Run

- NOT RUN. No official dataset was available.

## Actual Results

- NOT RUN.

## Known Issues

- Exact `.venv` runtime: Python 3.13.5; NumPy 2.5.2; PyTorch 2.13.0+cpu; FastAPI 0.141.1; Uvicorn 0.52.4; Pillow 12.3.0; `opencv-python` package 5.0.0.93 (`cv2.__version__` reports 5.0.0). `torch.cuda.is_available()` is `False`, device count is `0`, and `torch.version.cuda` is `None`; the installed CPU PyTorch build does not detect the RTX 2050.
- `opencv-python` 5.0.0.93 was replaced inside `.venv` only with 4.10.0.84 because the former lacked `cv2.CascadeClassifier`. The working package reports `cv2.__version__ == 4.10.0` and exposes `CascadeClassifier`; the requirements file is pinned accordingly.
- Real training remains blocked by absent official data/manifests and trained-checkpoint file inference. The trainer is implemented, but a real run still requires official label mappings, source-disjoint manifests, and real-data verification of the lightweight face/rPPG preprocessing.
- Real FaceForensics++ preprocessing validation passed for all 4 pilot videos: each decoded at 1280×720/30 FPS, yielded 32 sampled frames, had Haar detections on 32/32 sampled frames, generated 32×160×160 RGB crops, a 3×160×160 normalized tensor, and a finite 32-sample CHROM rPPG signal. This validates ingestion only, not model performance.
- The 4-video pilot represents one connected source relationship (`585` and `599`, with both directional fakes). It cannot safely populate separate train/validation/test sets without source leakage; no manifest, training run, checkpoint, or FaceForensics++ metric has been created.
- The partial 2026-08-30 expansion remains insufficient for the planned 12-group split. Do not generate a manifest or train until the bounded original/Deepfakes download is completed and all group relationships are verified. The official downloader safely skips completed target MP4s on rerun, but it does not resume a partial temporary transfer: retrying the same `-n 24` command would download only missing target files but start `866.mp4` again from scratch. The official script also declares `EU` and `CA` server choices alongside `EU2`; they are the only legitimate alternatives identified in the supplied downloader, and no switch has been made.
- The initial OpenCV Haar detector performs detection plus geometric crop/resize, not landmark alignment. It is lightweight but needs real-data validation; default configuration correctly rejects videos where no face is detected.

## Research Decisions

- Labels are accepted only from explicit manifests; source-level split information is mandatory to prevent leakage. Lightweight CHROM rPPG is an initial physiological signal, not duplicated RGB features.

## Next Steps

- Keep FaceForensics++ download paused pending a later EU2 retry. For DFDC/IDNet, each assigned group member must legitimately obtain the data, verify official label/source semantics, create a source-disjoint manifest, run the standardized preflight, and then use the shared runner. Select a compatible CUDA PyTorch build only if GPU training is required.
