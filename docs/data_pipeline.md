# Real-data ingestion and preprocessing

The adapters remain manifest-first. Each official dataset root supplies its own `manifest.csv` with `path,label,source_id,split`; `path` is relative to the root and `split` is one of `train`, `val`, or `test`. `source_id` is passed through every dataset item and batch so a future split validator/trainer can enforce source-disjoint evaluation.

Document samples are decoded with Pillow, converted to RGB, resized, normalized, and emitted as `[3,H,W]`. Video samples are decoded with OpenCV and uniformly sampled to the configured frame count. Each RGB frame goes through OpenCV Haar face detection; the largest face is padded, cropped, and resized. The crop is geometric normalization rather than landmark alignment—this limitation is deliberate and documented. In the default configuration, a missed face raises an error. The center-crop fallback is disabled by default and is used only by tiny software test fixtures.

Face batches expose a representative visual frame `[B,3,H,W]` plus a CHROM rPPG signal `[B,1,T]`; document batches expose `[B,3,H,W]`. Batches retain `source_id`, `split`, input path, and manifest metadata. Face and document samples must be loaded in separate homogeneous batches; the masked multi-task loss supports their partial labels without inventing labels.
