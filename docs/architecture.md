# Architecture and methodology

WonderOfU combines related fraud signals rather than treating the three tasks as interchangeable labels. Aligned face-video frames feed a lightweight visual encoder. The same face ROI sequence feeds CHROM: per-frame central skin-region RGB means are normalized; chrominance projections are combined using their standard-deviation ratio, then standardized. This yields a real temporal physiological signal of shape `[batch, 1, frames]`, which the temporal encoder processes.

Document images use a separate encoder and a document-specific projection into the same shared-representation space. Face visual and rPPG embeddings are fused through a face-specific projection before reaching that shared representation. This avoids duplicating document features or pretending a document has an rPPG modality. Spoof, forgery, and document heads are separate binary classifiers. Independent image-only baseline models bypass that shared representation.

The current CHROM implementation is intentionally lightweight. It depends on an upstream aligned face/skin crop and has not yet been validated on SiW; illumination, motion and makeup are known limitations. It must not be described as a medical measurement.

## Partial labels and leakage

Each batch carries labels and boolean task masks. Cross-entropy is computed only over samples with a true label for that head and averaged across active tasks. Missing labels never become negative labels. Dataset adapters require an explicit `manifest.csv` with `path,label,source_id,split`; split assignment must be performed at source-video/document/subject level, never at individual-frame level.
