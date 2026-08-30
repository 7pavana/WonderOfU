# Training and evaluation engine

`ml.training.run_training` accepts separate homogeneous DataLoaders for each available task. A unified model can therefore process SiW/FaceForensics/DFDC video batches and IDNet/recapture document batches in one epoch; masks activate only the labels present in each batch. Baselines accept only their matching task and bypass the unified shared representation.

Before a real run, call `validate_source_disjoint_splits` on every participating adapter. It rejects source IDs that appear in more than one `train`, `val`, or `test` split within the same dataset. The trainer validates every epoch and saves `config.json`, `metrics.json`, `last.pt`, and `best.pt` under a timestamped experiment directory. `evaluate_model` evaluates a separate test DataLoader without updating weights. Resuming restores the model, optimizer, history, and next epoch.

Evaluation reports accuracy, precision, recall, F1, confusion counts, and ROC-AUC. ROC-AUC is `null` when the evaluated labels contain only one class; no substitute value is fabricated. This engine is software-validated only until official data is supplied.
