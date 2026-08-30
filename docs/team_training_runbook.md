# WonderOfU team training runbook

## Shared path

`Dataset → Adapter → manifest → mandatory preflight → DataLoader → model → shared engine → validation → checkpoint → held-out test → metrics`.

Use the one team command; do not create a personal training loop:

```powershell
.\.venv\Scripts\python scripts\run_experiment.py --dataset dfdc --dataset-root C:\WonderOfU_Data\DFDC --manifest C:\WonderOfU_Data\DFDC\manifest.csv --preflight-only
.\.venv\Scripts\python scripts\run_experiment.py --dataset dfdc --dataset-root C:\WonderOfU_Data\DFDC --manifest C:\WonderOfU_Data\DFDC\manifest.csv --epochs 5 --batch-size 4
```

The runner selects the task-matched baseline by default. Choose `--model unified` only for an approved unified-model experiment. It writes `config.json`, `metrics.json`, `best.pt`, `last.pt`, `preflight.json`, `reproducibility.json`, and `test_metrics.json` into one timestamped experiment directory.

## Non-negotiable manifest contract

Every row must contain `path,label,source_id,split`. Paths are relative to the dataset root; labels are explicit `0` or `1`; splits are `train`, `val`, or `test`. `source_id` represents the complete underlying video/identity/document relationship. No relationship may cross splits. The preflight fails closed on missing files, missing splits, bad schema, incompatible baselines, leakage, or unsuccessful real preprocessing.

Never randomly split related files; invent label mappings; bypass preflight/leakage checks; write a personal trainer; change architecture independently; tune against test; fabricate metrics; or call a dataset-specific baseline the final unified WonderOfU model.

## Friend 1 — DFDC (face forgery)

Obtain DFDC legitimately, inspect its actual metadata, and verify its official real/fake semantics before writing a binary manifest. Determine whether source-video, actor, or another relationship must form `source_id`; retain every related derivative in one split. Run preflight, inspect its class/source counts and preprocessing result, then use the default `baseline_forgery`. Record the exact subset and any preprocessing failures. Do not assume DFDC metadata or mappings from this repository.

## Friend 2 — IDNet EST (document fraud)

Inspect the officially obtained EST data structure and metadata before assigning labels. Verify authentic/manipulated semantics and choose a `source_id` that keeps related documents, templates, captures, or identities together. Run preflight, inspect its report, then use the default `baseline_document`. Do not infer mappings or grouping semantics from filenames alone.

## Return package

Return the manifest, preflight report, reproducibility record, metrics, checkpoints, run configuration, and a concise note of verified label/source decisions. Test metrics are held out and must not drive tuning.
