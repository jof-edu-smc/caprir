# IR Semantic Dataset Workflow

This repository contains the scripts used to build the dataset, split it, train captioning models, and evaluate them. The recommended workflow is:

1. Build convolved audio and stimulus metadata with `convolution_pipeline.py` when needed.
2. Build metadata with `dataset_build_natural.py` or `dataset_build_synthetic.py`.
3. Optionally inspect the room impulse response corpus with `clustering.py`.
4. Create train and test splits with `dataset_create_split.py`.
5. Fine-tune either Whisper or WSTXXN.
6. Generate evaluation candidates with `evaluate_dcase.py`.
7. Score the generated captions with `evaluate_aac_metrics.py`.

For environment and credential setup, see [Appendix A: Environment Setup](#appendix-a-environment-setup).

## 0. Generate Convolved Audio

Use `convolution_pipeline.py` if you need to build convolved speech, music, and clap stimuli from room impulse responses before assembling metadata.

```bash
python convolution_pipeline.py --openair_dir openAIR --mit_dir MIT_audio --wet_mix 1.0
```

Notes:

- `--openair_dir` points to the OpenAIR RIR directory.
- `--mit_dir` points to the MIT IR directory.
- `--wet_mix` controls the dry/wet balance, where `0.0` is fully dry and `1.0` is fully wet.
- The script writes `rir_dataset.csv` when it has data to save.
- See [Appendix A](#appendix-a-environment-setup) for the shared `.env` setup.

## 1. Build Metadata

The metadata builders read local audio assets and Google Sheets-backed annotations, then write CSV metadata used by the downstream scripts.

### Natural metadata

Use `dataset_build_natural.py` when you want to assemble the natural-caption metadata from the Google Sheets tables.

```bash
python dataset_build_natural.py
```

Notes:

- The script reads Google Sheets data using the credentials described in [Appendix A](#appendix-a-environment-setup).
- It expects the local audio directories referenced in the script to exist, including `MIT_audio` and `openAIR`.
- The current code writes `metadata_synthetic_2.csv` by default. If you want a different filename such as `metadata_natural.csv`, update `Config.output_file` before running.

### Synthetic metadata

Use `dataset_build_synthetic.py` when you want to generate synthetic captions with Gemini and combine them with the stimulus metadata.

```bash
python dataset_build_synthetic.py
```

Notes:

- The script reads the same Google Sheets source tables and the same local audio folders as the natural builder.
- It writes `metadata_synthetic.csv` by default.
- Synthetic caption generation uses the Gemini API key described in [Appendix A](#appendix-a-environment-setup).

## 1a. Inspect or Cluster the Dataset

`clustering.py` is an optional analysis step for exploring the generated caption embeddings and acoustic metadata. It can be run after the metadata CSV exists.

```bash
python clustering.py --metadata metadata_synthetic.csv --output_png clustering_dbscan.png --max_len 100
```

Notes:

- `--metadata` points to the CSV file to analyze.
- `--output_png` controls the main output image filename.
- `--max_len` is accepted by the CLI, but the current implementation does not use it as a filtering step.

## 1b. Split the Dataset

Use `dataset_create_split.py` to create train and test CSV files from the synthetic and natural metadata.

```bash
python dataset_create_split.py \
	--dataset_synthetic metadata_synthetic.csv \
	--dataset_natural metadata_natural.csv \
	--train_split 0.75 \
	--test_split 0.25
```

Notes:

- The script expects the `RIR` column to exist in both input CSV files.
- The default split is 75% train and 25% test.
- The output filenames are derived from the synthetic dataset name.

## 2. Fine-Tune a Model

The fine-tuning scripts expect a prepared training metadata CSV. Use the environment that matches the model you want to train.

### Whisper

Create the Whisper environment from the matching YAML file, then run training.

```bash
conda env create -n whisper -f env_whisper.yml
python finetune_whisper.py --metadata train_metadata_synthetic_075.csv --epochs 25 --learning_rate 4e-6
```

Notes:

- The default metadata argument is `metadata.csv`, but the example above points to the generated training split.
- The model is saved to `./whisper-finetuned`.
- If you are using a different training CSV, pass that file through `--metadata`.

### WSTXXN

Create the WSTXXN environment from the matching YAML file, then run training.

```bash
conda env create -n wstxxn -f env_wstxxn.yml
python finetune_wstxxn.py --metadata train_metadata_synthetic_75.csv --epochs 20 --learning_rate 4e-6
```

Notes:

- The default metadata argument is `metadata.csv`, but the example above points to the generated training split.
- The model is saved to `./wstxxn-finetuned`.
- The script also accepts `--batch_size` if you want to override the default batch size.

## 3. Evaluate Caption Generation

Use `evaluate_dcase.py` to run inference on the held-out test metadata and produce a candidates-and-references JSON file for metric scoring.

```bash
python evaluate_dcase.py \
	--metadata test_metadata_synthetic_025.csv \
	--whisper \
	--checkpoint ./whisper-finetuned
```

or

```bash
python evaluate_dcase.py \
	--metadata test_metadata_synthetic_025.csv \
	--wstxxn \
	--checkpoint ./wstxxn-finetuned
```

Notes:

- Exactly one of `--whisper` or `--wstxxn` must be set.
- `--checkpoint` should point to the model directory you want to evaluate. If omitted, the script falls back to the default Hugging Face checkpoint inside the code path for that model.
- The script writes a JSON file named like `candidates_references_whisper_<metadata>.json` or `candidates_references_wstxxn_<metadata>.json`.

## 4. Score AAC / DCASE Metrics

Use `evaluate_aac_metrics.py` after `evaluate_dcase.py` to score the generated candidates against their references.

```bash
python evaluate_aac_metrics.py \
	--json_file candidates_references_whisper_metadata_synthetic_025.json \
	--output_file candidates_references_whisper_metadata_synthetic_025_scores.json
```

Notes:

- The script consumes the JSON emitted by `evaluate_dcase.py`.
- The underlying metric configuration is `dcase2024` through the `aac_metrics` package.
- The script uses a Java 11 runtime path inside the code; see [Appendix A](#appendix-a-environment-setup) for the metrics environment note.

## Appendix A: Environment Setup

This appendix covers Conda environment creation and the `.env` values expected by the scripts.

### Conda environments

Create each environment from the matching YAML file in the repository root.

```bash
conda env create -n whisper -f env_whisper.yml
conda env create -n wstxxn -f env_wstxxn.yml
conda env create -n hf -f env_hf.yml
conda env create -n aac-metrics -f env_aac_metrics.yml
```

If you already have an environment, you can also update it with the same YAML file using your preferred Conda workflow.

### `.env` file

Create a `.env` file in the repository root with the variables used by the dataset builders and evaluation scripts.

Recommended variables:

```bash
GOOGLE_SHEETS_ID=your_google_sheet_id
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

Notes:

- `GOOGLE_SHEETS_ID` and `GOOGLE_SERVICE_ACCOUNT_JSON` are required for the Google Sheets-backed dataset builders.
- `GEMINI_API_KEY` is required by `dataset_build_synthetic.py`.
- `OPENAI_API_KEY` is only used if you enable the OpenAI-based branch in `dataset_build_natural.py`.
- `OPENAI_API_KEY` is also required by `convolution_pipeline.py`.

### Metrics runtime

`evaluate_aac_metrics.py` expects a Java 11 runtime available at the path configured in the script. If your Java installation lives elsewhere, update the `java_path` value in the script before running the metrics step.

## Repository Outputs

Common files produced by the workflow include:

- `metadata_natural.csv`
- `metadata_synthetic.csv`
- `train_metadata_synthetic_075.csv`
- `test_metadata_synthetic_025.csv`
- `candidates_references_whisper_*.json`
- `candidates_references_wstxxn_*.json`
- `*_scores.json`

These outputs are used by the training and evaluation scripts in the order described above.
