# Neuroscience in Conversation: computational analysis of the CNSeminar archive

Code accompanying the Aperture Neuro paper *"Neuroscience in Conversation: Tracking Scientific Change Through an Open Seminar Archive"*, a computational text analysis of the transcript archive of the [CNSeminars YouTube channel](https://www.youtube.com/@CNSeminars) (about 470 transcripts, 2010-2026).

The pipeline embeds, clusters and tracks the language of the archive to ask how the topics, vocabulary and conceptual organisation of neuroscience have changed over time, and whether emerging topics can be forecast from past trends.

> **Data.** The transcripts are not included in this repository. The scripts expect a root folder with one subfolder per year, each containing `.txt` transcript files (see [Input data](#input-data)).

## Repository layout

```
src/        analysis pipeline (run from this folder; scripts import each other)
extras/     standalone scripts for channel metrics and journal-bias figures
```

| Script | Purpose | Main outputs |
|---|---|---|
| `cns_analysis_pipeline.py` | Shared preprocessing (tokenisation, lemmatisation, stopwords, compound terms, content-type classification) plus overview figures; run with `--stratify` for per-content-type analyses | overview figures, `content_type_summary.csv`, `file_manifest.csv` |
| `cns_figstyle.py` | Shared figure-readability settings (font sizes, tick decluttering) | - |
| `cns_r1_demographics.py` | **R1** Corpus characteristics and channel growth | `r1_yearly_summary.csv`, `r1_file_manifest.csv` |
| `cns_r2_embedding.py` | **R2** Intellectual landscape: TF-IDF, UMAP, HDBSCAN | `r2_document_clusters.csv`, `r2_cluster_summary.csv` |
| `cns_r2_sweep.py` | HDBSCAN parameter sweep (`min_cluster_size` x `min_samples`) | `r2_sweep_summary.csv` |
| `cns_r3_vocabulary.py` | **R3** Changing vocabulary: TF-IDF by year, vocabulary growth, emerging and declining terms | `r3_vocabulary_stats.csv`, `r3_term_trajectories.csv`, `r3_emerging.csv`, `r3_declining.csv` |
| `cns_r4_topic_dynamics.py` | **R4** Topics that emerged, persisted and declined | `r4_topic_dynamics.csv` |
| `cns_r567_thematic.py` | **R5-R7** Conceptual shifts, methods/populations/resources, open science | `r567_thematic_trajectories.csv` |
| `cns_r89_forecasting.py` | **R8-R9** Rolling-window forecasting of emerging terms, backtested against a persistence baseline | `r89_backtest_results.csv`, `r89_<year>_forecast.csv` |
| `cns_r10_sensitivity.py` | **Supplementary** Sensitivity and robustness analyses (CNSeminars only, without Neuroccino, without short-form, year-balanced subsampling, 2020-2026, parameter grids) | `sensitivity_*.csv`, `baseline_labels.csv` |
| `cns_r10b_crosscheck.py` | **Supplementary** Do the main themes survive clustering in 10 UMAP dimensions? | `crosscheck_*.csv`, `fig_crosscheck_map.png` |
| `cns_r10c_supplement.py` | **Supplementary** Builds the supplementary tables and figures from the CSVs above (no recomputation) | `Table_S*`, `Fig_S*` |
| `extras/make_channel_figures.py` | YouTube channel metrics 2017-2026 (data are edited at the top of the script) | `fig_a_views.jpg` ... `fig_combined_5panels.jpg` |
| `extras/neuroccino_journal_bias.py` | Journal and publisher composition of the papers presented in Neuroccino sessions | `neuroccino_parsed.csv` + printed summary |

## Installation

Python 3.8 or later.

```bash
git clone <repository-url>
cd cns-seminar-text-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

NLTK's WordNet data are downloaded automatically on first run.

## Input data

```
transcripts/
    2016/  talk_title.txt  ...
    2017/  ...
    ...
```

The content type of each video (Neuroccino, CNSeminars, courses and educational series, community and open science, other) is inferred from the file name by `classify_content_type()` in `cns_analysis_pipeline.py`, so file names should follow the channel's naming conventions.

`extras/neuroccino_journal_bias.py` reads a tab-separated file `neuroccino_raw.txt` (columns: date `dd/mm/yyyy`, presenter, second name, paper URL, title) from the working directory.

## Usage

Run everything from `src/`. Each script takes `--folder` (transcript root) and `--output`; see `--help` for all options.

```bash
cd src

# Overview figures (add --stratify for per-content-type analyses)
python cns_analysis_pipeline.py --folder /path/to/transcripts --output ./figures --stratify

# Results 1-9
python cns_r1_demographics.py  --folder /path/to/transcripts --output ./r1_figures
python cns_r2_embedding.py     --folder /path/to/transcripts --output ./r2_figures
python cns_r3_vocabulary.py    --folder /path/to/transcripts --output ./r3_figures
python cns_r4_topic_dynamics.py --folder /path/to/transcripts --output ./r4_figures \
    --clusters ./r2_figures/r2_document_clusters.csv
python cns_r567_thematic.py    --folder /path/to/transcripts --output ./r567_figures
python cns_r89_forecasting.py  --folder /path/to/transcripts --output ./r89_figures \
    --trajectories ./r3_figures/r3_term_trajectories.csv
```

R2 settings used in the paper: TF-IDF with at most 3,000 terms (`min_df` 2, L2 norm); UMAP with cosine metric, `n_neighbors` 20, `min_dist` 0.05, seed 42; HDBSCAN with `min_cluster_size` 15 and `min_samples` 3.

### Sensitivity and robustness analyses (Supplementary)

```bash
python cns_r10_sensitivity.py  --root /path/to/transcripts --out sensitivity_out_v2
python cns_r10b_crosscheck.py  --root /path/to/transcripts \
    --baseline sensitivity_out_v2/baseline_labels.csv --out crosscheck_out
python cns_r10c_supplement.py  --v2 sensitivity_out_v2 --grid sensitivity_out \
    --cc crosscheck_out --out supplement_out
```

`cns_r10c_supplement.py` only reads CSV files. The theme labels in its `THEMES` dictionary are descriptive labels derived from each cluster's top terms and should match the topic names used in the paper.

### Channel figures

```bash
python extras/make_channel_figures.py figures
```

## Notes on the analysis

- Stopwords remove only non-discriminative words; genuinely neuroscientific terms are kept even when frequent.
- Neuroscience compound terms (for example "white matter", "working memory") are joined before tokenisation and restored for display.
- Short acronyms (MRI, EEG, PET and similar) are preserved by the tokenizer.
- Lemmatisation uses NLTK, so keyword and stopword lists only need base forms.
- Results R8-R9: the forecast is only reported if the model beats the persistence baseline (lower MAE and a hit rate above `--hit_threshold`).

## Troubleshooting

If `hdbscan` fails with an error about `force_all_finite` / `ensure_all_finite`, this is a version mismatch between `hdbscan` and `scikit-learn`; installing a matching pair of versions (typically an older `scikit-learn`) resolves it.

## Citation

If you use this code, please cite the paper:

> [Authors]. Neuroscience in Conversation: Tracking Scientific Change Through an Open Seminar Archive. *Aperture Neuro* (2026). [DOI]

## License

[To be added]
