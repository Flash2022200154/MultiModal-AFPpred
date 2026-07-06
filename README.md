# MultiModal-AFPpred

A multimodal deep learning model for antimicrobial peptide (AMP) prediction, integrating ESM-2 protein language model embeddings, physicochemical properties, and secondary structure features through hierarchical cross-modal attention fusion.

## Architecture

MultiModal-AFPpred employs a three-stage architecture:

1. **Multimodal Feature Extraction**
   - **ESM-2 embeddings** (`facebook/esm2_t30_150M_UR50D`, 640-dim): pre-trained protein language model representations
   - **Physicochemical features** (256-dim): 12 AAIndex-derived properties with multi-scale sliding windows and Daubechies wavelet transformation
   - **Secondary structure features** (128-dim): predicted by NetSurfP-2.0 with Daubechies wavelet transformation

2. **Hierarchical Cross-Modal Fusion** (`HierarchicalTriModalFusion`, 512-dim output)
   - Layer 1: Cross-modal attention between PhysChem and Secondary Structure features
   - Layer 2: Cross-modal attention between ESM-2 and fused PhysChem+SS features

3. **Sequence-Aware Classification**
   - Residual BiLSTM block with parallel 1D convolutions (kernel=3, 5) and max-pooling
   - MLP classifier head with sigmoid output (AMP / non-AMP)

## Directory Structure

```
MultiModal-AFPpred/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── data_positive.csv
│   └── data_negative.csv
├── src/
│   ├── aaindex_analysis.py              # AAIndex property selection
│   ├── classifier.py                    # MLP classifier head
│   ├── data_loader.py                   # Data loading and preprocessing
│   ├── datasets.py                      # PyTorch dataset classes
│   ├── enhanced_physchem_features.py    # AAIndex + wavelet features
│   ├── feature_extractor.py             # ESM-2 embedding extractor
│   ├── fusion.py                        # Cross-modal attention fusion
│   ├── physchem_features.py             # Physicochemical feature extraction
│   ├── pipeline.py                      # Full model pipeline (MultiModalAFPpred)
│   ├── secondary_structure_features.py  # Secondary structure features
│   ├── sequence_model.py                # Residual BiLSTM block
│   ├── standardize.py                   # Feature standardization
│   └── trainer.py                       # Training utilities
└── tests/
    └── run_final_training_demo.py       # Training demo script
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

## Installation

```bash
git clone https://github.com/<your-username>/MultiModal-AFPpred.git
cd MultiModal-AFPpred
pip install -r requirements.txt
```

## ESM-2 Model Setup

This project uses `facebook/esm2_t30_150M_UR50D` from Hugging Face. The model weights (~600 MB) are **not included** in this repository. Download them before use:

```bash
# Option 1: Use huggingface-cli
pip install huggingface_hub
huggingface-cli download facebook/esm2_t30_150M_UR50D --local-dir ./esm2_t30_150M_UR50D

# Option 2: Use Python
python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D').save_pretrained('./esm2_t30_150M_UR50D'); AutoModel.from_pretrained('facebook/esm2_t30_150M_UR50D').save_pretrained('./esm2_t30_150M_UR50D')"
```

## Data

The `data/` directory contains:
- `data_positive.csv`: antimicrobial peptide sequences (positive samples)
- `data_negative.csv`: non-antimicrobial peptide sequences (negative samples)

Peptides were preprocessed with:
- Sequence similarity filtering (remove >70% similarity)
- Length filtering (5-50 amino acids)

## Usage

### Training

```python
import sys
sys.path.insert(0, "src")

from pipeline import MultiModalAFPpred

# Initialize model
model = MultiModalAFPpred(
    model_dir="./esm2_t30_150M_UR50D",
    classifier_return_logits=True
)

# Fit standardizers on training sequences
model.fit_standardizers(train_sequences)

# Predict
probabilities = model.predict_on_sequences(test_sequences)
```

### Run Training Demo

```bash
cd tests
python run_final_training_demo.py
```

## Citation

If you use this code, please cite our paper.

## License

This project is licensed under the MIT License.
