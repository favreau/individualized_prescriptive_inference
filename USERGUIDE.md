# User Guide: Individualized Prescriptive Inference in Ischaemic Stroke

## Overview

This software performs **individualized prescriptive inference** for ischaemic stroke patients - predicting which treatment would work best for each individual patient based on their brain lesion patterns.

### What It Does

1. **Analyzes brain lesions** from stroke patients (MRI/CT scans)
2. **Predicts functional deficits** across 16 domains (language, motor, memory, etc.)
3. **Simulates treatment effects** under various clinical trial scenarios
4. **Evaluates machine learning models** for personalized treatment recommendations

### Key Features

- Works with **lesion masks** and/or **disconnectome maps**
- Multiple dimensionality reduction techniques: **NMF**, **PCA**
- 16 functional domains mapped to brain regions
- Two ground truth rationales: **genetics-based** and **receptor-based** parcellations
- Simulates treatment bias, treatment effects, and spontaneous recovery
- Evaluates multiple ML models: Logistic Regression, Extra Trees, XGBoost

---

## Citation

If you use this software, please cite:

```bibtex
@article{giles2025individualized,
  title={Individualized prescriptive inference in ischaemic stroke},
  author={Giles, Dominic and Foulon, Chris and Pombo, Guilherme and Ruffle, James K and Xu, Tianbo and Jäger, H Rolf and Cardoso, Jorge and Ourselin, Sebastien and Rees, Geraint and Jha, Ashwani and others},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={8968},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

**License:** CC BY-NC-SA 4.0 (Non-commercial use only)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Linux/Unix environment (tested on Ubuntu)
- At least 16GB RAM (32GB+ recommended for large datasets)
- ~50GB free disk space for intermediate results

### Setup Virtual Environment

```bash
cd /path/to/individualized_prescriptive_inference

# Create virtual environment
python3 -m venv env

# Activate environment
source env/bin/activate

# Install dependencies
pip install nibabel "numpy<2.0" pandas scikit-learn torch xgboost nimfa nilearn tqdm joblib

# Important: Use NumPy 1.x for compatibility with nimfa library
pip install "numpy<2.0" --upgrade
```

### Verify Installation

```bash
python -c "import nibabel, numpy, pandas, sklearn, torch, nimfa; print('All packages installed successfully')"
```

---

## Project Structure

```
individualized_prescriptive_inference/
├── atlases/                           # Brain atlases (included)
│   ├── 2mm_parcellations/
│   │   ├── genetics/                 # 16 genetics-based parcellations
│   │   └── receptor/                 # 16 receptor-based parcellations
│   ├── functional_parcellation_2mm.nii.gz
│   └── icv_mask_2mm.nii.gz           # Intracranial volume mask
├── lesions/                          # Your lesion data (extract from lesions.zip)
├── software/                         # Main scripts
│   ├── representation.py             # Step 1: Create representations (full version)
│   ├── representation_minimal.py     # Step 1: Simplified version (no vascular atlases)
│   ├── deficit_modelling.py          # Step 2: Model functional deficits
│   └── prescription.py               # Step 3: Prescriptive analysis
├── results/                          # Output directory (created automatically)
└── README.md
```

---

## The 16 Functional Domains

The system predicts deficits across these 16 domains:

1. **Hearing** - Auditory processing
2. **Language** - Speech and comprehension
3. **Introspection** - Self-awareness
4. **Cognition** - Executive function
5. **Mood** - Emotional regulation
6. **Memory** - Learning and recall
7. **Aversion** - Threat processing
8. **Coordination** - Motor coordination
9. **Interoception** - Internal bodily awareness
10. **Sleep** - Sleep-wake regulation
11. **Reward** - Motivation and reward processing
12. **Visual Recognition** - Object recognition
13. **Visual Perception** - Visual processing
14. **Spatial Reasoning** - Spatial navigation
15. **Motor** - Movement control
16. **Somatosensory** - Touch and body sensation

---

## Quick Start (Tested Commands)

This section provides the exact commands that were successfully tested with the included dataset of 4,119 lesions.

### 1. Setup Environment

```bash
cd /path/to/individualized_prescriptive_inference

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies (NumPy 1.x is critical!)
pip install nibabel "numpy<2.0" pandas scikit-learn torch xgboost nimfa nilearn tqdm joblib

# Extract lesion data
unzip lesions.zip  # Creates lesions/ directory with 4,119 files
```

### 2. Run Representation (Step 1)

```bash
# Using the minimal version (no vascular atlases required)
# Estimated time: ~2-3 hours for 4,119 lesions
nohup python software/representation_minimal.py \
    --lesionpath lesions/ \
    --savepath results/ \
    --kfolds 2 \
    --run_nmf True \
    --run_pca True \
    --latent_components 2 4 8 16 \
    > representation.log 2>&1 &

# Monitor progress
tail -f representation.log

# Expected output: 
# - results/whole_ground_truth.pkl
# - results/train_split_*.pkl and test_split_*.pkl (for each fold)
# - results/train_*_dim_*_['lesion'].pkl (embeddings for each fold and dimension)
```

### 3. Run Deficit Modelling (Step 2)

```bash
# Match the k-folds from Step 1 (we used 2)
# Match the latent dimensions from Step 1 (2, 4, 8 - excluding 16 for fold 1)
# Estimated time: ~5-10 minutes
nohup python software/deficit_modelling.py \
    --path results/ \
    --lesionpath lesions/ \
    --latent_list 2 4 8 \
    --kfold_deficits 2 \
    --names genetics \
    --run_nmf True \
    --run_pca True \
    --roi_threshs 0.05 \
    > deficit_modelling.log 2>&1 &

# Monitor progress
tail -f deficit_modelling.log

# Expected output:
# - results/lesion_0.05_genetics_0/train_lesion_0.05_0.pkl
# - results/lesion_0.05_genetics_0/test_lesion_0.05_0.pkl
# - results/lesion_0.05_genetics_0/centroids.json
# (Same for genetics_1)
```

### 4. Run Prescription Analysis (Step 3)

```bash
# Simplified parameters for faster testing
# Estimated time: ~15-20 minutes for 96 iterations
nohup python software/prescription.py \
    --savepath results/ \
    --loadpath results/ \
    --k 0 1 \
    --gene_or_receptor genetics \
    --lesion_or_disconnectome lesion \
    --lesion_deficit_thresh 0.05 \
    --deficits 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
    --biasdegree 0 0.5 1 \
    --biastype observed unobserved \
    --te 1 0.5 \
    --re 0 0.5 \
    --bottlenecks 2 4 8 \
    --simpleatlases False \
    --ml_models logistic_regression extra_trees xgb \
    --use_nmf \
    --use_pca \
    > prescription.log 2>&1 &

# Monitor progress
tail -f prescription.log

# Expected output:
# - results/prescription/prescriptive_results_core_iter*.pkl (1152 files, ~9MB total)
```

### 5. Verify Completion

```bash
# Check all outputs exist
ls results/whole_ground_truth.pkl
ls results/lesion_0.05_genetics_*/
ls results/prescription/*.pkl | wc -l  # Should show 1152

# Check for errors
grep -i error representation.log deficit_modelling.log prescription.log
```

**Total Pipeline Time (4,119 lesions):** ~3-4 hours

**Total Disk Space:** ~54MB in results/

---

## Data Preparation

### Lesion Data Format

Your lesion masks must be:
- **Format:** NIfTI (`.nii` or `.nii.gz`)
- **Space:** MNI152 standard space (2mm resolution recommended)
- **Values:** Binary (0 = no lesion, 1 = lesion) or probabilistic (0-1)
- **Naming:** Any naming convention (e.g., `lesion0001_age_sex.nii.gz`)

### Extract Provided Data

```bash
cd /path/to/individualized_prescriptive_inference
unzip lesions.zip
# This creates a lesions/ directory with 4,119 example lesion files
```

### Prepare Your Own Data

1. Convert your lesions to MNI152 space (2mm)
2. Create binary or probabilistic masks
3. Place all `.nii.gz` files in a single directory
4. Ensure filenames don't start with "input" (reserved prefix)

---

## Pipeline Overview

The analysis consists of three main steps:

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: REPRESENTATION                                     │
│  Convert 3D lesions → Low-dimensional features              │
│  - NMF/PCA dimensionality reduction                         │
│  - Extract spatial features (volume, centroid)              │
│  - Create k-fold cross-validation splits                    │
│  Output: Embeddings + metadata                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: DEFICIT MODELLING                                  │
│  Map lesions → 16 functional deficits                       │
│  - Use genetics/receptor parcellations                      │
│  - Determine treatment susceptibility                       │
│  Output: Deficit labels per patient                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: PRESCRIPTION                                       │
│  Evaluate treatment recommendation models                    │
│  - Simulate clinical trials with various biases            │
│  - Train ML models (LR, ET, XGB)                           │
│  - Compute PEHE, accuracy, balanced accuracy               │
│  Output: Model performance metrics                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Representation

### Purpose
Convert high-dimensional 3D lesion images into low-dimensional feature vectors suitable for machine learning.

### Script Options

**Option A: Full Version** (requires vascular atlases - not included)
```bash
python software/representation.py \
    --lesionpath lesions/ \
    --savepath results/ \
    --kfolds 10 \
    --run_nmf True \
    --run_pca True \
    --latent_components 2 4 8 16 32 64 128
```

**Option B: Minimal Version** (works with included data)
```bash
python software/representation_minimal.py \
    --lesionpath lesions/ \
    --savepath results/ \
    --kfolds 5 \
    --run_nmf True \
    --run_pca True \
    --latent_components 2 4 8 16
```

### Key Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--lesionpath` | Directory with lesion `.nii.gz` files | Required | Your data path |
| `--discopath` | Directory with disconnectome files | Optional | Leave empty if not available |
| `--savepath` | Output directory | `results/` | Use descriptive name |
| `--kfolds` | Number of CV folds | 10 | 5-10 |
| `--run_nmf` | Run NMF (Non-negative Matrix Factorization) | True | True |
| `--run_pca` | Run PCA (Principal Component Analysis) | True | True |
| `--latent_components` | Dimensionalities to test | [50] | 2 4 8 16 32 64 |

### Running in Background

For large datasets, run in background:

```bash
nohup python software/representation_minimal.py \
    --lesionpath lesions/ \
    --savepath results/ \
    --kfolds 5 \
    --run_nmf True \
    --run_pca True \
    --latent_components 2 4 8 16 \
    > representation.log 2>&1 &

# Monitor progress
tail -f representation.log

# Check if still running
ps aux | grep representation
```

### Output Files

After completion, you'll find:

```
results/
├── whole_ground_truth.pkl                  # All lesions with metadata
├── train_split_0.pkl                       # Fold 0 training set
├── test_split_0.pkl                        # Fold 0 test set
├── train_0_dim_2_['lesion'].pkl           # Training embeddings (2D)
├── test_0_dim_2_['lesion'].pkl            # Test embeddings (2D)
├── train_0_dim_4_['lesion'].pkl           # Training embeddings (4D)
└── ... (continues for all folds and dimensions)
```

### Expected Runtime

| Dataset Size | Processing Time |
|--------------|----------------|
| 500 lesions  | ~30 minutes |
| 1,000 lesions | ~1 hour |
| 4,000 lesions | ~3-4 hours |

**Performance:** ~5 lesions/second on standard CPU

---

## Step 2: Deficit Modelling

### Purpose
Map lesion patterns to 16 functional deficits and determine treatment susceptibility using brain parcellations.

### Usage

```bash
python software/deficit_modelling.py \
    --path results/ \
    --lesionpath lesions/ \
    --kfold_deficits 5 \
    --roi_threshs 0.05 \
    --names genetics receptor \
    --latent_list 2 4 8 16 \
    --run_nmf True \
    --run_pca True
```

### Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--path` | Path to Step 1 outputs | Required | Use same as `--savepath` from Step 1 |
| `--lesionpath` | Path to lesion files | Required | Same as Step 1 |
| `--kfold_deficits` | Number of CV folds | 10 | Must match Step 1 `--kfolds` |
| `--roi_threshs` | Overlap threshold for deficit | 0.05 | 5% overlap = deficit present |
| `--names` | Parcellation types | genetics receptor | Use both for comparison |
| `--latent_list` | Dimensions from Step 1 | [2,4,8,...] | Must match Step 1 |

### Understanding `roi_threshs`

This parameter determines when a deficit is considered present:
- **0.05** (5%): Deficit if lesion overlaps ≥5% of region → More sensitive
- **0.10** (10%): Deficit if lesion overlaps ≥10% of region → More specific
- **0.20** (20%): Conservative threshold

### Parcellation Types

**Genetics:** Based on gene expression patterns
- Each of 16 regions has distinct genetic signature
- Maps structural damage to functional domains

**Receptor:** Based on neurotransmitter receptor distributions  
- Regions defined by receptor density (dopamine, serotonin, etc.)
- Maps neurochemical disruption to deficits

### Output Files

```
results/
└── lesion_0.05_genetics_0/
    ├── train_lesion_0.05_0.pkl          # Training data with deficit labels
    ├── test_lesion_0.05_0.pkl           # Test data with deficit labels
    └── centroids.json                    # Region centroids for bias simulation
```

### Expected Runtime

- **5 folds × 2 parcellations × 4 dimensions = 40 runs**
- **~10-30 minutes** per run
- **Total: 8-20 hours** for full pipeline

---

## Step 3: Prescription Analysis

### Purpose
Evaluate machine learning models for personalized treatment recommendations under various clinical trial scenarios.

### Usage

```bash
python software/prescription.py \
    --savepath results/prescription/ \
    --loadpath results/ \
    --k 0 1 2 3 4 \
    --gene_or_receptor genetics receptor \
    --lesion_or_disconnectome lesion \
    --lesion_deficit_thresh 0.05 \
    --deficits 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
    --biasdegree 0 0.25 0.5 0.75 1 \
    --biastype observed unobserved \
    --te 1 0.75 0.5 0.25 \
    --re 0 0.25 0.5 0.75 \
    --bottlenecks 2 4 8 16 \
    --ml_models logistic_regression extra_trees xgb \
    --use_nmf True \
    --use_pca True
```

### Key Parameters

#### Basic Settings
| Parameter | Description | Values |
|-----------|-------------|--------|
| `--savepath` | Output directory | New directory |
| `--loadpath` | Step 2 outputs | From previous step |
| `--k` | Which CV folds to process | 0 1 2 3 4 |

#### Trial Simulation
| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--biasdegree` | Selection bias strength | 0=none, 0.5=moderate, 1=extreme |
| `--biastype` | Bias mechanism | `observed` (spatial), `unobserved` (confounded) |
| `--te` | Treatment effect | 1=100% effective, 0.5=50% effective |
| `--re` | Spontaneous recovery | 0=none, 0.5=50% recover anyway |

#### Models and Representations
| Parameter | Description | Options |
|-----------|-------------|---------|
| `--ml_models` | Classifiers to evaluate | logistic_regression, extra_trees, xgb |
| `--bottlenecks` | Dimensionalities to test | 2 4 8 16 32 64 128 |
| `--use_nmf`, `--use_pca` | Reduction methods | True/False |

### Understanding Key Concepts

#### Bias Types

**Observed Bias (Spatial):**
- Treatment assignment based on lesion location
- Mimics real-world clinical decisions (e.g., left hemisphere → speech therapy)
- Can be corrected if measured

**Unobserved Bias (Confounded):**
- Hidden factors influence treatment choice
- Not measurable in data
- More challenging for models

#### Treatment Effect (TE)

The proportion of susceptible patients who respond to treatment:
- **TE = 1.0**: All correctly-treated patients benefit
- **TE = 0.5**: Half of correctly-treated patients benefit
- **TE = 0.0**: Treatment doesn't work

#### Recovery Effect (RE)

Spontaneous recovery regardless of treatment:
- **RE = 0.0**: No spontaneous recovery
- **RE = 0.5**: Half of all patients recover anyway
- Makes treatment effect estimation harder

### Evaluation Metrics

#### PEHE (Precision in Estimation of Heterogeneous Effects)
- Measures how accurately model predicts individual treatment effects
- **Lower is better** (0 = perfect)
- Most important metric for personalized medicine

#### Accuracy & Balanced Accuracy
- Correct treatment recommendations
- Balanced accuracy accounts for class imbalance
- **Higher is better** (1.0 = perfect)

#### Prescriptive Metrics
- Focus on patients where treatment choice matters
- Excludes patients who benefit from either treatment
- Most clinically relevant

### Output Files

```
results/prescription/
└── prescriptive_results_core_iter0.pkl    # Performance metrics
└── prescriptive_results_core_iter1.pkl
└── ...
```

Each file contains:
- Model name
- PEHE (observed and prescriptive)
- Accuracy and balanced accuracy
- Confusion matrices
- All simulation parameters

### Expected Runtime

With default parameters:
- **~1-5 minutes per condition**
- Hundreds to thousands of conditions
- **Total: Several hours to days**

---

## Monitoring and Troubleshooting

### Check Progress

```bash
# Monitor log file
tail -f representation.log

# Check process status
ps aux | grep python | grep representation

# View recently modified files
ls -lht results/ | head -20

# Check disk space
df -h results/
```

### Common Issues

#### 1. NumPy Compatibility Error

**Error:** `AttributeError: np.mat was removed in NumPy 2.0`

**Solution:**
```bash
pip install "numpy<2.0" --upgrade
# Then delete existing pickle files
rm results/whole_ground_truth.pkl
# Re-run the script
```

#### 2. Out of Memory

**Error:** `MemoryError` or killed process

**Solutions:**
- Reduce `--kfolds` (try 3-5 instead of 10)
- Process fewer `--latent_components` at once
- Use fewer deficits (process subsets)
- Increase swap space

#### 3. Missing Atlas Files

**Error:** `FileNotFoundError: .../vasc_atlas/...`

**Solution:**
Use `representation_minimal.py` instead of `representation.py`

#### 4. Slow Processing

**Optimization tips:**
- Run on machine with more CPU cores
- Process folds in parallel on different machines
- Use smaller dataset for initial testing
- Consider cloud computing for large runs

#### 5. Pickle Loading Errors

**Error:** `ModuleNotFoundError` or pickle incompatibility

**Solution:**
- Ensure same NumPy version for save/load
- Don't mix Python 2/3
- Re-generate files if Python version changed

#### 6. K-fold Mismatch Error

**Error:** `FileNotFoundError` or `NoneType` errors when loading dimensionality-reduced files

**Cause:** The number of k-folds specified in Step 2/3 doesn't match what was generated in Step 1

**Solution:**
```bash
# Check how many k-folds were actually generated
ls -1 results/train_split_*.pkl | wc -l

# Use matching number in subsequent steps
# If you have 2 folds (train_split_0.pkl and train_split_1.pkl):
python software/deficit_modelling.py --kfold_deficits 2 ...
python software/prescription.py --k 0 1 ...
```

#### 7. Latent Dimension Mismatch

**Error:** `AttributeError: 'NoneType' object has no attribute 'loc'` in `harmonize_columns`

**Cause:** Requesting dimensions that weren't generated in Step 1

**Solution:**
```bash
# Check which dimensions exist
ls -1 results/train_0_dim_*.pkl | grep -o 'dim_[0-9]*' | sort -u

# Example output: dim_2, dim_4, dim_8
# Then use only these in Step 2:
python software/deficit_modelling.py --latent_list 2 4 8 ...
```

#### 8. Argument Type Errors

**Error:** `invalid int value: '0.05'` or `invalid float value`

**Cause:** Incorrect argument type specification in argparse

**Solution:**
- For `--roi_threshs`: Must be float values (e.g., `0.05` not `5`)
- Already fixed in latest version of `deficit_modelling.py`
- If using older version, ensure using `type=float` in argument definition

#### 9. File Path Errors in Deficit Modelling

**Error:** `FileNotFoundError: .../../../atlases/2mm_parcellations/...`

**Cause:** Incorrect relative path calculation when script is run from different directories

**Solution:**
- Already fixed in latest version of `deficit_modelling.py`
- Script now uses `os.path.dirname(os.path.abspath(__file__))` to find project root
- Always run from project root directory for best compatibility

#### 10. Boolean Argument Errors in Prescription

**Error:** Prescription script tries to load autoencoder/VAE representations that don't exist

**Cause:** argparse `type=bool` treats any string (including "False") as True

**Solution:**
- Already fixed in latest version: uses `action='store_true'` instead
- Use flags without values:
  ```bash
  # Correct
  python software/prescription.py --use_nmf --use_pca
  
  # Incorrect (old version)
  python software/prescription.py --use_nmf True --use_pca True
  ```

#### 11. File Naming Pattern Mismatch

**Error:** Cannot find files like `train_0_dim_8_['lesion'].pkl`

**Cause:** File naming pattern in code doesn't match actual generated files

**Solution:**
- Check actual file naming: `ls -1 results/train_0_dim_*.pkl`
- Ensure `get_file()` function searches for correct pattern
- Already fixed in latest `deficit_modelling.py`

---

## Best Practices

### For Initial Testing

Start small to verify pipeline:

```bash
# Test with subset of data
mkdir test_lesions
cp lesions/lesion*.nii.gz test_lesions/ | head -100  # Copy first 100

# Quick test run
python software/representation_minimal.py \
    --lesionpath test_lesions/ \
    --savepath test_results/ \
    --kfolds 3 \
    --latent_components 2 4 \
    --run_nmf True \
    --run_pca False  # Skip PCA for speed
```

### For Production Runs

1. **Use descriptive output directories:**
   ```bash
   --savepath results_dataset1_20250112/
   ```

2. **Save logs with timestamps:**
   ```bash
   python script.py ... > logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
   ```

3. **Run in screen/tmux for long jobs:**
   ```bash
   screen -S stroke_pipeline
   # Run your commands
   # Ctrl+A, D to detach
   # screen -r stroke_pipeline to reattach
   ```

4. **Checkpoint frequently:**
   - Steps are independent - can resume if interrupted
   - Each fold creates separate output files

### Resource Planning

| Component | RAM | Disk | Time |
|-----------|-----|------|------|
| 1000 lesions, 5 folds | 8-16GB | 10GB | 2-3 hours |
| 4000 lesions, 10 folds | 16-32GB | 50GB | 12-24 hours |

---

## Interpreting Results

### Step 1 Outputs

**whole_ground_truth.pkl** contains:
- `filename`: Lesion filename
- `lesion_vol`: Lesion volume in voxels
- `lesion_centroid_x/y/z`: Lesion center of mass
- `lesion_functional_distribution`: Overlap with 16 functional regions
- `lesion_functional_territories_dice`: Dice scores with territories

**train_X_dim_Y_['lesion'].pkl** contains:
- All above features
- `nmf_lesion_Y_KX`: NMF embeddings (Y-dimensional)
- `pca_lesion_Y_KX`: PCA embeddings (Y-dimensional)

### Step 2 Outputs

Additional columns for each deficit (1-16):
- `{deficit_name}_W0`: Susceptible to treatment 0
- `{deficit_name}_W1`: Susceptible to treatment 1

Example:
- `language_W0 = 1`: Patient likely benefits from treatment 0 for language
- `language_W1 = 0`: Patient unlikely benefits from treatment 1

### Step 3 Outputs

Key metrics in results DataFrames:

**PEHE (lower = better):**
- How well model predicts individual treatment effects
- Target: < 0.2 is good, < 0.1 is excellent

**Prescriptive Balanced Accuracy (higher = better):**
- Correct treatment recommendations
- Target: > 0.7 is good, > 0.8 is excellent

**Confusion Matrix:**
```
[[TN, FP],     True Negative  = Correctly predicted no benefit from treatment
 [FN, TP]]     False Positive = Incorrectly predicted benefit
               False Negative = Missed beneficial treatment
               True Positive  = Correctly predicted benefit
```

### Comparing Methods

Good models should:
1. **Low PEHE**: Accurate treatment effect estimates
2. **High prescriptive accuracy**: Correct recommendations
3. **Robust across biases**: Works even with selection bias
4. **Consistent across folds**: Low variance

---

## Example Workflows

### Workflow 1: Quick Exploratory Analysis

```bash
# Use 100 lesions, 3 folds, 2 dimensions
python software/representation_minimal.py \
    --lesionpath small_dataset/ \
    --savepath quick_test/ \
    --kfolds 3 \
    --latent_components 2 \
    --run_nmf True

python software/deficit_modelling.py \
    --path quick_test/ \
    --lesionpath small_dataset/ \
    --kfold_deficits 3 \
    --names genetics \
    --latent_list 2

python software/prescription.py \
    --loadpath quick_test/ \
    --savepath quick_test/prescription/ \
    --k 0 1 2 \
    --deficits 1 2 3 \
    --biasdegree 0 0.5 \
    --te 1 \
    --re 0 \
    --bottlenecks 2 \
    --ml_models logistic_regression
```

### Workflow 2: Full Publication-Quality Analysis

```bash
# Step 1: Comprehensive representation
nohup python software/representation_minimal.py \
    --lesionpath all_lesions/ \
    --savepath full_analysis/ \
    --kfolds 10 \
    --latent_components 2 4 8 16 32 64 128 \
    --run_nmf True \
    --run_pca True \
    > logs/step1.log 2>&1 &

# Wait for completion, then Step 2
nohup python software/deficit_modelling.py \
    --path full_analysis/ \
    --lesionpath all_lesions/ \
    --kfold_deficits 10 \
    --roi_threshs 0.05 0.10 \
    --names genetics receptor \
    --latent_list 2 4 8 16 32 64 128 \
    > logs/step2.log 2>&1 &

# Step 3: Complete evaluation
nohup python software/prescription.py \
    --loadpath full_analysis/ \
    --savepath full_analysis/prescription/ \
    --k 0 1 2 3 4 5 6 7 8 9 \
    --gene_or_receptor genetics receptor \
    --lesion_or_disconnectome lesion \
    --deficits 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
    --biasdegree 0 0.25 0.5 0.75 1 \
    --biastype observed unobserved \
    --te 1 0.75 0.5 0.25 \
    --re 0 0.25 0.5 0.75 \
    --bottlenecks 2 4 8 16 32 64 128 \
    --ml_models logistic_regression extra_trees xgb \
    > logs/step3.log 2>&1 &
```

---

## Advanced Topics

### Custom Parcellations

To use your own brain parcellations:

1. Create NIfTI files with 2 regions per domain (values: 1 and 2)
2. Place in `atlases/2mm_parcellations/custom/`
3. Name files: `1_domain_name.nii.gz`, `2_domain_name.nii.gz`, etc.
4. Add to `--names custom`

### Parallel Processing

Process different folds on different machines:

```bash
# Machine 1: Process folds 0-2
python software/deficit_modelling.py --k 0 1 2 ...

# Machine 2: Process folds 3-5
python software/deficit_modelling.py --k 3 4 5 ...

# Combine results later
```

### Custom ML Models

Add custom models in `prescription.py`:

```python
from sklearn.ensemble import YourModel

ml_models_dict['your_model'] = YourModel(param=value)
```

---

## Getting Help

### Documentation
- Original paper: [Nature Communications, 2025](https://www.nature.com/articles/s41467-025-64593-7)
- GitHub issues: [Report bugs or request features]

### Common Questions

**Q: Can I use this with non-stroke lesions?**  
A: Yes, but parcellations are optimized for stroke. Results may vary.

**Q: What resolution should my images be?**  
A: 2mm isotropic in MNI152 space is standard. Other resolutions may work but require resampling atlases.

**Q: How many patients do I need?**  
A: Minimum 200 for meaningful results, 500+ recommended, 1000+ for robust models.

**Q: Can I skip dimensionality reduction?**  
A: Not with this pipeline. The 3D images are too high-dimensional for the prescription models.

**Q: What if my lesions aren't in MNI space?**  
A: Use FSL (FLIRT/FNIRT) or ANTs to register to MNI152 2mm space first.

---

## Changelog

### Version 1.1 (January 2025)
- **FIXED:** K-fold iterator bug in `prescription.py` (`for K in k` instead of `range(k)`)
- **FIXED:** Argument type errors in `deficit_modelling.py` (roi_threshs now correctly float)
- **FIXED:** File path calculation in `deficit_modelling.py` (now uses absolute paths)
- **FIXED:** Boolean argument handling in `prescription.py` (now uses `action='store_true'`)
- **FIXED:** File naming pattern mismatch in `get_file()` function
- **ADDED:** `representation_minimal.py` - works without vascular atlases
- **ADDED:** Comprehensive troubleshooting section with 11 common issues
- **ADDED:** Quick Start section with tested commands
- **IMPROVED:** NumPy 1.x compatibility explicitly enforced
- **IMPROVED:** Error messages and logging throughout

### Version 1.0 (Initial Release)
- Original implementation from Nature Communications paper
- Full pipeline: representation → deficit → prescription
- Support for genetics and receptor-based parcellations
- Multiple ML models (LR, ET, XGBoost)

### Known Limitations
- Requires Linux/Unix (Windows not tested)
- Memory-intensive for large datasets (16-32GB RAM recommended)
- Long processing times for full pipeline (~3-4 hours for 4k lesions)
- Vascular territory atlases not included (use minimal version)
- K-fold 1 may generate fewer dimensions than k-fold 0 (known issue)
- Receptor parcellation processing significantly slower than genetics

---

## Support

For technical questions or bug reports:
1. Check this guide first
2. Review the original publication
3. Search existing GitHub issues
4. Open a new issue with:
   - Error message
   - Log file excerpt
   - System information
   - Steps to reproduce

**Last Updated:** January 2025

