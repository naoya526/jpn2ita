# Italian-Japanese Translator

A deep learning project focused on building an Italian-Japanese translator using LSTM and Transformer architectures.

## Project Overview

This project aims to develop a neural machine translation model for Italian-Japanese language pairs, addressing the unique challenges of translating between a Romance language and a logographic language system.

## Objectives

1. **Morphological Analysis Implementation**
   - Process non-alphabetic languages (Japanese) using morphological analysis
   - Utilize **Vibrato** for Japanese tokenization
   - Address word segmentation challenges (Japanese has no spaces between words)
   - Implement proper tokenization as a preprocessing step

2. **Dataset Refinement**
   - Resolve significant misalignment issues in the current corpus
   - Clean duplicate entries and improve data quality
   - Ensure proper sentence alignment for effective training

3. **Multi-Architecture Implementation**
   - Implement LSTM-based models for baseline performance
   - Develop Transformer models for advanced translation quality
   - Support both cloud (GCP with $300 credits) and local execution

4. **Model Training & Inference**
   - Train models on refined parallel corpus
   - Implement inference pipeline for real-time translation

5. **Fine-tuning (Optional)**
   - Apply PPO (Proximal Policy Optimization) on high-quality subset
   - Improve translation quality through reinforcement learning

## References & Baselines

### Primary References
1. **Deep Learning from Scratch (O'Reilly)**
   - Repository: [oreilly-japan/deep-learning-from-scratch-2](https://github.com/oreilly-japan/deep-learning-from-scratch-2)
   - Personal fork: [naoya526/Deeplearning2](https://github.com/naoya526/Deeplearning2)

2. **Pavia University - Machine Learning (Prof. Claudio Cusano)**
   - Reference: `English_to_Italian_automatic_translation.ipynb`


## Dataset Information

### Available Datasets
1. **JAICO** - Japanese-Italian Corpus
   - [Research Paper](https://www2.ninjal.ac.jp/past-events/2009_2021/event/specialists/project-meeting/files/JCLWorkshop_no6_papers/JCLWorkshop_No6_26.pdf)

2. **A4EDU** - Educational Language Resources
   - [Website](https://a4edu.unive.it/ita/index#do)

3. **TED Multilingual Parallel Corpus** (Currently Used)
   - [GitHub Repository](https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus)
   - Source: TED Talks transcriptions and translations

### Current Dataset Status

**Original Corpus:**
- Italian sentences: 349,048 lines
- Japanese sentences: 389,764 lines
- **Misalignment:** 40,716 line difference

**After Duplicate Removal:**
- Italian (cleaned): 346,929 lines
- Japanese (cleaned): 384,363 lines
- **Remaining misalignment:** 37,434 lines

### Sample Alignment

**Italian (Line #349044-349048):**
```
E questo è il piano per i tetti della città.
Abbiamo sopraelevato la terra sui tetti.
Gli agricoltori hanno piccoli ponti per circolare da un tetto all'altro.
Occupiamo la città con spazi abitativi e lavorativi in tutti i piani terra.
Quindi, questa è la città esistente, e questa è la città nuova.
```

**Japanese (Line #389760-389764):**
```
従来の土壌を屋根の上に持ち上げ
農業者は屋根から屋根へと
一階部分は仕事と生活のための
これが現在の街で こちらが新しい街です
（拍手）
```

**Verified Alignment:**
- IT (#349048): "Quindi, questa è la città esistente, e questa è la città nuova."
- JA (#389763): "これが現在の街で こちらが新しい街です"

## Dataset Challenges

### Primary Issues
1. **Significant Misalignment**
   - 37,434 line difference after cleaning
   - Inconsistent sentence boundaries
   - Missing or extra content in Japanese corpus

2. **Duplicate Content**
   - Repeated phrases, especially in Japanese corpus
   - Example: "E questo è il piano per i tetti della città." appears multiple times

3. **Quality Concerns**
   - Incomplete sentence fragments
   - Inconsistent formatting
   - Translation quality variations

### Implemented Solutions

#### 1. Duplicate Removal
  - Italian: 349,048 → 346,929 (-2,119 lines)
  - Japanese: 389,764 → 384,363 (-5,401 lines)

#### 2. Batch Processing
- Split into 1,000-line batches for manageable processing
- Enable systematic quality control

#### 3. Alignment Analysis Tools
- Visual inspection of translation pairs
- Identification of misalignment patterns



### Dataset Solutions (Priority Order)

#### Option 1: Alternative Datasets
- **English-Multilingual Approach**
  - Use English-Japanese + English-Italian datasets
  - Pivot through English for indirect translation
  - Higher quality, better-aligned datasets available

#### Option 2: Manual Alignment (Small Scale)
- **Targeted Correction**
  - Process 500-1,000 high-quality sentence pairs
  - Manual verification and alignment correction
  - Use as high-quality validation/test set

#### Option 3: Alternative Language Pairs
- **English-Italian Translation**
  - Better dataset availability
  - Similar linguistic families
  - Proof of concept for Romance language pairs

### Technical Implementation Plan

1. **Phase 1: Baseline LSTM Model**
   - Implement basic sequence-to-sequence architecture
   - Train on best-aligned subset of current data
   - Establish baseline BLEU scores

2. **Phase 2: Advanced Architectures**
   - Implement Transformer model
   - Compare performance with LSTM baseline
   - Optimize for computational efficiency

3. **Phase 3: Production Optimization**
   - Model compression and quantization
   - Real-time inference optimization
   - API development for practical usage

## Project Structure

```
jpn2ita/
├── data/
│   ├── raw/                    # Original corpus files
│   ├── processed/              # Cleaned and aligned data
│   └── split_files/            # Batched data files
├── tools/                      # Data processing scripts
├── models/                     # Model implementations
├── notebooks/                  # Jupyter notebooks for analysis
└── README.md                   # This file
```

## Resources

- **Morphological Analysis:** [Vibrato](https://github.com/daac-tools/vibrato)
- **BLEU Evaluation:** [SacreBLEU](https://github.com/mjpost/sacrebleu)
- **Model Framework:** PyTorch / TensorFlow
- **Cloud Platform:** Google Cloud Platform (GCP)

## Success Metrics

- **BLEU Score:** Target >20 for acceptable quality
- **Processing Speed:** <1 second per sentence inference
- **Dataset Quality:** >95% alignment accuracy on test set
- **Model Size:** <500MB for local deployment

---

**Note:** This is an ongoing research project focusing on the unique challenges of Italian-Japanese translation, particularly addressing morphological differences and dataset alignment issues.




