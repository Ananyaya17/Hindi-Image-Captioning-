## Image Captioning for Hindi in Low-Resource Settings

Benchmarking deep learning architectures for Hindi image captioning on Flickr8k Hindi, from a CNN-LSTM baseline to BLIP + LoRA.

Paper shortlisted (top 25% of submissions) at ICDAM-2026 (Elsevier SSRN). Department of CSE, Indira Gandhi Delhi Technical University for Women. Supervisor: Dr. Jagrati Singh.

## Research Questions
  Q1. Do transformer-based architectures outperform CNN-LSTM for Hindi captioning?
  
  Q2. What is the quantitative impact of beam search on BLEU scores?
  
  Q3. Which architectural elements most influence performance in low-resource Hindi?
  
  Q4. What are the failure modes of BLIP + LoRA fine-tuned on Hindi?

## Repository Contents
Each notebook is self-contained and runs on Google Colab (T4 GPU).

| Notebook | Model(s) | Framework |
|---|---|---|
| `base_attn_trans_comparison.ipynb` | Base: InceptionV3 + LSTM; Attn: + Bahdanau attention; Trans: InceptionV3 + Transformer decoder | TensorFlow / Keras |
| `Inception + LSTM.ipynb` | InceptionV3 global features + LSTM + LayerNorm, with greedy and beam search decoding | TensorFlow / Keras |
| `a hybrid CNN + multi-head self-attention + Bahdanau attention model .ipynb` | Multi-head self-attention over image regions + BiLSTM + two Bahdanau attention layers (global and top-region) | TensorFlow / Keras |
| `InceptionV3 + Multi-Head Attention + BiLSTM + Bahdanau Attention.ipynb` | 8-head image self-attention + BiLSTM + Bahdanau cross-attention, concatenation fusion, greedy and beam search (B=5) | TensorFlow / Keras |
| `BLIP + LoRA + Hindi projection head.ipynb` | BLIP (ViT + BERT decoder) + LoRA (r=16) + Hindi projection head | PyTorch / Hugging Face |

All TensorFlow models use frozen InceptionV3 features (8x8x2048, reshaped to 64x2048 regions) unless noted.

## Dataset

- Flickr8k Hindi : 8,091 images with about 40,000 Hindi captions (5 per image).

Notebooks	Split (train / val / test)
TensorFlow models	70 / 15 / 15
BLIP + LoRA	80 / 10 / 10

The dataset and trained weights are not included in this repo.

## Results

Corpus BLEU on the Flickr8k Hindi test set.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| M1: CNN-LSTM baseline | 0.557 | 0.359 | - | - |
| M2: Inception-LSTM + LayerNorm | 0.553 | 0.331 | 0.182 | 0.097 |
| M3: Hybrid (greedy) | 0.478 | 0.293 | - | 0.113 |
| M4: Hybrid + beam search | 0.504 | 0.316 | 0.205 | 0.133 |
| M5: BLIP + LoRA + Hindi head | 0.188 | 0.056 | 0.022 | 0.011 |

## Key Findings
1. Simpler is better with small data. A ~15M-parameter CNN-LSTM beat the larger attention hybrid on BLEU-1 and BLEU-2, which overfit the small training set.
2. Beam search is free. Moving the hybrid from greedy to beam search improved BLEU-1 by 5.5%, BLEU-2 by 7.8% and BLEU-4 by 17.7%, with no retraining.
3. BLEU-1 and BLEU-4 rankings differ. Choose the metric that fits the application.
4. English-pretrained BLIP fails on Hindi. Generated captions lose Devanagari vowel signs and virama marks (for example "तस्वीर में" becomes "तसवीर म"), pointing to a tokenizer mismatch between BLIP's English BERT tokenizer and Devanagari.

## Known Limitations
1. In the BLIP notebook, the Hindi projection head is defined and saved but is not used in the loss or generation. Training uses BLIP's own language-model loss.
2. BLIP training has two stages: LoRA on a frozen vision encoder, then the LoRA weights are merged and the full model is fine-tuned.
3. The BLIP and TensorFlow notebooks use different splits, so test sets differ.
4. BLEU is computed on a sample of test images (500 in most notebooks), not the full test set.
5. Beam width and n-gram blocking differ between notebooks. Check each notebook for exact values.

## How to Run
1. Download Flickr8k Hindi and upload it as data.zip to the root of your Google Drive.
2. Open a notebook in Colab and select Runtime, Change runtime type, T4 GPU.
3. Run the cells in order. InceptionV3 feature extraction takes about 12 to 15 minutes.
4. For the BLIP notebook, run the install cell, restart the runtime, then continue from the second cell.

Open In Colab Base / Attn / Trans

Open In Colab Inception + LSTM

Open In Colab Hybrid (dual attention)

Open In Colab Hybrid + beam search

Open In Colab BLIP + LoRA

## Tech Stack

Python, TensorFlow/Keras, PyTorch, Hugging Face Transformers, PEFT (LoRA), NLTK, InceptionV3.

## Future Work
1. Multilingual pretraining (BLIP-style models on Hindi + English)
2. Hindi-specific tokenization (BPE / SentencePiece for Devanagari)
3. Data augmentation (back-translation, image augmentation, pseudo-labeling)
4. SCST reinforcement learning to optimize BLEU directly
5. Hindi video captioning and VQA

## Authors
Aditi, Aditi Chhikara, Ananya Kumar, Archita Gupta
Supervised by Dr. Jagrati Singh, Assistant Professor, CSE, IGDTUW.
