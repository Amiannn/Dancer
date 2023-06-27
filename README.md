# DANCER💃: Entity Description Augmented Named Entity Corrector for Automatic Speech Recognition
Implementation of Entity Description Augmented Named Entity Corrector for Automatic Speech Recognition.

## Getting Started

### Dependency / Install

(This work was tested with PyTorch 1.7.0, CUDA 11.7, python 3.8 and Ubuntu 20.04.)

- Install [PyTorch](https://pytorch.org/get-started/locally/)

- `$ pip install -r requirements`


## Scripts

```
$ git clone https://github.com/Amiannn/NameEntityCorretor.git
```

### Prediction

```bash
$ python3 -m entity_correction \
    --asr_transcription_path "./datas/aishell_test_set/asr_transcription/conformer/hyp"         \
    --asr_nbest_transcription_path "./datas/aishell_test_set/asr_transcription/conformer/nbest" \
    --asr_manuscript_path "./datas/aishell_test_set/ref"                                        \
    --entity_path "./datas/entities/aishell/test_1_entities.txt"                                \
    --entity_content_path "./datas/entities/aishell/descriptions"                               \
    --entity_vectors_path "./datas/entities/aishell/descriptions/embeds.npy"                    \
    --detection_model_type "bert_detector"                                                      \
    --detection_model_path "./ckpts/ner/best_model"                                             \
    --retrieval_model_type "prsr_retriever"                                                     \
    --retrieval_model_path "./ckpts/ranker/dpr_biencoder.39"                                    \
    --use_rejection "True"
```

### Train NER Model

For example, we train models on Aishell dataset as follows:

```
$ 
$ 
```

### Train Semantic Ranking Model

For example, we train models on Aishell dataset as follows:

```
$ 
$ 
```

### Evaluation

```
$
```

## Checkpoints
- Download [Google-Drive](https://pytorch.org/get-started/locally/)