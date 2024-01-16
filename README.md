# MMMM
## Note
```diff
- This repository is currently under development. All code is subject to final changes, and the author does not guarantee that the code is organized in any way, or immediately runnable.
```
This is an official implementation of M4, a universal **M**eta-Aligned **M**ulti-**M**odal **M**ulti-Task (4M) framework capable of training any model for the 4M setting with the correct modifications. This current example implementation of M4 is set for the task of decoding EEG signals.

## Abstract
Since the days of Turing, a longstanding goal of artificial intelligence has been to understand the
intricate workings of the human brain. As such, we explore the task of decoding EEG signals
using a 4M paradigm - one that is Meta learned, Multi-Modal, and Multi-task. While state-of-the-art
(SOTA) Large Language Models (LLMs) such as LLaMA-2 and GPT-4 have recently been augmented
with some multimodal capabilities alongside their formidable Natural Language Processing (NLP)
performance, they do not readily fit into a true 4M setting.
Unlike GPT-4 which employs DALL-E 3 in a somewhat plug-and-play fashion, we utilize cross-modal
alignment techniques to more tightly integrate multiple modalities into a unified 4M framework.
Through comprehensive benchmarking, we address alignment method limitations around noisy data,
low-resource tasks, and generalizability. We further assemble a novel EEG analysis dataset by
consolidating several existing ones.
Based on curated data, we develop an encompassing 4M training paradigm and employ it to construct
$M^4$ - our Meta-aligned Multi-Modal Multi-task Model. As a tri-modal architecture with 1.5 billion
parameters, $M^4$ is capable of encoding text, image, and EEG inputs to fulfill numerous downstream
prediction tasks. We enhance the training process using a learning rate modulation technique to
promote balanced multi-task abilities.
In the experiment, our training data is derived from ZuCoV1.0/2.0 and Brain2Image. We also build
a new downstream benchmark focused on cross-modal EEG decoding, and image generation from
EEG, among other proxy tasks. In addition to the conventional approach of learning single-purpose
adapter modules per task, we also experiment with meta-adapters to impart meta-learning capacities
to $M^4$. Finally, we discuss limitations, societal implications, and upcoming research for our model

## Environment Setup
To set up the environment, run either `sh pip_install.sh` or the following command:
```bash
pip install -q scipy
pip install -q h5py
pip install -q matplotlib
pip install -q torch torchvision torchaudio
pip install -q transformers
pip install -q fuzzy_match
pip install -q nltk
pip install -q rouge
pip install -q diffusers
pip install -q tensorboard
pip install -q tqdm
pip install -q accelerate
pip install -q optuna optuna-dashboard
```
Our project relies on CUDA at the moment, so please also take time to either modify your code to use the hardware you have available, and/or ensure that your cuda driver and the current pytorch version do not clash. For the code to run with the settings described within the paper, you will require at least 4 x NVIDIA A100 80GB GPUs.

## Data Preparation
The data of our code is preprocessed and can be loaded via the command `sh LoadItemsGdrive.sh`. However, the command may not work in some environments, so you may wish to navigate to the DreamDiffusion and/or EEG-To-Text Githubs and follow their instructions there. After that is done, please also run `sh preprocess/preprocess.sh` to preprocess some of the necessary data.

## Task List
Below is a list of the tasks that we have implemented:
1. EEG-TEXT MATCHING
2. EEG-IMAGE MATCHING
3. EEG-TEXT GENERATION
4. EEG-IMAGE GENERATION
5. EEG-SENTIMENT CLASSIFICATION
6. EEG-IMAGE CLASSIFICATION
   
## Train
### With Pretrain
1. Modify the code within the necessary files to suit your needs.
2. Run `python3 ./training/pretrain.py` to start the pretraining.
3. Run `python3 ./training/continue_from_pretrain.py` to start training on the other tasks.
### Without Pretrain
1. Modify the code within the necessary files to suit your needs.
2. Run `python3 ./training/training.py` to start the pretraining.

## Evaluate
1. Modify the code within the necessary files to suit your needs.
2. Run `python3 ./evaluator/{task_name}.py` to start the pretraining where `{task_name}` is the task you wish to evaluate.

## Results
### Text-Based Tasks

### Image-Based Tasks

### Generative Results

### Demos
#### EEG-Text Matching
#### EEG-Image Matching
#### EEG-Text Generation
#### EEG-Image Generation
#### EEG-Sentiment Classification
#### EEG-Image Classification

## Acknowledgements
Some code from this repository has been borrowed from EEG-To-Text and (Pytorch Parallel? Check which repo/forum thread). A sincere thank you to their wonderful work.

This research has been conducted while I was working a research internship at Carnegie Mellon University.
The views and conclusions contained herein are mine and should not be interpreted necessarily as those
representing that of Carnegie Mellon University, either expressed or implied.

As a final note, I would like to express their thanks to Microsoft Research for their hardware, and Carnegie
Mellon University for their generous funding, which has been critical to ensuring the success of this project.
