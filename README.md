# DECODING BRAIN SIGNALS AND BEYOND WITH GENERATIVE AI: INTRODUCING M4 - A FRAMEWORK FOR META-ALIGNED MULTI-MODAL MULTI-TASK LEARNING
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
1. EEG-TEXT MATCHING (ETM, Pretrained)
2. EEG-IMAGE MATCHING (EIM, Pretrained)
3. EEG-TEXT GENERATION (TEXT-GEN)
4. EEG-IMAGE GENERATION (IMG-GEN)
5. EEG-SENTIMENT CLASSIFICATION (SENT-CLASS)
6. EEG-IMAGE CLASSIFICATION (IMG-CLASS)
   
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
Below are some of results we present in our paper, which has not yet been released.
### Text-Based Tasks
<img width="536" alt="Screenshot 2024-01-17 at 16 30 26" src="https://github.com/MatthewS65537/M4/assets/136693480/096231e2-31ae-4b10-8a22-ac6066461040">
<img width="410" alt="Screenshot 2024-01-17 at 16 30 50" src="https://github.com/MatthewS65537/M4/assets/136693480/d7d009db-2ef5-4716-8082-4278cc19e7ad">


### Image-Based Tasks
<img width="536" alt="Screenshot 2024-01-17 at 16 31 14" src="https://github.com/MatthewS65537/M4/assets/136693480/4734b38a-5829-46b9-bb93-be9593f6adba">

### Generative Results
### Text Generation
<img width="511" alt="Screenshot 2024-01-17 at 16 33 17" src="https://github.com/MatthewS65537/M4/assets/136693480/894e6a13-40ef-4dc1-bb51-92a953f94679">

### Image Generation
<img width="492" alt="Screenshot 2024-01-17 at 16 31 21" src="https://github.com/MatthewS65537/M4/assets/136693480/f07c2eca-7045-4645-a03e-8d7645abfaa0">


### Downstream Applications of Our Tasks
#### EEG-Text Matching
![ETM-Demo-250](https://github.com/MatthewS65537/M4/assets/136693480/6aa43c3b-9f1c-4f3d-b240-88faee1948b2)

#### EEG-Image Matching
![EIM-Demo-250](https://github.com/MatthewS65537/M4/assets/136693480/aba891eb-635d-405c-bf4d-39971e457f14)
#### EEG-Text Generation
#### EEG-Image Generation
#### EEG-Sentiment Classification
![Sentiment-Classification-Demo-250](https://github.com/MatthewS65537/M4/assets/136693480/441a1549-f1f5-4cb8-8466-6d073542cb34)


#### EEG-Image Classification
![Image-Classification-Demo-250](https://github.com/MatthewS65537/M4/assets/136693480/fcca0f33-5c2c-486a-9c77-5eb6437f1dd4)


## Model Specifics
### Parameter Counts
#### Parameter Counts by Training Status
| Type | Count |
| - | - |
| Trainable | 104,394,283 |
| Pre-trained | 1,433,857,861 |
| Total | 1,538,252,144 |
#### Parameter Counts by Task
| Task | Active Parameters |
|---|---|
| EEG-Text Matching | 107,747,072 |
| EEG-Image Matching | 107,747,072 |
| EEG-Text Generation | 521,778,688 |
| EEG-Image Generation | 973,566,148 |
| EEG-Image Classification | 112,470,528 |
| EEG-Sentiment Classification | 117,459,971 |
### Training Descriptions
#### $M^4$ Simple Round Robin
|Epochs|Phase||
|---|---|---|
||Warmup (Pre-training Tasks)|N/A|
||Pre-training|N/A|
||Training|50|

#### $M^4$ DSG+
|Epochs|Phase||
|---|---|---|
||Warmup (Pre-training Tasks)|15|
||Pre-training|35|
||Training|50|
|**LR Decay** ($\gamma$)|||
||ETM|N/A|
||EIM|N/A|
||TEXT-GEN|0.5|
||IMG-GEN|0.5|
||SENT-CLASS|0.5|
||IMG-CLASS|0.5|

#### $M^4$ Layerwise
|Epochs|Phase||
|---|---|---|
||Warmup (Pre-training Tasks)|10|
||Pre-training|60|
||Training|50|
|**Pretrain LR Scaler**|||
||1-10|Linear 0.2 $\to$ 1.0|
||11-15|1.0|
||16-25|0.2|
||26-35|0.04|
||36-40|0.8|
||41-50|0.16|
||51-60|0.03|
||61-70|0.6|

### Training Hyperparameters
#### $M^4$ SRR
| Task Abbreviation | Learning Rate | Optimizer | Batch Size | Temperature |
| --- | --- | --- | --- | --- |
| ETM | 1e-5 | Adam | 256 | 20 |
| ETM | 1e-5 | Adam | 256 | 20 |
| TEXT-GEN | 5e-4 | Adam | 256 | N/A |
| IMG-GEN | 5e-4 | Adam | 32 | N/A |
| SENT-CLASSIFICATION | 5e-4 | Adam | 256 | 0.04 |
| IMG-CLASSIFICATION | 5e-4 | Adam | 256 | 0.04 |

#### $M^4$ DSG+
| Task Abbreviation | Learning Rate | Optimizer | Batch Size | Temperature |
|---|---|---|---|---|
| ETM | 5e-5 | Adam | 256 | 20 |
| EIM | 5e-5 | Adam | 256 | 20 |
| TEXT-GEN | 5e-5 | Adam | 256 | N/A |
| IMG-GEN | 5e-5 | Adam | 32 | N/A |
| SENT-CLASSIFICATION | 5e-5 | Adam | 256 | 0.04 |
| IMG-CLASSIFICATION | 5e-5 | Adam | 256 | 0.04 |

#### $M^4$ Layerwise
| Task Abbreviation | Learning Rate | Optimizer | $\beta_1$ | $\beta_2$ | $\epsilon$ | Weight Decay | $\gamma$ | Step Interval | Step Ratio | Batch Size | Temperature |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ETM | 5e-5 | Adam | Default | Default | Default | Default | N/A | N/A | 256 | 20 |
| EIM | 3e-5 | Adam | Default | Default | Default | Default | N/A | N/A | 256 | 20 |
| TEXT_GEN | 1.7e-5 | AdamW | 0.83 | 0.94 | 2.87e-6 | 1.76e-6 | 0.92 | 10 | 0.4 | 256 | N/A |
| IMG-GEN | 3e-5 | AdamW | 0.8 | 0.9 | 5e-7 | 0.01 | 0.9 | 5 | 0.3 | 32 | N/A |
| SENT-CLASSIFICATION | 1e-5 | AdamW | 0.85 | 0.97 | 6.7e-7 | 6.3e-7 | 0.912 | N/A | N/A | 256 | 0.04 |
| IMG-CLASSIFICATION | 5e-4 | AdamW | 0.9 | 0.999 | 6.7e-7 | 6.3e-7 | 0.95 | N/A | N/A | 256 | 0.04 |



## Acknowledgements
Some code from this repository has been borrowed from [EEG-To-Text](https://github.com/MikeWangWZHL/EEG-To-Text/tree/main) and thomwolf's [implementation](https://gist.github.com/thomwolf/7e2407fbd5945f07821adae3d9fd1312) of parallel optimizers and loss functions. A sincere thank you to their wonderful work.

This research has been conducted while I was working a research internship at Carnegie Mellon University.
The views and conclusions contained herein are mine and should not be interpreted necessarily as those
representing that of Carnegie Mellon University, either expressed or implied.

As a final note, I would like to express their thanks to Microsoft Research for their hardware, and Carnegie
Mellon University for their generous funding, which has been critical to ensuring the success of this project.
