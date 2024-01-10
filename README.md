# MMMM
Blah Blah Blah

## Code Explanation
Our code is structured as follows:
- `Colab Testing Notebooks/`: This directory contains Jupyter notebooks for testing.
- `ZuCo/`: This directory contains the ZuCo data processing script.
- `models/`: This directory contains the model definitions.
- `testing/`: This directory contains the testing scripts.
- `training/`: This directory contains the training scripts.
- `trainer/`: This directory contains the separate trainers and evaluators (if applicable) for each task.
- `tuning/`: This directory contains the tuning scripts.
- `utils/`: This directory contains utility scripts, such as the dataloaders, initialization functions, etc.
- `preprocess/`: This directory contains some preprocessing scripts.

## How to Run?
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run `sh pip_install.sh` to install the necessary dependencies.
4. Run `sh LoadItemsGdrive.sh` to install the files. If you don't find anything please check the file to ensure that all lines are uncommented the first time you run it through.
5. Run `sh preprocess/preprocess.sh` to preprocess some of the necessary data.
6. Run `python3 ./training/train.py` to start the training.
7. By commenting out some of the add_task() lines, you can choose train on only a few select tasks.

## Citation
If you use our software in your research, please cite our paper as follows:

```bibtex
@article{author2022MMMM,
  title={Title of the paper},
  author={Author, A. and Coauthor, B.},
  journal={Journal name},
  year={2022}
}
