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

## How to Run?
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run `sh pip_install.sh` to install the necessary dependencies.
4. Run `python3 ./training/train.py` to start the training.
5. By commenting out some of the add_task() lines, you can choose train on only a few select tasks.

## Citation
If you use our software in your research, please cite our paper as follows:

```bibtex
@article{author2022MMMM,
  title={Title of the paper},
  author={Author, A. and Coauthor, B.},
  journal={Journal name},
  year={2022}
}
