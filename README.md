# Welcome to MindGarage

Hi Applicant, Thank you for considering the opportunity to join our team as a `student assistant (HiWi)`. We greatly appreciate the time and effort you're investing in this application process. As part of our selection procedure, we have designed a test task that will help us understand your skills and approach to problem-solving. Your performance on this task will play a crucial role in our decision-making process. We look forward to seeing your innovative solutions and learning more about your capabilities. Good luck, and thank you once again for your interest in working with us.

## Task: Feathered Gems - Classifying the Avian Treasures of the Indian Subcontinent
### Task Overview

Welcome to Project `Feathered Gems`, where we aim to leverage the power of `AI` to recognize and classify birds that are endemic to the Indian subcontinent. This project is not just about building an image classification model; it's about connecting technology with the rich biodiversity of one of the world's most vibrant ecological zones. Your role in this venture is to develop a model that can accurately identify different species of birds exclusive to the Indian subcontinent, using images as the primary data source.

### Objective
Your main objective is to `design` and `implement` an image classification model:

* This model must utilize the `vision encoder` of the Contrastive Language–Image Pre-training ([CLIP](https://arxiv.org/pdf/2103.00020.pdf)) model.

The CLIP model, developed by OpenAI, represents a significant advancement in the field of AI, as it can understand images in the context of natural language descriptions. This unique capability makes CLIP especially suitable for a task like ours, where the diversity of species can be vast and nuanced.

**Note: The vision encoder of CLIP was not designed for image classification**. Your task is to adapt it to image classification.

Hint: See how [ViT](https://arxiv.org/pdf/2010.11929.pdf) was trained for image classification. Also here is the codebase for CLIP [here](https://github.com/openai/CLIP)

Your adapted CLIP vision encoder should have the weights of the CLIP vision encoder. Then you should finetune this adpated model using the `train` set of the dataset below. Train for `10` epochs max. Then evaluate the performance on the `validation` set of the dataset.

Please utilize CLIP's `ViT` image encoder to build your adapted model, you can choose the smallest ViT model i.e: `ViT-B/32 or ViT-B/16`.

You are free to choose other hyper-parameters like batch-size, learning rate, schedular, optimizer etc, based on the computational restrictions of your system.


For your trained model, report the following metrics on the `validation` set:
1. Avg Accuracy
2. Classwise Accuracy
3. Confusion Matrix




### Dataset
You will train your model on the `Indian-Birds-Species-Image-Classification` dataset. Link [here](https://www.kaggle.com/datasets/ichhadhari/indian-birds/data).
The dataset consists of `25 bird species` found in India, including Asian Green Bee-eater, Brown-Headed Barbet, Cattle Egret, Common Kingfisher, Common Myna, Common Rosefinch, Common Tailorbird, Coppersmith Barbet, Forest Wagtail, Grey Wagtail, Hoopoe, House Crow, Indian Grey Hornbill, Indian Peacock, Indian Pitta, Indian Roller, Jungle Babbler, Northern Lapwing, Red-Wattled Lapwing, Ruddy Shelduck, Rufous Treepie, Sarus Crane, White-Breasted Kingfisher, White-Breasted Waterhen, and White Wagtail.

The dataset contains a total of `37,000` images split into train and validation sets in an `80:20` ratio, with `30,000` images in the training set and `7,500` images in the validation set. Each species has `1,500` images in the dataset. This dataset can be used for image classification tasks and to develop machine learning models to recognize different species of birds found in India.


### Evaluation Criterion
Note: **We do not want you to chase metrics like accuracy, recall, etc.**

Our goal is to assess your coding and problem-solving abilities specifically in the context of deep learning.
Your solution to the task will be judged on:
1. Ability to write modular code that is functional, clear, and readable.
2. Ability to problem solve, i.e., build a classification model out of a model that originally was not designed for image classification :p
3. Ability to convert your theoretical understanding and solutions into code.


### Codebase Template
We have provided a template for the codebase for the project. We expect your code to follow this template.

Here are some of the key features in this template:

1. We handle the configurations using `yacs`. In yacs, there is a base configuration that is defined in a python file called `cfg.py`. Then new configs can be made by listing the changes in a new `.yaml` file, and these will be merged with the base configuration.

2. The script `scripts/install.sh` builds a `venv` for the project using the requirements specified in the `requirements.txt` file.

3. To run the training and evaluation, we use `scripts/train.sh` and `scripts/eval.sh`.


```text
HIWI_TEST_TASK/
├── project007/
│   ├── configs/
│   │   └── default.yaml            # Write your configuration here, this file overrides the base config in cfg.py
│   ├── datasets/
│   │   └── __init__.py             # Write your PyTorch datasets here
│   ├── models/                     
│   │   └── __init__.py             # Write your PyTorch models here
│   ├── scripts/                    # Write your scripts to run training, evaluation here, basic ones are provided
│   │   ├── eval.sh
│   │   ├── install.sh
│   │   └── train.sh
│   └── utils/                      # Write your utility functions here
│       ├── __init__.py
│       └── build_logger.py
├── cfg.py                          # This is the base configuration file
├── eval.py                         # Script that runs the evaluation code
├── README.md                       # Readme file
├── requirements.txt                # The requirements file
└── train.py                        # Script that
```

### Contact Us


Should you have any questions or require further clarification regarding the task at hand, please do not hesitate to reach out. We are here to assist you every step of the way and ensure that you have all the information you need to succeed. Your understanding and comfort with the task are paramount to us, so let's collaborate to make this experience smooth and rewarding. Good luck, and we're looking forward to your creative solution!
