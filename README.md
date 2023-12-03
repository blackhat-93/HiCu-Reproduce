# HiCu-Reproduce
This repo contains code for a reproduction of the MLHC 2022 paper [HiCu: Leveraging Hierarchy for Curriculum Learning in Automated ICD Coding](https://arxiv.org/abs/2208.02301).

Setup
-----
Follow the guidance text in `requirements.txt` to setup your environment and install the recommended packages. 

Data Preprocessing
-----
We use the MIMIC-III v1.4 dataset for model training and evaluation. 
Follow the guidance text in `/data/README.md` to download and preprocess the data.

Training
-----
If your local machine has at least 22Gb of RAM and 20Gb of GPU memory, you can train the models locally using step 1 & 2 below: 
1. For MultiResCNN and RAC models, run the appropriate .py file under `/runs`.
2. For LAAT (Bi-LSTM) models, switch to `LAAT` branch and use the training configs in the root folder.

If you want to train the models on the cloud using Google Colab, follow the steps below:
1. Create a folder called `HiCu-Reproduce` at the root of your Google Drive storage
2. Clone this repo into the `HiCu-Reproduce` folder from step 1
3. Copy your locally preprocessed data into the subfolder `/HiCu-Reproduce/data`
4. Run the file `Colab_Notebook.ipynb`
(Most models can be trained on the lowest-tier GPU. An exception is the RAC models, which may need multiple high-end GPUs)

Acknowledgement
-----
The majority of the code in this repository is borrowed from the original authors' repo [wren93/HiCu-ICD](https://github.com/wren93/HiCu-ICD), with some updates to improve reproducibility & facilitate model training on Google Colab. We greatly appreciate their high quality work.