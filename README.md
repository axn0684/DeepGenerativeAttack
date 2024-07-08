# Deep Generative Attack on Offline Signature verification

## Paper Overview

This is the code for IJCB 2024 paper Deep Generative Attacks and Countermeasures for Data-Driven Writer-Dependent Static Signature Verification.

### Research Purpose

This study explores the use of modern Deep Generative Models (DGMs) for both challenging ASV systems and crafting effective countermeasures. Our research focuses on:

1. Creation of synthetic datasets using VAE and CGAN, derived from CEDAR, BHSig260-B, and BHSig260-H collections.
2. Evaluation of the resilience of state-of-the-art DASV models, including DenseNet201, ResNet152V2, and Xception architectures, against various attack scenarios.
3. Development of a novel countermeasure: retraining DASV models with SSIM-optimized synthetic forgeries.

### Significance of the Research

This study makes several important contributions to the field of signature verification and biometric security:

- Our research provides a comprehensive assessment of deep learning-based signature verification systems' robustness against a wide range of attacks, which is crucial for understanding the true security landscape of these systems.

- We discovered a strong negative correlation between the structural similarity (SSIM) of generated forgeries and false acceptance rates (FARs) allows for better control over the quality of generated forgeries, enabling the development of more effective attacks and, crucially, more robust countermeasures.

- The retraining strategy we present, using synthetic forgeries, significantly enhances the robustness of baseline models, demonstrating a practical path forward in defending against sophisticated attacks.

These contributions collectively advance our understanding of the vulnerabilities in current signature verification systems and provide concrete strategies for improving their security. The insights gained from this research have broad implications for the development of more robust biometric security systems in the future and applicability to other fields.

### Real-World Applications Beyond DASV

While our research focuses on signature verification, the implications extend to various domains:

- Biometric Security: The techniques developed could be applied to other biometric verification systems, such as facial recognition or fingerprint scanning.
- Document Verification: Enhancing the security of legal, financial, and medical document authentication processes.
- Digital Identity Protection: Improving safeguards against identity theft and fraud in online transactions.

## Code execution

### Environment requirements

To install the environment requirements, please run:

```shell
pip install -r setup.py
```

### Dataset

As discussed in the paper, we used CGAN and VAE to generate synthetic signatures. The codes included in this repository is made for the CEDAR dataset from https://www.kaggle.com/datasets/shreelakshmigp/cedardataset. Downloaded dataset should be placed in the current directory.

### Synthetic Image Generation

The datasets are preprocessed using OTSU binarializing as put in file utils.py.

To start image generation process, please run:

```py
python CGAN\ generation.py # Using CGAN
python VAE\ generation.py  # Using VAE
python VAE\ SSI.py   # Using VAE-SSI controlled
```

For controlled SSIM generation with VAE as discussed in the paper, see the VAE SSI file.

### Preprocessing and training/testing data for DASV baselines

Datasets used to train and test the DASV baselines include:

- Original CEDAR dataset: [link](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
- Synthetic dataset: Generate from the above section or download from [link](https://drive.google.com/drive/folders/1KbbJ5pCx9CVjlFgt28j4bY9KaFcELHb_). The generated data folders got 9 generated images for each forgery signatures from CEDAR.

To preprocess the data, run the file preprocess_data.py.

After running the files, the user should get 3 numpy files containing the processed real, forg and generated data as well as the average SSIM score for the generated data.

Follow the instructions from the file DenseNet201_single.ipynb to run the baseline model DenseNet201 and test it with the generated attack. For the testing with retrained, see the demo in DenseNet201_cgan_retrain_cgan file.ipynb. Similar preprocessing, training/testing protocols are done with other datasets, generated datasets and model architectures.

### Synthetic data from BHSIG260 and CEDAR datasets links:

The synthetic data included in the paper are located in [link](https://drive.google.com/drive/folders/1KbbJ5pCx9CVjlFgt28j4bY9KaFcELHb_).

# Citations

```
@inproceedings{ngo2024dgac,
    title={Deep Generative Attacks and Countermeasures for Data-Driven Offline Signature Verification},
    author={Ngo, An and Kumar, Rajesh and Cao, MinhPhuong},
    booktitle={2024 IEEE International Joint Conference on Biometrics (IJCB 2024)},
    year={2024},
    organization={IEEE}
}
```
