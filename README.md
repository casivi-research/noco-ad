# A Norm-Constrained Method for Semantic and Structured Anomaly Detection Tasks
## Reproduce our experiment

1. Config the environment
```
conda create -n noco python=3.8 -y

conda activate noco

conda install cudatoolkit==11.3.1  cudnn -y

pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install opencv-python scikit-learn scikit-image matplotlib albumentations einops kornia tqdm pandas
```
2. For the Structured AD tasks, please run the train_mvtec.py. MVTec AD, MPDD and VisA datasets are evaluated based on this script.
3. For the Semantic AD tasks, please run the train_cifar10.py (cifar10, cifar100 and cat_vs_dog datasets) and train_mnist.py (mnist and f-mnist datasets).


## Abstract
Benefiting from complex modeling processes, mainstream anomaly detection (AD) methods have achieved remarkable results on specific tasks. However, in the real world, especially within industrial inspection scenarios, tasks such as Semantic AD and Structured AD often coexist. Data from different AD tasks carry distinct attributes and follow different distributions. Such discrepancy makes it hard to design an anomaly detection method that can achieve excellent performance across various AD tasks. To address this issue, this paper proposes a simple and effective anomaly detection method, capable of achieving excellent performance in both Semantic AD and Structured AD tasks. Such good performance can be attributed to a key discovery in this study: \textit{Constraining the norm of the optimization objective within the optimal range helps enhance the performance of the anomaly detector.} Therefore, by constraining the norm within the shared optimal range across various AD tasks, the performance in both Semantic AD and Structured AD tasks can be improved. Extensive experiments validate the correctness of the proposed method. In addition, benefiting from the data-independent initiation and the fewer trainable parameters, the proposed method can be flexible to meet cold-start requirements in online AD scenarios.

## Method
![image]( ./figures/method.png)

## Performance
![image]( ./figures/figure1.png)
![image]( ./figures/two_ad.png)
![image]( ./figures/mvtec.png)
![image]( ./figures/norm.png)