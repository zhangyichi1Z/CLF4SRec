# CLF4SRec
## Model
Our model CLF4SRec is implemented based on the [RecBole](https://github.com/RUCAIBox/RecBole). 

Both the processing of the dataset and the metrics calculation follow the implementation of RecBole.

CLF4SRec in /recbole/model/sequential_recommender/clf4srec.py
### Usage
We provide the script main_run.py to run the model
## Citation
If you use this code, please cite the pape

@article{ZHANG2023110481,
title = {Contrastive Learning with Frequency Domain for Sequential Recommendation},
journal = {Applied Soft Computing},
pages = {110481},
year = {2023},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2023.110481},
url = {https://www.sciencedirect.com/science/article/pii/S1568494623004994},
author = {Yichi Zhang and Guisheng Yin and Yuxin Dong and Liguo Zhang},
keywords = {Sequential recommendation, Contrastive learning, Frequency-domain augmentation, Recommender systems},
abstract = {Sequential recommendation has recently played an important role on various platforms due to its ability to understand users’ intentions from their historical interactions. However, modeling user’s intention on time-based representations poses challenges, such as fast-evolving interests, noisy interactions, and sparse data. While contrastive self-supervised learning can mitigate these issues, the complexity of time-based interactions limits the utility of understanding the user’s intention. Motivated by this limitation, we posit that intent representations need to accommodate different domains. To this end, we expect both the frequency-domain augmented view and the time-domain augmented view of the same sample should be maximally consistent with their original input in their corresponding domains. Inspired by this idea, we propose Contrastive Learning with Frequency Domain for Sequential Recommendation (CLF4SRec), where a learnable Fourier layer provides the frequency-based self-supervised signal.Instead of pre-training, we employ a multi-task learning framework jointly with contrastive learning and recommendation learning to optimize the user representation encoder. We conduct comprehensive experiments on four real-world datasets, where CLF4SRec outperforms the recent strongest baselines from 7.58% to 67.85%, showing its effectiveness for sequential recommendation tasks. Specifically, CLF4SRec can achieve a boost of 41.88% to 67.85% on the dense dataset, which might be attributed to the ability of frequency domain technology to handle dense signals. The implementation code is available at https://github.com/zhangyichi1Z/CLF4SRec.}
}
# Credit
This repo is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [DuoRec](https://github.com/RuihongQiu/DuoRec)
