# Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model

**[CVPR2022] Official Pytorch based implementation.** 

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal)

<hr />

> **Abstract:** *In this paper, an ill-posed problem of multiple adverse weather removal is investigated. Our goal is to train a model with a 'unified' architecture and only one set of pretrained weights that can tackle multiple types of adverse weathers such as haze, snow, and rain simultaneously. To this end, a two-stage knowledge learning mechanism including knowledge collation (KC) and knowledge examination (KE) based on a multi-teacher and student architecture is proposed. At the KC, the student network aims to learn the comprehensive bad weather removal problem from multiple well-trained teacher networks where each of them is specialized in a specific bad weather removal problem. To accomplish this process, a novel collaborative knowledge transfer is proposed. At the KE, the student model is trained without the teacher networks and examined by challenging pixel loss derived by the ground truth. Moreover, to improve the performance of our training framework, a novel loss function called multi-contrastive knowledge regularization (MCR) loss is proposed. Experiments on several datasets show that our student model can achieve promising results on different bad weather removal tasks simultaneously.* 
## Architecture

<table>
  <tr>
    <td colspan="2" align="center"> <img src = "https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal/blob/main/images/architecture.png"> </td>
  </tr>
  <tr>
    <td colspan="2" align="center"><p><b>Overall Architecture</b></p></td>
  </tr>
  <tr>
    <td align="center"> <img src = "https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal/blob/main/images/ckt.png" width="400"> </td>
    <td align="center"> <img src = "https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal/blob/main/images/mcr.png" width="400"> </td>
  </tr>
  <tr>
    <td align="center"><p><b>Collaborative Knowledge Trasfer</b></p></td>
    <td align="center"><p><b>Multi-contrastive Regularization</b></p></td>
  </tr>
</table>

## Quantitative Result

![image](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal/blob/main/images/quantitative_result.png)

## Qualitative Result

![image](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal/blob/main/images/qualitative_result.png)

## Usage

#### Trained Model

* **[Setting1] [CSD, Rain1400, ITS, OTS]** Download link: [TBA](https://github.com/fingerk28)
* **[Setting2] [Snow100k, Raindrop, Rainfog]** Download link: [TBA](https://github.com/fingerk28)

#### Install

```sh
git clone https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal.git
```

#### Requirement

```shell
pip install -r requirements.txt
```
#### Training

```shell
python main.py --train --save-dir CHECKPOINT_DIR 
```
#### Inference

```shell
python main.py --test --checkpoint CHECKPOINT_PATH
```

## Citation
Please cite this paper in your publications if it is helpful for your tasks.
```bib
@inproceedings{Chen2022MultiWeatherRemoval,
  title={Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model},
  author={Chen, Wei-Ting and Huang, Zhi-Kai and Tsai, Cheng-Che and Yang, Hao-Hsiang and Ding, Jian-Jiun and Kuo, Sy-Yen},
  journal={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```


