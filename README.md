# Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model

**[CVPR2022] Official Pytorch based implementation.** 

[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.pdf)

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

<table>
  <tr>
    <td align="center"> <img src = "https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal/blob/main/images/quantitative_result.png" width="400"> </td>
  </tr>
  <tr>
    <td align="center"><p><b>Setting1</b></p></td>
  </tr>
</table>

## Qualitative Result

<table>
  <tr>
    <td align="center"> <img src = "https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal/blob/main/images/qualitative_result.png"> </td>
  </tr>
  <tr>
    <td align="center"><p><b>Setting1</b></p></td>
  </tr>
</table>


## Usage

#### Pre-trained Models

* **[Setting1] [CSD, Rain1400, ITS, OTS]** Download link: 
  * **CSD Teacher** &rarr; [Google Drive](https://drive.google.com/file/d/12IiwMeWI6Li5USrtUeaRCrqkHYmPBdKE/view?usp=sharing)
  * **Rain1400 Teacher** &rarr; [Google Drive](https://drive.google.com/file/d/11Z7_0awGLbFtOzbi2Mzra0Cz1q8aEfg1/view?usp=sharing)
  * **ITS, OTS Teacher** &rarr; [Google Drive](https://drive.google.com/file/d/1eCJ47fYdcuirqZW7xHhu0SgoX_pre2D5/view?usp=sharing)
  * **Student** &rarr; [Google Drive](https://drive.google.com/file/d/1TP2IFPDJYlNnV2QJO6_2UrAmoY6IOFpQ/view?usp=sharing)

* **[Setting2] [Snow100k, Raindrop, Rainfog]** Download link:
  * **Raindrop Teacher** &rarr; [Google Drive](https://drive.google.com/file/d/1Pyoy7-sjCwp_bMcJ8foBcIy9u4G0EwLQ/view?usp=sharing)
  * **Rainfog Teacher** &rarr; [Google Drive](https://drive.google.com/file/d/17Sub3v3fw8DxjG9EyyqAQNum1YQb1SiA/view?usp=sharing)
  * **Snow100k Teacher** &rarr; [Google Drive](https://drive.google.com/file/d/1UdjT_3QOcjlUsQ1fG8hMjBwuty9Zn-nH/view?usp=sharing)
  * **Student** &rarr; [Google Drive](https://drive.google.com/file/d/16Ux9UPCxw6M8tkoHJFrLaRPqUJaPVZAL/view?usp=sharing)

#### Install

```sh
git clone https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal.git
```

#### Training

```shell
python train.py --teacher TEACHER_CHECKPOINT_PATH_0 TEACHER_CHECKPOINT_PATH_1 TEACHER_CHECKPOINT_PATH_2 --save-dir RESULTS_WILL_BE_SAVED_HERE
```
> **--teacher** &rarr; input any amout of teacher checkpoint path
>
> ---
>
> You need to prepare the meta file (.json) under the `./meta`
>
> Class `DatasetForTrain` and `DatasetForValid` would take all meta files as the datasources.
>
> The structure should be:
>
> ```
> .
> ├── inference.py
> ├── meta
> │   ├── train
> │   │   ├── CSD_meta_train.json
> │   │   └── Rain1400_meta_train.json
> │   │   └── ...
> │   └── valid
> │       ├── CSD_meta_valid.json
> │       └── Rain1400_meta_valid.json
> │       └── ...
> ├── models
> │   ├── ...
> ├── train.py
> └── utils
>     ├── ...
> 
> ```
>
> ---
>
> The structure of the `.json` file should be:
>
> ```
> [
>     [
>         "path_to_GT_image0",
>         "path_to_Input_image0"
>     ],
>     [
>         "path_to_GT_image1",
>         "path_to_Input_image1"
>     ],
>     [
>     		 ...
>     ],
>     ...
>     
> ]
> ```

#### Inference

```shell
python inference.py --dir_path DIR_OF_TEST_IMAGES --checkpoint CHECKPOINT_PATH --save_dir RESULTS_WILL_BE_SAVED_HERE 
```



## Other Works for Image Restoration

You can also refer to our previous works:

* Desnowing &rarr; [[JSTASR]](https://github.com/weitingchen83/JSTASR-DesnowNet-ECCV-2020) (ECCV'20) and [[HDCW-Net]](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet) (ICCV'21)
* Dehazing &rarr; [[PMS-Net]](https://github.com/weitingchen83/PMS-Net) (CVPR'19) and [[PMHLD]](https://github.com/weitingchen83/Dehazing-PMHLD-Patch-Map-Based-Hybrid-Learning-DehazeNet-for-Single-Image-Haze-Removal-TIP-2020) (TIP'20)
* Deraining &rarr; [[ContouletNet]](https://github.com/cctakaet/ContourletNet-BMVC2021) (BMVC'21)
* Image Relighting &rarr; [[MB-Net]](https://github.com/weitingchen83/NTIRE2021-Depth-Guided-Image-Relighting-MBNet) (NTIRE'21 1st solution) and [[S3Net]](https://github.com/dectrfov/NTIRE-2021-Depth-Guided-Image-Any-to-Any-relighting) (NTIRE'21 3 rd solution)



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

