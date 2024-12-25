# RAPID: Recognition of Any-Possible DrIver Distraction via Multi-view Pose Generation Models

<p align="center">
  <img src="https://github.com/user-attachments/assets/4eb99116-44dc-4c2c-89af-38660736708e" width="700"/>
</p>

## Contributions
- Traditional DMS solutions rely on post-processing procedure to improve detection ability. However, RAPID possesses a greater ability to judge with an end-to-end and frame-level prediction.
- We utilize DDPM to generate possible future driver poses and determine whether the driver is distracted by clustering, which enables recognition of undefined actions.
- In order to be put into practice, privacy protection is a problem to be solved. Based on human pose keypoints, RAPID could not only protect drivers' privacy but also support rapid inference.

## sktDD Dataset
Our original dataset is uploaded in folder **original_sktDD** whose single file includes one driver's one view. In order to reproduce our results, folder **sktDD** can be utilized directly. In our dataset, column **PersonID** means different views (0/1/2 means rearview mirror/passenger-side window/dashboard view). 


## Usage 
### Setup

```bash
conda env create -f environment.yaml
conda activate rapid
```
### Train
```bash
python train_RAPID.py --config train.yaml
```
### Test
```bash
python test_RAPID.py --config test.yaml
```


