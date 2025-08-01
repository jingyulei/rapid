# RAPID: Recognition of Any-Possible DrIver Distraction via Multi-view Pose Generation Models 🚗

<p align="center">
  <img src="https://github.com/user-attachments/assets/4eb99116-44dc-4c2c-89af-38660736708e" width="600"/>
</p>

## 🧠Contributions
✨Traditional DMS solutions rely on post-processing procedure to improve detection ability. However, RAPID possesses a greater ability to judge with an end-to-end and frame-level prediction.

✨We utilize DDPM to generate possible future driver poses and determine whether the driver is distracted by clustering, which enables recognition of undefined actions.

✨In order to be put into practice, privacy protection is a problem to be solved. Based on human pose keypoints, RAPID could not only protect drivers' privacy but also support rapid inference.

## Our dataset: sktDD  🤳 💬 🍔 ☕
During the experiment, in order to recognize any driver distraction behavior that is not predefined, we design a variety of normal and abnormal driving behaviors. The normal driving behaviors include not only mechanical operations with both hands on the steering wheel but also permissible non-distracting actions such as adjusting glasses and changing posture. As for abnormal driving, we design at least ten different behaviors, as shown in the following table.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fcd6338d-ebc4-49ea-81ca-2b2603703bb9" width="524"/>
</p>

Our original dataset is uploaded in folder `original_sktDD` whose single file includes one driver's one view. In order to reproduce our results, folder `sktDD` can be utilized directly. In our dataset, column **PersonID** means different views (0/1/2 means rearview mirror/passenger-side window/dashboard view). 

Based on the assumption of unsupervised learning, our training set only contains normal driving multi-view pose keypoints, while the test set includes both normal driving and distracted driving, with the labels stored in `test_frame_mask`.


## Usage 
### ⚙️Setup

```bash
conda env create -f environment.yaml
conda activate rapid
```
### 🚀Training 
```bash
python train_RAPID.py --config train.yaml
```
Past frame number **k** can be changed in `train.yaml` (discussion in III.A).
### 🧪Testing
- Testing your own training results

  Fill in `load_ckpt` in `checkpoints/sktDD/train_experiment/config.yaml` and run:
```bash
python test_RAPID.py --config checkpoints/sktDD/train_experiment/config.yaml
```

- Reproducing our result

  Run:
```bash
python test_RAPID.py --config test.yaml
```
- You can view the result images in the directory `./pictures`.

## 📚References
We referenced the repos below for the code.

[MoCoDAD](https://github.com/aleflabo/MoCoDAD)
