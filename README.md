# Recipe for VoicePrivacy Challenge 2024 
(Under development)

## Install

1. `git clone -b vpc  https://github.com/DigitalPhonetics/VoicePAT.git`
2. `./00_install.sh`
3. `source env.sh`

## Download data and pretrianed models

`./01_download_data_model.sh` Password required, please register to get password.

### B2 Anonymization + Evaluation 
If you would like to generate B2 audio and evaluate it without modifying, or submit Python scripts separately, simply run

`./02_run.sh`

then you can get the results for B2.

## Using Anonymization and Evaluation Flexibly 
The recipe uses [VoicePAT](https://github.com/DigitalPhonetics/VoicePAT) toolkit, consists of **two separate procedures for anonymization and evaluation**. This means that the generation of anonymized speech is independent of the evaluation of anonymization systems. Both processes do not need to be executed in the same run or with the same settings. 

### Anonymization: 
The recipe supports B2 and [GAN-based](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10096607) speaker anonymization systems.
#### B2: Anonymization using McAdams coefficient (randomized version)
This is the same baseline as the secondary baseline for the VoicePrivacy-2022. It does not require any training data and is based upon simple signal processing techniques using the McAdams coefficient.
You may modify entry `$results_dir` in [config/anon_dsp.yaml](https://github.com/DigitalPhonetics/VoicePAT/blob/vpc/configs/anon_dsp.yaml), default value is `exp/dsp_anon`
```
python run_anonymization_dsp.py --config anon_dsp.yaml
```
The anonymized audios will be saved to `$results_dir`, including 7 folders:

```
  $results_dir/libri_dev_enrolls/*wav
  $results_dir/libri_dev_trials_m/*wav
  $results_dir/libri_dev_trials_f/*wav

  $results_dir/libri_test_enrolls/*wav
  $results_dir/libri_test_trials_m/*wav
  $results_dir/libri_test_trials_f/*wav

  $results_dir/train-clean-360/*wav
```

#### GAN-based: Anonymization using Transformer-based ASR, FastSpeech2-based TTS and WGAN-based anonymizer.
All the pretrained models downloaded by 01_download_data_model
You may modify entry `$results_dir` in [config/anon_ims_sttts_pc.yaml](https://github.com/DigitalPhonetics/VoicePAT/blob/vpc/configs/anon_ims_sttts_pc.yaml), default value is `exp/gan_anon`
```
python run_anonymization.py --config anon_ims_sttts_pc.yaml --gpu_ids 0  --force_compute True
```
The anonymized audios will be saved to `$results_dir`


### Evaluation
Evaluation metrics includes:
- Privacy: Equal error rate (EER) for Ignorant, lazy-informed, and semi-informed attackers
- Utility:
  - Word Error Rate (WER) by an pretrained ASR model trained on original 360h LibriSpeech dataset
  - Voice Distinctiveness ($G_{vd}$) by an pretrained ASV model trained on original 360h LibriSpeech dataset

The tookit supports the evaluation for any anonymized data:
1. prepare 7 anonymized folders each containing the anonymized wav files:
```
   libri_dev_enrolls/*wav
   libri_dev_trials_m/*wav
   libri_dev_trials_f/*wav

   libri_test_enrolls/*wav
   libri_test_trials_m/*wav
   libri_test_trials_f/*wav

   train-clean-360/*wav
```
2. modify entries in [configs/eval_pre_from_anon_datadir.yaml](https://github.com/DigitalPhonetics/VoicePAT/blob/vpc/configs/eval_pre_from_anon_datadir.yaml) and [configs/eval_post_scratch_from_anon_datadir.yaml](https://github.com/DigitalPhonetics/VoicePAT/blob/vpc/configs/eval_pre_from_anon_datadir.yaml) :
```
anon_data_dir: !PLACEHOLDER # TODO path to anonymized data (raw audios), e.g. <anon_data_dir>/libri_test_enrolls/*wav etc.
anon_data_suffix: !PLACEHOLDER  # suffix for dataset to signal that it is anonymized, e.g. b2, b1b, or gan
```
3. perform evaluations
  ```
  python run_evaluation.py --config eval_pre_from_anon_datadir.yaml
  python run_evaluation.py --config eval_post_scratch_from_anon_datadir.yaml
  ```







