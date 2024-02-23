# Recipe for VoicePrivacy Challenge 2024 
(Under development)

## Install

1. `git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024.git`
2. `./00_install.sh`
3. `source env.sh`

## Download data and pretrianed models

`./01_download_data_model.sh` Password required, please register to get password.  
You can modify the `librispeech_corpus` variable to avoid downloading LibriSpeech 360.  
You should modify the `iemocap_corpus` variable to where it is located on your server.

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

```
python run_anonymization_dsp.py --config anon_dsp.yaml
```
The anonymized audios will be saved to `$data_dir=data`, and the `$anon_data_suffix=_mcadams`, including 9 folders:

```
  data/libri_dev_enrolls_mcadams/anon_wav/*wav
  data/libri_dev_trials_m_mcadams/anon_wav/*wav
  data/libri_dev_trials_f_mcadams/anon_wav/*wav

  data/libri_test_enrolls_mcadams/anon_wav/*wav
  data/libri_test_trials_m_mcadams/anon_wav/*wav
  data/libri_test_trials_f_mcadams/anon_wav/*wav

  data/IEMOCAP_dev_mcadams/anon_wav/*wav
  data/IEMOCAP_test_mcadams/anon_wav/*wav

  data/train-clean-360_mcadams/anon_wav/*wav
```

#### GAN-based: Anonymization using Transformer-based ASR, FastSpeech2-based TTS and WGAN-based anonymizer.

```
bash anonymization/pipelines/sttts/sttts_install.sh
source env.sh
python run_anonymization.py --config anon_ims_sttts_pc.yaml --gpu_ids 0  --force_compute True
```
The anonymized audios will be saved to `$data_dir`


### Evaluation
Evaluation metrics includes:
- Privacy: Equal error rate (EER) for Ignorant, lazy-informed, and semi-informed attackers
- Utility:
  - Word Error Rate (WER) by an pretrained ASR model trained on original 360h LibriSpeech dataset

The tookit supports the evaluation for any anonymized data:
1. prepare 9 anonymized folders each containing the anonymized wav files:
```
  data/libri_dev_enrolls$anon_data_suffix/anon_wav/*wav
  data/libri_dev_trials_m$anon_data_suffix/anon_wav/*wav
  data/libri_dev_trials_f$anon_data_suffix/anon_wav/*wav

  data/libri_test_enrolls$anon_data_suffix/anon_wav/*wav
  data/libri_test_trials_m$anon_data_suffix/anon_wav/*wav
  data/libri_test_trials_f$anon_data_suffix/anon_wav/*wav

  data/IEMOCAP_dev$anon_data_suffix/anon_wav/*wav
  data/IEMOCAP_test$anon_data_suffix/anon_wav/*wav

  data/train-clean-360_mcadams/anon_wav/*wav
```
2. modify entry in [configs/eval_pre_from_anon_datadir.yaml](https://github.com/DigitalPhonetics/VoicePAT/blob/vpc/configs/eval_pre.yaml) and [configs/eval_post.yaml](https://github.com/DigitalPhonetics/VoicePAT/blob/vpc/configs/eval_post.yaml) :
```
anon_data_suffix: !PLACEHOLDER  # suffix for dataset to signal that it is anonymized, e.g. _mcadams, _b1b, or _gan
```
3. perform evaluations
  ```
  python run_evaluation.py --config eval_pre.yaml
  python run_evaluation.py --config eval_post.yaml
  ```

### Some potential questions you may have and how to solve:
> 1. $ASV_{eval}^{anon}$ training is too slow!!

If you have an SSD or a high-performance drive, $ASV_{eval}^{anon}$ takes <3h, but if the drive is old and slow, in a worse case,  $ASV_{eval}^{anon}$ takes ~10h Increase $num_workers in config/eval_post.yaml may help to speed up the processing.

> 2. OOM problem when decoding by $ASR_{eval}$

Reduce the $eval_bachsize in config/eval_pre.yaml

> 3. The default $ASR_{eval}$ is a [pretrained wav2vec+ctc trained on LibriSpeech-960h](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech) if you want to use a pretrained transformer+tramsformerLM trained on LibriSpeech-360h,

Modify entries in configs/eval_pre.yaml: 
```
asr:
  model_name: asr_pre_transformer_transformerlm
evaluation:
  model_type: EncoderDecoderASR
```

