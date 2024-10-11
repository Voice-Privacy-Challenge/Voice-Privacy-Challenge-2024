# Recipe for VoicePrivacy Challenge 2024

Please visit the [challenge website](https://www.voiceprivacychallenge.org/) for more information about the Challenge.

## Data submission
The anonymization and evaluation scripts should have generated the files and the directories with the explained format of `$anon_data_suffix` suffix.  
For data submission, the following command submit everything given a `$anon_data_suffix` argument:
```
VPC_DROPBOX_KEY=XXX VPC_DROPBOX_SECRET=YYY VPC_DROPBOX_REFRESHTOKEN=ZZZ VPC_TEAM=TEAM_NAME ./03_upload_submission.sh $anon_data_suffix
```
`VPC_DROPBOX_KEY`, `VPC_DROPBOX_SECRET`, `VPC_DROPBOX_REFRESHTOKEN`, and `VPC_TEAM=TEAM_NAME` are sent individually to each team upon receiving their system description.  

## Install

1. `git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024.git`
2. `./00_install.sh`
3. `source env.sh`

## Download data

`./01_download_data_model.sh` 
A password is required; please register to get the password.  

You can modify the `librispeech_corpus` variable of `./01_download_data_model.sh` to avoid downloading LibriSpeech 360.  
You have to modify the `iemocap_corpus` variable of `./01_download_data_model.sh` to where it is located on your server.  

> [!IMPORTANT]  
> The [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) corpus must be downloaded on your own by submitting a request at https://sail.usc.edu/iemocap/iemocap_release.htm. The waiting time may take up to 7-9 days.


## Anonymization and Evaluation
There are two options:
1. Run anonymization and evaluation: `./02_run.sh configs/anon_mcadams.yaml`.  
    For each anonymization baseline, there is a corresponding config file:
    -  #### [Anonymization using the McAdams coefficient](https://arxiv.org/abs/2011.01130): **B2**
         [`configs/anon_mcadams.yaml`](configs/anon_mcadams.yaml)  A fast CPU-only signal processing-based system  (default).

    -  #### [Anonymization using phonetic transcriptions and GAN (STTTS)](https://ieeexplore.ieee.org/document/10096607): **B3**
         [`configs/anon_sttts.yaml`](configs/anon_sttts.yaml)  A system based on unmodified phone sequence, modified prosody, modified speaker embedding representations and speech synthesis.

    -  #### [Anonymization using **n**eural audio codec (NAC) language modeling](https://arxiv.org/abs/2309.14129): **B4**

        [`configs/anon_nac.yaml`](configs/anon_nac.yaml) 

    -  #### [Anonymization using ASR-BN with vector quantization (VQ)](https://arxiv.org/abs/2308.04455): **B5** and **B6** 

        [`configs/anon_asrbn.yaml`](configs/anon_asrbn.yaml) A fast system based on vector quantized acoustic bottleneck, pitch, and one-hot speaker representations and  a HiFi-GAN speech synthesis model.

      - #### ⚠ [Anonymization using x-vectors and a neural source-filter model](https://www.isca-archive.org/interspeech_2020/srivastava20_interspeech.pdf): **B1**
        anonymization scripts from https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022 can be used to obtain anonymized data for B1. To perform utterance-level  (in contrast to speaker-level) anonymization of the enrollment and trial data for B1, the corresponding parameters should be setup in  [https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022/blob/master/baseline/config.sh](config.sh): `anon_level_trials=utt` and `anon_level_enroll=utt` (https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022/blob/d72b50c44677aa9a1ba37b7f0c383c4fde13e05f/baseline/config.sh#L59-L60)
      
      
2. Run anonymization and evaluation separately in two steps:



#### Step 1: Anonymization
```sh
python run_anonymization.py --config configs/anon_mcadams.yaml  #Computational time varies from 30 minutes to 10 hours, depending on the number of cores, for other methods it may be longer and depending on the available hardware. 

```
The anonymized audios will be saved in `$data_dir=data` into 9 folders corresponding to datasets.
The names of the created dataset folders for anonymized audio files are appended with the suffix, i.e. `$anon_data_suffix=_mcadams`

```log
data/libri_dev_enrolls${anon_data_suffix}/wav/*wav
data/libri_dev_trials_m${anon_data_suffix}/wav/*wav
data/libri_dev_trials_f${anon_data_suffix}/wav/*wav

data/libri_test_enrolls${anon_data_suffix}/wav/*wav
data/libri_test_trials_m${anon_data_suffix}/wav/*wav
data/libri_test_trials_f${anon_data_suffix}/wav/*wav

data/IEMOCAP_dev${anon_data_suffix}/wav/*wav
data/IEMOCAP_test${anon_data_suffix}/wav/*wav

data/train-clean-360${anon_data_suffix}/wav/*wav
```
For the next evaluation step, you should replicate the corresponding directory structure when developing your anonymization system.  

#### Step 2: Evaluation
Evaluation metrics include:
- Privacy: Equal error rate (EER) for ignorant, lazy-informed, and semi-informed attackers (only results from the semi-informed attacker will be used in the challenge ranking) 
- Utility:
  - Word Error Rate (WER) by an automatic speech recognition (ASR) model (trained on LibriSpeech)
  - Unweighted Average Recall (UAR) by a speech emotion recognition (SER) model (trained on IEMOCAP).


To run evaluation for arbitrary anonymized data:
1. prepare 9 anonymized folders each containing the anonymized wav files:
```log
data/libri_dev_enrolls${anon_data_suffix}/wav/*wav
data/libri_dev_trials_m${anon_data_suffix}/wav/*wav
data/libri_dev_trials_f${anon_data_suffix}/wav/*wav

data/libri_test_enrolls${anon_data_suffix}/wav/*wav
data/libri_test_trials_m${anon_data_suffix}/wav/*wav
data/libri_test_trials_f${anon_data_suffix}/wav/*wav

data/IEMOCAP_dev${anon_data_suffix}/wav/*wav
data/IEMOCAP_test${anon_data_suffix}/wav/*wav

data/train-clean-360${anon_data_suffix}/wav/*wav
```
2. perform evaluations
   
```sh
python run_evaluation.py --config configs/eval_pre.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
python run_evaluation.py --config configs/eval_post.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
```

3. get the final results for ranking
```sh
results_summary_path_orig=exp/results_summary/eval_orig${anon_data_suffix}/results_orig.txt # the same value as $results_summary_path in configs/eval_pre.yaml
results_summary_path_anon=exp/results_summary/eval_anon${anon_data_suffix}/results_anon.txt # the same value as $results_summary_path in configs/eval_post.yaml
results_exp=exp/results_summary
{ cat "${results_summary_path_orig}"; echo; cat "${results_summary_path_anon}"; } > "${results_exp}/result_for_rank${anon_data_suffix}"
zip ${results_exp}/result_for_submission${anon_data_suffix}.zip -r exp/asr/*${anon_data_suffix} exp/asr/*${anon_data_suffix}.csv exp/ser/*${anon_data_suffix}.csv exp/results_summary/*${anon_data_suffix}* exp/asv_orig/*${anon_data_suffix} exp/asv_orig/*${anon_data_suffix}.csv exp/asv_anon${anon_data_suffix}
```

> All of the above steps are automated in [02_run.sh](./02_run.sh).

## Results
#### Note, that WER results are computed on the trials part
The result file with all the metrics and all datasets for submission will be generated in:
* Summary results: `./exp/results_summary/result_for_rank$anon_data_suffix`
* Additional information for submission: `./exp/results_summary/result_for_submission${anon_data_suffix}.zip`

Please see the [RESULTS folder](./results) for the provided anonymization baselines:

* [Results B1](./results/result_for_rank_b1b)
* [Results B2](./results/result_for_rank_mcadams)
* [Results B3](./results/result_for_rank_sttts)
* [Results B4](./results/result_for_rank_nac)
* [Results B5](./results/result_for_rank_asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1)
* [Results B6](./results/result_for_rank_asrbn_hifigan_bn_tdnnf_600h_vq_48_v1)

## General information

#### Evaluation plan
For more details about the baseline and data, please see [The VoicePrivacy 2024 Challenge Evaluation Plan](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2024_Eval_Plan_v2.0.pdf) - Updated on 1st April 2024

#### Training data
[Final list of models and data](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2024_Challenge_Final_list_of_models_and_data_for_training_anonymization_systems_-_26.03.2024.pdf) for training anonymization systems.

#### Registration
Participants are requested to register for the evaluation. Registration should be performed once only for each participating entity using the following form: **[Registration](https://forms.office.com/r/T2ZHD1p3UD)**.

## Some potential questions you may have and how to solve them:
> 1. $ASV_{eval}^{anon}$ training is slow

Training of the $ASV_{eval}^{anon}$ model may vary from about 2 up to 10 hours depending on the available hardware.
If you have an SSD or a high-performance drive, $ASV_{eval}^{anon}$ takes ~2h, but if the drive is old and slow, in a worse case,  $ASV_{eval}^{anon}$ training takes ~10h. Increasing $num_workers in config/eval_post.yaml may help to speed up the processing.

> 2. OOM problem when decoding by $ASR_{eval}$

Reduce the $eval_bachsize in config/eval_pre.yaml

> 3. The $ASR_{eval}$ is a [pretrained wav2vec+ctc trained on LibriSpeech-960h](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech)

> 4. Error on `utils.prepare_results_in_kaldi_format`

means something bad happened when running the anonymization pipeline.  
Remove all `data/*$anon_data_suffix` directories and re-run anonymization and evaluation steps (if `$anon_data_suffix=suff`, also remove the directories that share a matching suffix: `$anon_data_suffix=something_suff`). Check that your anonymization pipeline produces a wav file for each dataset entry, every original wav should have its anonymized counterpart.  



## Organizers (in alphabetical order)

- Pierre Champion - Inria, France
- Nicholas Evans - EURECOM, France
- Sarina Meyer - University of Stuttgart, Germany
- Xiaoxiao Miao - Singapore Institute of Technology, Singapore
- Michele Panariello - EURECOM, France
- Massimiliano Todisco - EURECOM, France
- Natalia Tomashenko - Inria, France
- Emmanuel Vincent - Inria, France
- Xin Wang - NII, Japan
- Junichi Yamagishi - NII, Japan

Contact: organisers@lists.voiceprivacychallenge.org

## License

Copyright (C) 2024

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

## Reference

```
@article{tomashenko2024voiceprivacy,
      title={The {VoicePrivacy} 2024 Challenge Evaluation Plan}, 
      author={Natalia Tomashenko and Xiaoxiao Miao and Pierre Champion and Sarina Meyer and Xin Wang and Emmanuel Vincent and Michele Panariello and Nicholas Evans and Junichi Yamagishi and Massimiliano Todisco},
      year={2024},
      eprint={2404.02677},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## Acknowledgments

Some parts of the code and structure are based on [VoicePAT](https://github.com/DigitalPhonetics/VoicePAT) (Paper: https://ieeexplore.ieee.org/document/10365329)
