# Recipe for VoicePrivacy Challenge 2024 

Please visit the [challenge website](https://www.voiceprivacychallenge.org/) for more information about the Challenge.

## Install

1. `git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024.git`
2. `./00_install.sh`
3. `source env.sh`

## Download data and pretrianed models

`./01_download_data_model.sh` 
Password required, please register to get password.  

You can modify the `librispeech_corpus` variable to avoid downloading LibriSpeech 360.  
[IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) corpus is required to download separetely by submitting request: https://sail.usc.edu/iemocap/iemocap_release.htm.

You should modify the `iemocap_corpus` variable to where it is located on your server.

## Anonymization and Evaluation
There are 2 options: 
1.  Run McAdams (B2) anonymization and evaluation: `./02_run.sh`

2.  Run separetely anonymization and evaluation in two steps (currently McAdams (B2) is supported):


#### Step 1: Anonymization
```
python run_anonymization.py --config anon_mcadams.yaml
```
The anonymized audios will be saved in `$data_dir=data` into 9 folders corresponding to datasets.  
The names of the created dataset folders for anonymized audio files are appended with the suffix, i.e. `$anon_data_suffix=_mcadams`. 
For McAdams, anonymization of all data may vary from 30 min up to 10 hours depending on the available hardware (in particular, the number of available CPU cores (https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/blob/8df345b196e0c81291160a31ff00f6b56f5231db/anonymization/modules/mcadams/anonymise_dir_mcadams_rand_seed.py#L82). 

```
  $data_dir/libri_dev_enrolls$anon_data_suffix/wav/*wav
  $data_dir/libri_dev_trials_m$anon_data_suffix/wav/*wav
  $data_dir/libri_dev_trials_f$anon_data_suffix/wav/*wav

  $data_dir/libri_test_enrolls$anon_data_suffix/wav/*wav
  $data_dir/libri_test_trials_m$anon_data_suffix/wav/*wav
  $data_dir/libri_test_trials_f$anon_data_suffix/wav/*wav

  $data_dir/IEMOCAP_dev$anon_data_suffix/wav/*wav
  $data_dir/IEMOCAP_test$anon_data_suffix/wav/*wav

  $data_dir/train-clean-360$anon_data_suffix/wav/*wav
```


#### Step 2: Evaluation
Evaluation metrics includes:
- Privacy: Equal error rate (EER) for ignorant, lazy-informed, and semi-informed attackers (only results from the semi-informed attacker will be used in the challenge ranking) 
- Utility:
  - Word Error Rate (WER) by an automatic speech recognition (ASR) model (trained on LibriSpeech)
  - Unweighted Average Recall (UAR) by a speech emotion recognition (SER) model (trained on IEMOCAP).


To run evaluation for arbitrary anonymized data:

1. prepare 9 anonymized folders each containing the anonymized wav files:
```
  data/libri_dev_enrolls$anon_data_suffix/wav/*wav
  data/libri_dev_trials_m$anon_data_suffix/wav/*wav
  data/libri_dev_trials_f$anon_data_suffix/wav/*wav

  data/libri_test_enrolls$anon_data_suffix/wav/*wav
  data/libri_test_trials_m$anon_data_suffix/wav/*wav
  data/libri_test_trials_f$anon_data_suffix/wav/*wav

  data/IEMOCAP_dev$anon_data_suffix/wav/*wav
  data/IEMOCAP_test$anon_data_suffix/wav/*wav

  data/train-clean-360_$anon_data_suffix/wav/*wav
```

2. modify entry in [configs/eval_pre.yaml](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/blob/main/configs/eval_pre.yaml)
   and [configs/eval_post.yaml](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/blob/main/configs/eval_post.yaml):
```
anon_data_suffix: !PLACEHOLDER  # suffix for dataset to signal that it is anonymized, e.g. _mcadams, _b1b, or _gan
```
3. perform evaluations
  ```
  python run_evaluation.py --config eval_pre.yaml
  python run_evaluation.py --config eval_post.yaml
  ```

4. get the final relevant results for ranking

```
anon_suffix=$anon_data_suffix  # TODO suffix for dataset to signal that it is anonymized, e.g. _mcadams, _b1b, or _gan
results_summary_path_orig=exp/results_summary/eval_orig${anon_suffix}/results_orig.txt # the same value as $results_summary_path in configs/eval_pre.yaml
results_summary_path_anon=exp/results_summary/eval_anon${anon_suffix}/results_anon.txt # the same value as $results_summary_path in configs/eval_post.yaml
results_exp=exp/results_summary
{ cat "${results_summary_path_orig}"; echo; cat "${results_summary_path_anon}"; } > "${results_exp}/result_for_rank${anon_suffix}"
```



## Results

The result file with all the metrics and all datasets for submission will be generated in:
* Summary results: `./exp/results_summary/result_for_rank$anon_data_suffix`

Please see 
* Summary [RESULTS B1](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/blob/main/results/result_for_rank_b1b)
* Summary [RESULTS B2](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/blob/main/results/result_for_rank_mcadams)

for the evalation and development data sets.



### Some potential questions you may have and how to solve:
> 1. $ASV_{eval}^{anon}$ training is slow

Training of the $ASV_{eval}^{anon}$ model may vary from about 2 up to 10 hours depending on the available hardware.
If you have an SSD or a high-performance drive, $ASV_{eval}^{anon}$ takes ~2h, but if the drive is old and slow, in a worse case,  $ASV_{eval}^{anon}$ training takes ~10h. Increasing $num_workers in config/eval_post.yaml may help to speed up the processing.

> 2. OOM problem when decoding by $ASR_{eval}$

Reduce the $eval_bachsize in config/eval_pre.yaml

> 3. The $ASR_{eval}$ is a [pretrained wav2vec+ctc trained on LibriSpeech-960h](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech)

## General information

For more details about the baseline and data, please see [The VoicePrivacy 2024 Challenge Evaluation Plan](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2024_Eval_Plan_v1.0.pdf)

#### Registration
Participants are requested to register for the evaluation. Registration should be performed once only for each participating entity using the following form: **[Registration](https://forms.office.com/r/T2ZHD1p3UD)**.

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

---------------------------------------------------------------------------

