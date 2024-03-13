# Recipe for VoicePrivacy Challenge 2024

Please visit the [challenge website](https://www.voiceprivacychallenge.org/) for more information about the Challenge.

## Evaluation Plan

For more details about the baseline and data, please see [The VoicePrivacy 2024 Challenge Evaluation Plan](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2024_Eval_Plan_v1.0.pdf).

#### Registration
Participants are requested to register for the evaluation. Registration should be performed once only for each participating entity using the following form: **[Registration](https://forms.office.com/r/T2ZHD1p3UD)**.

---

## Install
```bash
git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024.git
cd Voice-Privacy-Challenge-2024
./00_install.sh
```

## Download data

`./01_download_data_model.sh` 
Password required, please register to get the password.  

You can modify the `librispeech_corpus` variable of `./01_download_data_model.sh` to avoid downloading LibriSpeech 360.  
You should modify the `iemocap_corpus` variable of `./01_download_data_model.sh` to where it is located on your server.  

> [!IMPORTANT]  
> The [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) corpus needs to be downloaded on your own by submitting a request at https://sail.usc.edu/iemocap/iemocap_release.htm.

---

## Anonymization and Evaluation
There are 2 options:
1.  (Recommended) Run anonymization and evaluation: `./02_run.sh configs/anon_mcadams.yaml`.  
    The available configs are:
    - [configs/anon_mcadams.yaml](configs/anon_mcadams.yaml) (the default) a fast CPU only signal processing based system. [paper](https://arxiv.org/abs/2011.01130)
    - [configs/anon_template.yaml](configs/anon_template.yaml) a template here to guide you through creating your own system.
    - [configs/anon_sttts.yaml](configs/anon_sttts.yaml) a system based on (unmodified) phone sequence, (modified) prosody, and (modified) speaker embedding representations + TTS. [paper1](https://www.isca-archive.org/interspeech_2022/meyer22b_interspeech.html), [paper2](https://ieeexplore.ieee.org/document/10022601), [paper3](https://ieeexplore.ieee.org/document/10096607)
    - [configs/anon_nac.yaml](configs/anon_nac.yaml) a system based on **n**eural **a**udio **c**odecs. [paper](https://arxiv.org/abs/2309.14129)
    - [configs/anon_asrbn.yaml](configs/anon_asrbn.yaml) a system based on vector quantized acoustic bottleneck, pitch, and one-hot speaker representations + HiFi-GAN. [paper](https://arxiv.org/abs/2308.04455)
2.  Run separately anonymization and evaluation in the two steps detailed here:

---------------------------------------------------------------------------

#### Step 1: Anonymization
```sh
python run_anonymization.py --config configs/anon_mcadams.yaml  # Compute time vary from 30min to 10h depending on the the number of cores.

# Or your own script..
```
The names of the 9 necessary anonymized datasets are the original names appended with a suffix corresponding to an anonymization pipeline, i.e. `$anon_data_suffix=_mcadams` (see configs/anon_mcadams.yaml).  

After anonymization, the directories/wavs produced should be the following:
```log
anon_data_suffix=_mcadams
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

> [!IMPORTANT]  
> When developing your own anonymization system, you can simply replicate this directory/wav
> structure.  
> The evaluation script will take care of everything else. (Automatic creation
> of: wav.scp/spk2gender/...)  
>
> Or for a more detailed example of an anonymization system, with original wav.scp
> loading and directory structure creation, please refer to:
> - [configs/anon_template.yaml](./configs/anon_template.yaml)
> - [anonymization/modules/template/anonymise_dir.py](./anonymization/modules/template/anonymise_dir.py)
> - [anonymization/pipelines/template/template_pipeline.py](./anonymization/pipelines/template/template_pipeline.py)

#### Step 2: Evaluation
Evaluation metrics includes:
- Privacy: Equal error rate (EER) for ignorant, lazy-informed, and semi-informed attackers (only results from the semi-informed attacker will be used in the challenge ranking) 
- Utility:
  - Word Error Rate (WER) by an automatic speech recognition (ASR) model (trained on LibriSpeech)
  - Unweighted Average Recall (UAR) by a speech emotion recognition (SER) model (trained on IEMOCAP).


To run evaluation separately for a pipeline you need to:
1. Have prepare the above anonymized data directories with the correct `$anon_data_suffix`.
2. perform evaluations
    ```sh
    python run_evaluation.py --config configs/eval_pre.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
    python run_evaluation.py --config configs/eval_pre.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
    ```

3. get the final relevant results for ranking
    ```sh
    results_summary_path_orig=exp/results_summary/eval_orig${anon_data_suffix}/results_orig.txt # the same value as $results_summary_path in configs/eval_pre.yaml
    results_summary_path_anon=exp/results_summary/eval_anon${anon_data_suffix}/results_anon.txt # the same value as $results_summary_path in configs/eval_post.yaml
    results_exp=exp/results_summary
    { cat "${results_summary_path_orig}"; echo; cat "${results_summary_path_anon}"; } > "${results_exp}/result_for_rank${anon_data_suffix}"
    ```

> (All of the above steps are automated in [02_run.sh](./02_run.sh)).

## Results

The result file with all the metrics and all datasets for submission will be generated in:
* Summary results: `./exp/results_summary/result_for_rank$anon_data_suffix`

Please see the [RESULTS folder](./results) for the provided anonymization pipelines.

### Some potential questions you may have and how to solve:
> 1. $ASV_{eval}^{anon}$ training is slow

Training of the $ASV_{eval}^{anon}$ model may vary from about 2 up to 10 hours depending on the available hardware.
If you have an SSD or a high-performance drive, $ASV_{eval}^{anon}$ takes ~2h, but if the drive is old and slow, in a worse case,  $ASV_{eval}^{anon}$ training takes ~10h. Increasing $num_workers in config/eval_post.yaml may help to speed up the processing.

> 2. OOM problem when decoding by $ASR_{eval}$

Reduce the $eval_bachsize in config/eval_pre.yaml

> 3. The $ASR_{eval}$ is a [pretrained wav2vec+ctc trained on LibriSpeech-960h](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech)

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

