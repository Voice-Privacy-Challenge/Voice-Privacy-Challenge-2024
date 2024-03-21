from pathlib import Path
from datetime import datetime
import time

from .. import Pipeline, get_anon_level_from_config
from utils import read_kaldi_format, copy_data_dir, save_kaldi_format, check_dependencies, setup_logger

from ...modules.sttts.tts import SpeechSynthesis
from ...modules.sttts.text import SpeechRecognition
from ...modules.sttts.prosody import ProsodyExtraction, ProsodyAnonymization
from ...modules.sttts.speaker_embeddings.speaker_extraction import SpeakerExtraction
from ...modules.sttts.speaker_embeddings.speaker_anonymization import SpeakerAnonymization
import typing

logger = setup_logger(__name__)

class STTTSPipeline(Pipeline):
    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        """
        Instantiates a STTTSPipeline with the complete feature extraction,
        modification and resynthesis.

        This pipeline consists of:
              - ASR -> phone sequence                    -
        input - (prosody extr. -> prosody anon.)         - TTS -> output
              - speaker embedding extr. -> speaker anon. -

        Args:
            config (dict): a configuration dictionary, e.g., see anon_ims_sttts_pc.yaml
            force_compute (bool): if True, forces re-computation of
                all steps. otherwise uses saved results.
            devices (list): a list of torch-interpretable devices
        """
        self.total_start_time = time.time()
        self.config = config
        model_dir = Path(config.get("models_dir", "models"))
        vectors_dir = Path(config.get("vectors_dir", "original_speaker_embeddings"))
        self.results_dir = Path(config.get("results_dir", "results"))
        self.intermediate_dir = Path(config.get("intermediate_dir", "exp/intermediate_dir_stts"))
        self.data_dir = Path(config["data_dir"]) if "data_dir" in config else None
        self.anon_suffix = config.get("anon_suffix", "ims_sttts")
        save_intermediate = config.get("save_intermediate", True)

        modules_config = config["modules"]
        self.modules_config = config["modules"]

        # ASR component
        self.speech_recognition = SpeechRecognition(
            devices=devices,
            save_intermediate=save_intermediate,
            settings=modules_config["asr"],
            #force_compute=force_compute,
        )

        # Speaker component
        self.speaker_extraction = {}
        self.speaker_anonymization = {}
        for level in ["utt", "spk"]:
            modules_config["speaker_embeddings"]['emb_level'] = level
            self.speaker_extraction[level] = SpeakerExtraction(
                devices=devices,
                save_intermediate=save_intermediate,
                settings=modules_config["speaker_embeddings"],
                #force_compute=force_compute,
            )
            self.speaker_anonymization[level] = SpeakerAnonymization(
                vectors_dir=vectors_dir,
                device=devices[0],
                save_intermediate=save_intermediate,
                settings=modules_config["speaker_embeddings"],
                force_compute=force_compute,
            )

        # Prosody component
        if "prosody" in modules_config:
            self.prosody_extraction = ProsodyExtraction(
                device=devices[0],
                save_intermediate=save_intermediate,
                settings=modules_config["prosody"],
                #force_compute=force_compute,
            )
            if "anonymizer_type" in modules_config["prosody"]:
                self.prosody_anonymization = ProsodyAnonymization(
                    save_intermediate=save_intermediate,
                    settings=modules_config["prosody"],
                    force_compute=force_compute,
                )
            else:
                self.prosody_anonymization = None
        else:
            self.prosody_extraction = None

        # TTS component
        self.speech_synthesis = SpeechSynthesis(
            devices=devices,
            settings=modules_config["tts"],
            model_dir=model_dir,
            save_output=config.get("save_output", True),
            force_compute=force_compute,
        )

    def run_anonymization_pipeline(
        self,
        datasets: typing.Dict[str, Path],
    ):
        """
            Runs the anonymization algorithm on the given datasets. Optionally
            prepares the results such that the evaluation pipeline
            can interpret them.

            Args:
                datasets (dict of str -> Path): The datasets on which the
                    anonymization pipeline should be runned on. These dataset
                    will be processed sequentially.
        """
        anon_wav_scps = {}

        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            logger.info(f"{i + 1}/{len(datasets)}: Processing {dataset_name}...")
            # Step 1: Recognize speech, extract speaker embeddings, extract prosody
            start_time = time.time()
            texts = self.speech_recognition.recognize_speech(dataset_path=dataset_path, dataset_name=dataset_name)
            logger.info("--- Speech recognition time: %f min ---" % (float(time.time() - start_time) / 60))

            start_time = time.time()
            anon_level = get_anon_level_from_config(self.modules_config["speaker_embeddings"], dataset_name)
            spk_embeddings = self.speaker_extraction[anon_level].extract_speakers(dataset_path=dataset_path,
                                                                      dataset_name=dataset_name)
            logger.info(f"--- Speaker extraction anon_level ({anon_level}) time: {(float(time.time() - start_time) / 60)} min ---")

            if self.prosody_extraction:
                start_time = time.time()
                prosody = self.prosody_extraction.extract_prosody(dataset_path=dataset_path, dataset_name=dataset_name,
                                                                  texts=texts)
                logger.info(f"--- Prosody extraction time: {(float(time.time() - start_time) / 60)} min ---")
            else:
                prosody = None

            # Step 2: Anonymize speaker, change prosody
            if self.speaker_anonymization and anon_level in self.speaker_anonymization:
                start_time = time.time()
                anon_embeddings = self.speaker_anonymization[anon_level].anonymize_embeddings(speaker_embeddings=spk_embeddings,
                                                                                  dataset_name=dataset_name)
                logger.info(f"--- Speaker anonymization at anon_level ({anon_level}) time: {(float(time.time() - start_time) / 60)} min ---")
            else:
                anon_embeddings = spk_embeddings

            if self.prosody_anonymization:
                start_time = time.time()
                anon_prosody = self.prosody_anonymization.anonymize_prosody(prosody=prosody, dataset_name=dataset_name)
                logger.info("--- Prosody anonymization time: %f min ---" % (float(time.time() - start_time) / 60))
            else:
                anon_prosody = prosody

            # Step 3: Synthesize
            start_time = time.time()
            wav_scp = self.speech_synthesis.synthesize_speech(dataset_name=dataset_name, texts=texts,
                                                              speaker_embeddings=anon_embeddings,
                                                              prosody=anon_prosody, emb_level=anon_embeddings.emb_level)
            logger.info("--- Synthesis time: %f min ---" % (float(time.time() - start_time) / 60))

            anon_wav_scps[dataset_name] = wav_scp
            logger.info("Anonymization pipeline completed.")

            if self.speaker_anonymization:
                anon_vectors_path = self.speaker_anonymization[anon_level].results_dir
            else:
                anon_vectors_path = self.speaker_extraction[anon_level].results_dir

            logger.info(f"{i + 1}/{len(datasets)}: Preparing kaldi {dataset_name}...")

            output_path = Path(str(dataset_path) + self.anon_suffix)
            copy_data_dir(dataset_path, output_path)
            # Overwrite spk2gender if it has been modify
            spk2gender_anon = read_kaldi_format(anon_vectors_path / dataset_name / 'spk2gender')
            save_kaldi_format(spk2gender_anon, output_path / 'spk2gender')
            # Overwrite wav.scp with the paths to the anonymized wavs
            save_kaldi_format(wav_scp, output_path / 'wav.scp')

        logger.info("--- Total computation time: %f min ---" % (float(time.time() - self.total_start_time) / 60))

        return anon_wav_scps
