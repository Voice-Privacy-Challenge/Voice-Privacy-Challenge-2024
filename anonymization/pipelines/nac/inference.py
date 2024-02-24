import torch
from torch.utils.data import DataLoader
from accelerate import PartialState

import soundfile as sf
import librosa
import os
import sys
# relative paths don't seem to work in Accelerate and i have no idea how else to solve this, lord forgive me
sys.path.append(os.path.join(os.getcwd(), 'anonymization/modules/nac'))
from argparse import ArgumentParser

from anonymizer import Anonymizer
from data import SCPPathDataset

parser = ArgumentParser(description="Perform inference with the anonymizer.")
parser.add_argument('scp_path', help='A Kaldi-like .scp file that contains the utterances to anonymize.')
parser.add_argument('output_folder', help='Where to save anonymized files.')
parser.add_argument('checkpoint_dir', help='Where the model checkpoints are saved.')
parser.add_argument('--data_root', help='Additional root where the wave files are placed. Will be concatenated to the '
                                        'paths found in the scp. If None, nothing will be concatenated.', default=None)
parser.add_argument('--voice_dir', help='Where voice prompts are stored, in Bark-compliant format.', default=None)
parser.add_argument('--mapping_file', help="Path to a JSON mapping file. If given, the 'speaker_level' parameter will "
                                           "be ignored.", default=None)
parser.add_argument('--speaker_level', default=False, action='store_true',
                    help='Whether to use speaker-level anon or utterance-level anon. Ignored if mapping_file is given.')
parser.add_argument('--ds_type', help="Either 'libri' or 'vctk'. Defaults to the former.", default='libri')
parser.add_argument('--target_rate', type=int, help="Resample output audio to this sampling rate. If not given, does not resample", default=None)
args = parser.parse_args()


def collate_for_distributed_inference(batch):
    """
    For the purposes of distributed inference, it's more convenient to batch like
    [(utt_id1, path1, basename1)
    (utt_id1, path2, basename2)...]
    This is the quickest hack I can think of to do that, to be honest.
    """
    return batch


distributed_state = PartialState()
device = distributed_state.device

print(f'Num processes: {distributed_state.num_processes}')

# create output folder if needed
if not os.path.isdir(args.output_folder):
    print(f'Output folder {args.output_folder} does not exist, creating it')
    os.makedirs(args.output_folder, exist_ok=True)  # yes i know i'm already checking, but exist_ok=True is needed
    # because if you are using multi-GPU you can end up in a race condition

print(f'(device {device}) Speaker level: {args.speaker_level}')
if args.target_rate is not None:
    print(f'(device {device}) Set to resample at {args.target_rate}')
print(f'(device {device}) Instantiating model')
anonymizer = Anonymizer(args.checkpoint_dir, voice_dirs=[args.voice_dir]).to(device)

print(f'(device {device}) Creating dataloader (speaker mapping: {args.mapping_file})')
dataset = SCPPathDataset(
    args.scp_path,
    root=args.data_root,
    voice_folder=args.voice_dir,
    speaker_level=args.speaker_level,
    ds_type=args.ds_type,
    mapping_file=args.mapping_file
)

# --- this was to generate speaker mappings beforehand in multi-gpu settings
# speaker_mapping_path = os.path.join(args.output_folder, f'speaker_mapping_cuda_{device}.json')
# print(f"(device {device}) Saving speaker mapping to {speaker_mapping_path}")
# dataset.save_speaker_mapping(speaker_mapping_path)
# print(f'(device {device}) Quitting.')
# quit()

dl = DataLoader(
    dataset,
    batch_size=distributed_state.num_processes,
    collate_fn=collate_for_distributed_inference
)
len_dl = len(dl)

print(f'({device}) Starting inference')
for i, batch in enumerate(dl, 1):
    with distributed_state.split_between_processes(batch) as single_item:
        # in the last batch, some GPU will not receive any data
        # so we need to check if the batch actually contains something,
        # otherwise your GPU will explode and you will eternally burn in the fires of hell (ok maybe not)
        if single_item:
            utt_id, path, basename, proxy_speaker = single_item[0]
            print(f'[{i}/{len_dl}] {device}\t| {utt_id}\t| {proxy_speaker}')

            with torch.no_grad():
                anon_wav = anonymizer(path, target_voice_id=proxy_speaker)

            # we save the output in wav regardless of what the input format was
            basename_file, _ = os.path.splitext(basename)
            new_basename = f'{basename_file}.wav'
            out_path = os.path.join(args.output_folder, new_basename)
            if args.target_rate is None:
                sf.write(out_path, anon_wav, samplerate=anonymizer.sample_rate)
            else:
                anon_wav = librosa.resample(anon_wav, orig_sr=anonymizer.sample_rate, target_sr=args.target_rate)
                sf.write(out_path, anon_wav, samplerate=args.target_rate)
        else:
            print(f'[{i}/{len_dl}] {device}\t| Received item {single_item}, this is probably the last batch. Skipping.')
