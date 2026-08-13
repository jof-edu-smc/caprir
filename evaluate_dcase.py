import csv
import json
import os
import random
import argparse
import librosa
import torchaudio
import sys
from dataclasses import dataclass
import torch
import transformers
from finetune_whisper import WhisperCaptionDataset, collate_fn as whisper_collate_fn
from finetune_wstxxn import AudioCaptionDataset, collate_fn as wstxxn_collate_fn
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print(f"Transformers version: {transformers.__version__}")
from torch.utils.data import Dataset, DataLoader # type: ignore
from transformers import AutoModel, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download
import importlib.util
from tqdm import tqdm

@dataclass
class Config:
    stimuli_type: str = 'speech'
    speech_dir: str = "./stimuli/speech"
    music_dir: str = "./stimuli/music"
    clap_dir: str = "./stimuli/clap"
    clotho_eval_dir: str = "./clotho/evaluation"

def whisper(checkpoint: str = "MU-NLPC/whisper-small-audio-captioning"):
    path = hf_hub_download("MU-NLPC/whisper-small-audio-captioning", "model_class.py")
    spec = importlib.util.spec_from_file_location("model_class", path)
    mod = importlib.util.module_from_spec(spec);
    spec.loader.exec_module(mod)
    WhisperForAudioCaptioning = mod.WhisperForAudioCaptioning
    # checkpoint = "MU-NLPC/whisper-small-audio-captioning"
    print(f"Loading model from {checkpoint}")
    model = WhisperForAudioCaptioning.from_pretrained(
        checkpoint,
    )
    tokenizer = transformers.WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(checkpoint)
    return model, tokenizer, feature_extractor

def wstxxn(checkpoint: str = "wsntxxn/cnn14rnn-tempgru-audiocaps-captioning"):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model from {checkpoint}")
    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True
    ).to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "wsntxxn/audiocaps-simple-tokenizer"
    )
    return model, tokenizer


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config()
    
    if not args.whisper and not args.wstxxn:
        raise ValueError("Please specify a model to evaluate")
    with open(args.metadata, 'r') as f:
        metadata = f.readlines()
    # Setup Dataset & DataLoader
    

    print("\n--- Running caption evaluation ---")
    candidates: list[str] = []
    references: list[list[str]] = []

    if args.whisper:
        name_of_model = "whisper"
        # Load model
        model, tokenizer, feature_extractor = whisper(args.checkpoint)
        dataset = WhisperCaptionDataset(metadata, tokenizer, feature_extractor, source="caprir")
        print(f"Evaluating Whisper model on {len(dataset)} samples")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=whisper_collate_fn)
        # Prepare caption style
        style_prefix = "clotho > caption: "
        style_prefix_tokens = tokenizer("", text_target=style_prefix, return_tensors="pt",
                                        add_special_tokens=False).labels

        # Generate caption
        model.eval()
        
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating Whisper", unit="batch"):
            input_features = batch["input_features"].to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs=input_features.to(model.device),
                    forced_ac_decoder_ids=style_prefix_tokens,
                    max_length=100,
                )
            candidate = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace("clotho > caption: ", "")
            candidates.append(candidate)
            references.append([ref.replace("clotho > caption: ", "") for ref in batch["labels_text"]])
                
    if args.wstxxn:
        # Load model
        name_of_model = "wstxxn"
        model, tokenizer = wstxxn(args.checkpoint)
        dataset = AudioCaptionDataset(metadata, tokenizer, model.config.sample_rate, device=device, source="caprir")
        print(f"Evaluating wstxxn model on {len(dataset)} samples")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=wstxxn_collate_fn)
    

        model.eval()
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating wstxxn", unit="batch"):
            wav = batch['audio'].to(device)
        
            with torch.no_grad():
                word_idxs = model(
                    audio=wav,
                    audio_length=[wav.size(1)],
                )
            candidate = tokenizer.decode(word_idxs[0], skip_special_tokens=True)
            candidates.append(candidate) 
            references.append([ref.replace("clotho > caption: ", "") for ref in batch["labels_text"]])
    
    if len(candidates) != len(references):
        raise ValueError(f"Number of candidates ({len(candidates)}) does not match number of references ({len(references)})")

    # Combine them into a single dictionary
    evaluation_data = {
        "candidates": candidates,  # e.g., ["candidate 1", "candidate 2"]
        "references": references   # e.g., [["ref1a", "ref1b"], ["ref2a", "ref2b"]]
    }

    file_name = f"candidates_references_{name_of_model}_{args.metadata.replace('.csv', '')}.json"

    with open(file_name, "w") as f:
        json.dump(evaluation_data, f, indent=4)

    # for i, r in enumerate(references):
    #     # print the referenced audio file along with the reference caption and the output from the model.
    #     print("\n Reference audio: ", references_audio[i])
    #     print("   Reference caption:")
    #     print("     -->", r[0])
    #     print("   Output from model:")
    #     print("    -->", candidates[i])

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--metadata", type=str, default="metadata.csv", help="Path to the metadata CSV file for the dataset")
    args.add_argument("--wstxxn", action="store_true", help="Test wstxxn model")
    args.add_argument("--whisper", action="store_true", help="Test whisper model")
    args.add_argument("--checkpoint", type=str, help="directory for the fine-tuned model")
    args = args.parse_args()
    if not args.whisper and not args.wstxxn:
        raise ValueError("Please specify a model to evaluate")
    if args.whisper and args.wstxxn:
        raise ValueError("Please specify only one model to evaluate")
    if args.whisper:
        print("Evaluating Whisper model")
    if args.wstxxn:
        print("Evaluating WSTXXN model")
    run_evaluation(args)
    exit(0)