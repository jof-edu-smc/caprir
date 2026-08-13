# https://huggingface.co/MU-NLPC
import torch # type: ignore
import torchaudio # type: ignore
import argparse
import os
import transformers # type: ignore
import librosa # type: ignore
from dotenv import load_dotenv # type: ignore
from huggingface_hub import hf_hub_download # type: ignore
import importlib.util, sys
path = hf_hub_download("MU-NLPC/whisper-small-audio-captioning", "model_class.py")
spec = importlib.util.spec_from_file_location("model_class", path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
WhisperForAudioCaptioning = mod.WhisperForAudioCaptioning

def run_whisper(audio_file, checkpoint="MU-NLPC/whisper-small-audio-captioning"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # OpenAI Whisper Original
    # checkpoint = "openai/whisper-tiny"
    if "openai/whisper" in checkpoint:
        model = transformers.WhisperForConditionalGeneration.from_pretrained(checkpoint).to(device)
    else:
        model = WhisperForAudioCaptioning.from_pretrained(
            checkpoint,
        )

    # MU-NLPC Whisper Fine-tuned
    # checkpoint = "MU-NLPC/whisper-tiny-audio-captioning"
    # checkpoint = "MU-NLPC/whisper-small-audio-captioning"
    tokenizer = transformers.WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(checkpoint)

    # Load your audio (16kHz mono)
    # audio, sr = torchaudio.load(audio_file)
    audio, sr = librosa.load(audio_file, sr=16000)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    audio = audio.astype("float32")
    # audio = audio.mean(dim=0).squeeze().to(torch.float32).numpy()
    features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features

    # Prepare caption style
    style_prefix = "clotho > caption: "
    style_prefix_tokens = tokenizer("", text_target=style_prefix, return_tensors="pt", add_special_tokens=False).labels

    # Generate caption
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs=features.to(model.device),
            forced_ac_decoder_ids=style_prefix_tokens,
            max_length=100,
            task="transcribe",
            language="english",
        )

    # with torch.no_grad():
    #     outputs = model.generate(
    #         inputs=features.to(model.device),
    #         max_length=100,
    #     )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", type=str, help="Path to a single audio file for inference")
    parser.add_argument("--checkpoint", type=str, default="MU-NLPC/whisper-tiny-audio-captioning", help="Path to the checkpoint")
    args = parser.parse_args()
    print(run_whisper(args.audio_file, args.checkpoint))