# https://huggingface.co/wsntxxn/cnn14rnn-tempgru-audiocaps-captioning
import torch # type: ignore
from transformers import AutoModel, PreTrainedTokenizerFast # type: ignore
from dotenv import load_dotenv # type: ignore
import torchaudio # type: ignore
import argparse # type: ignore
import librosa # type: ignore

def run_whisper(audio_file, temporal_tag=0, checkpoint="wsntxxn/cnn14rnn-tempgru-audiocaps-captioning"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True
    ).to(device)

    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "wsntxxn/audiocaps-simple-tokenizer"
    )

    if audio_file.endswith(".flac"):
        wav, sr = librosa.load(audio_file, sr=16000)
        wav = torch.from_numpy(wav).unsqueeze(0)
    else:
        wav, sr = torchaudio.load(audio_file)

    if wav.size(0) > 1:
        wav = wav.mean(0).unsqueeze(0)

    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)

    wav = wav.to(device)

    with torch.no_grad():
        word_idxs = model(
            audio=wav,
            audio_length=[wav.size(1)],
            temporal_tag=[temporal_tag]
        )

    caption = tokenizer.decode(word_idxs[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", type=str, help="Path to a single audio file for inference", required=True)
    parser.add_argument("--temporal_tag", type=int, default=0, help="Temporal tag for the audio")
    parser.add_argument("--checkpoint", type=str, default="wsntxxn/cnn14rnn-tempgru-audiocaps-captioning", help="Path to the checkpoint")
    args = parser.parse_args()
    print(run_whisper(args.audio_file, args.temporal_tag, checkpoint=args.checkpoint))