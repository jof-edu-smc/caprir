import os
import librosa # type: ignore
import matplotlib.pyplot as plt #type: ignore
from dotenv import load_dotenv # type: ignore
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import soundfile as sf # type: ignore
import librosa # type: ignore
import scipy.signal # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt #type: ignore
import json 
import argparse
from tqdm import tqdm # type: ignore


def plot_waveform(y, sr: int = 44100, title: str = "Audio Waveform") -> None:
    """
    Plots the raw audio waveform and optionally overlays beat timestamps.
    """
    plt.figure(figsize=(14, 5))
    
    # Plot the raw waveform
    librosa.display.waveshow(y, sr=sr, alpha=0.6, color='blue')
        
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    
    
# openAIR Processing Block
def build_openair_rir_list(dataset_dir: str) -> None: 
    rir_list = {}
    spaces = sorted(os.listdir(dataset_dir))
    
    for space in spaces:
        path_to_space = os.path.join(dataset_dir, space)
        if ".DS_Store" in space:
            continue
        
        if os.path.isfile(path_to_space):
            print(f"File Skipped: {space}")
            continue 
        
        for subfolder in os.listdir(path_to_space):
            if "format" in subfolder or "stereo" in subfolder or "mono" in subfolder:
                path_to_subfolder = os.path.join(path_to_space, subfolder)
                
                if space not in rir_list:
                    rir_list[space] = []
            
                subfolder_files = [f for f in os.listdir(path_to_subfolder) if f.endswith(".wav")]
                
                if not subfolder_files:
                    print(f"No .wav files found in {path_to_subfolder}")
                    continue
                for f in subfolder_files:
                    if f.endswith(".wav"):
                        file_path = os.path.join(path_to_subfolder, f)
                        sr = librosa.get_samplerate(file_path)
                        rir_list[space].append((file_path, sr))
    # with open("openair_ir_list.json", "w") as f:
    #     json.dump(rir_list, f, indent=4)
    print(f"Total RIRs Processed in {dataset_dir}: {sum(len(v) for v in rir_list.values())}")
    return rir_list


def build_mit_ir_list(dataset_dir: str) -> None: 
    rir_list = {}
    files = sorted(os.listdir(dataset_dir))
    for file in files:
        key = file.split("_")[1].lower()
        if not file.endswith(".wav"):
            continue
        
        if key == "outside":
            key = file.split("_")[2].lower()
            
        if key not in rir_list:
            rir_list[key] = [] 
        file_path = os.path.join(dataset_dir, file)
        
        sr = librosa.get_samplerate(file_path)
        rir_list[key].append((file_path, sr))
    
    # with open("mit_ir_list.json", "w") as f:
    #     json.dump(rir_list, f, indent=4)
    print(f"Total RIRs Processed in {dataset_dir}: {sum(len(v) for v in rir_list.values())}")
    return rir_list

# Convolution Block 
def convolve_audio(rir_path: str, dry_audio_path: str, sample_rate: int, wet_mix: float = 1.0) -> tuple[np.ndarray, int]:
    """Convolve dry audio with a room impulse response, auto-align, and adjust mix.
    
    Args:
        rir_path: Path to the RIR file (.wav)
        dry_audio_path: Path to the dry audio file (.wav)
        sample_rate: Target sample rate for the convolution
        wet_mix: Float between 0.0 (100% dry) and 1.0 (100% wet). Default is 0.5.
        
    Returns:
        Tuple of (mixed_audio, sample_rate)
    """
    wet_mix = max(0.0, min(1.0, wet_mix))
    dry_mix = 1.0 - wet_mix

    # Load RIR and dry audio
    rir, _ = librosa.load(rir_path, sr=sample_rate, mono=False)
    dry_audio, dry_sr = librosa.load(dry_audio_path, sr=sample_rate, mono=False)
    
    # Ensure both are mono
    if rir.ndim > 1:
        rir = rir[0,:]
    if dry_audio.ndim > 1:
        dry_audio = dry_audio[0,:]

    if dry_sr != sample_rate:
        dry_audio = scipy.signal.resample(dry_audio, int(len(dry_audio) * sample_rate / dry_sr)) 
    
    # 100% Dry Early Exit
    if wet_mix == 0.0:
        # print(f"Returning 100% Dry Audio for {rir_path} and {dry_audio_path}")
        return dry_audio.astype(np.float32), sample_rate
    
    # # Trim the RIR if the energy drops below a threshold in decibels to avoid long tails in the convolved audio
    # threshold_db = -60  # dB
    # threshold = 10 ** (threshold_db / 20)
    # energy = np.abs(rir)
    # above_threshold = np.where(energy >= threshold)[0]
    # if above_threshold.size > 0:
    #     last_idx = above_threshold[-1]
    #     rir = rir[:last_idx + 1]
        
    # Convolve 
    convolved = scipy.signal.fftconvolve(dry_audio, rir, mode='full')
    
    if np.max(np.abs(convolved)) > 0:
        convolved /= np.max(np.abs(convolved)) 
    
    # 100% Wet Early Exit
    if wet_mix == 1.0:
        # print(f"Returning 100% Wet Convolved Audio for {rir_path} and {dry_audio_path}")
        return convolved.astype(np.float32), sample_rate

    peak_idx = np.argmax(np.abs(rir))
    rir = rir[peak_idx:]
    
    # Dry/Wet Mixing
    padding_length = len(convolved) - len(dry_audio)
    
    # Guard against negative padding if the trimmed RIR made the convolved file shorter than the dry file
    if padding_length > 0:
        dry_padded = np.pad(dry_audio, (0, padding_length), mode='constant')
    else:
        # If dry is longer, pad the convolved audio instead to match lengths safely
        dry_padded = dry_audio
        convolved = np.pad(convolved, (0, abs(padding_length)), mode='constant')

    # Blend the signals
    mixed_audio = (dry_padded * dry_mix) + (convolved * wet_mix)
    
    if np.max(np.abs(mixed_audio)) > 0:
        mixed_audio /= np.max(np.abs(mixed_audio))
    
    return mixed_audio.astype(np.float32), sample_rate

def write_audio_file(audio_data: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Write the convolved audio data to a .wav file.
    
    Args:
        audio_data: The convolved audio data as a numpy array.
        sample_rate: The sample rate of the audio data.
        output_path: The path where the output .wav file will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith(".wav"):
        sf.write(output_path, audio_data, sample_rate)
    
    if output_path.endswith(".mp3"):
        sf.write(output_path, audio_data, sample_rate, bitrate_mode='CONSTANT', compression_level=0.5)
    
    if output_path.endswith(".flac"):
        sf.write(output_path, audio_data, sample_rate, format='FLAC')
        
    elif ".wav" not in output_path or ".mp3" not in output_path:
        raise ValueError("Output path must end with .wav or .mp3")

def convolve_audio_pipeline(
    collection_name: str,
    file_num: int,
    ir_list: dict[str, list[tuple[str, int]]], 
    stimuli_dirs: list[str], 
    wet_mix: float, 
    df: pd.DataFrame
) -> pd.DataFrame:
    i = file_num
    warning_lengths = []
    for space, rir_files in tqdm(ir_list.items(), desc=f"Processing {collection_name} RIRs", unit="RIR"):
        
        for rir_path, sr in rir_files:
            # print(f"Processing RIR: {rir_path} with Sample Rate: {sr}")
            if sr > 48000:
                sr = 48000
            if sr < 32000:
                raise ValueError(f"Sample Rate {sr} is too low for {rir_path}. Must be at least 32kHz.") 
            filename = os.path.basename(rir_path)
            convolved_paths = {}

            for stimuli_dir in stimuli_dirs:
                if not os.path.exists(stimuli_dir):
                    raise FileNotFoundError(f"Stimuli directory {stimuli_dir} does not exist.")
                dry_audio_files = [f for f in os.listdir(stimuli_dir) if f.endswith(".wav")]
                if not dry_audio_files or len(dry_audio_files) != 1:
                    raise FileNotFoundError(f"No or too many .wav files found in stimuli directory {stimuli_dir}.")
                dry_audio_path = os.path.join(stimuli_dir, dry_audio_files[0])
                
                # print(f"Convolving {dry_audio_path} with {rir_path} at a Sample Rate of {sr} in space {space}, wet_mix={wet_mix}")
                convolved_audio, sample_rate = convolve_audio(
                    rir_path=rir_path, 
                    dry_audio_path=dry_audio_path, 
                    sample_rate=sr, 
                    wet_mix=wet_mix,
                )
                
                convolved_audio_path = os.path.join(stimuli_dir, f"C{i}_{filename.replace(".wav","")}_{dry_audio_files[0]}".replace(".wav", ".flac"))
                if len(convolved_audio) > sample_rate * 15:
                    warning_lengths.append(f"Warning: {convolved_audio_path} exceeds 15 seconds. The length is {len(convolved_audio)/sample_rate:.2f} seconds. It has been trimmed.")
                    convolved_audio = convolved_audio[:int(sample_rate * 15)]  
                
                # duration = 10
                write_audio_file(convolved_audio, sample_rate, convolved_audio_path)
                duration = librosa.get_duration(path=convolved_audio_path)
                convolved_paths[stimuli_dir] = (convolved_audio_path.split("/")[-1], "{:.2f}".format(duration))
    
            
            new_row = pd.DataFrame({
                "rir_id": [rir_path.split("/")[-1]],
                "audio_speech_id": [convolved_paths.get("stimuli/speech", None)[0]],
                "audio_music_id": [convolved_paths.get("stimuli/music", None)[0]],
                "audio_clap_id": [convolved_paths.get("stimuli/clap", None)[0]],
                "source_dataset": [f"{collection_name.replace(".","").replace("/","")}-{space}"],
                "duration_seconds": [convolved_paths.get("stimuli/speech", None)[1]],  
                "processing_step": ["Gathering"],
                "num_of_captions": [0],
                "num_of_scored_captions": [0],
                "max_captions": [5],
                "location_tags": "",
            })
            i += 1
            df = pd.concat([df, new_row], ignore_index=True)
    for warning in warning_lengths:
        print(warning)
    return df, i

def _build_parser() -> argparse.ArgumentParser:
    print(f"Loading Arguments")
    parser = argparse.ArgumentParser(description="Caption examples template")
    parser.add_argument("--wet_mix", type=int, default=1.0, help="Wet mix ratio for the convolved audio (0.0 = 100 percent dry, 1.0 = 100 percent wet, default: 1.0)", required=False)
    parser.add_argument("--openair_dir", help="Location of the Open AIR Survey Directory", required=False)
    parser.add_argument("--mit_dir", help="Location of the MIT IR survey directory", required=False)
    
    return parser

def main():
    df = pd.DataFrame(columns=[
        "rir_id",
        "audio_speech_id",
        "audio_music_id",
        "audio_clap_id",
        "source_dataset",
        "duration_seconds",
        "processing_step",
        "num_of_captions",
        "num_of_scored_captions",
        "max_captions",
        "location_tags",
        ]
    )
    
    args = _build_parser().parse_args()
    
    wet_mix = args.wet_mix
    stimuli_dirs = ["stimuli/speech", "stimuli/music", "stimuli/clap"]
    i = 1
    if args.openair_dir:
        openair_ir_list = build_openair_rir_list(args.openair_dir)
        df, i = convolve_audio_pipeline(args.openair_dir, i, openair_ir_list, stimuli_dirs, wet_mix, df)
        
    if args.mit_dir:
        mit_ir_list = build_mit_ir_list(args.mit_dir)
        df, _ = convolve_audio_pipeline(args.mit_dir, i, mit_ir_list, stimuli_dirs, wet_mix, df)
        
    if not df.empty:
        df.to_csv("rir_dataset.csv", index=False)   
        exit(0)
    else:
        print("No data to save. Exiting without creating CSV.")
        exit(1)
    
if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    main()