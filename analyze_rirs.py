import argparse
import sys
import os
import csv
import librosa
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Config:
    stimuli_type: str = 'speech'
    speech_dir: str = "./stimuli/speech"
    music_dir: str = "./stimuli/music"
    clap_dir: str = "./stimuli/clap"
    clotho_eval_dir: str = "./clotho/evaluation"
    MIT_AUDIO: str = "./MIT_audio"
    openAIR: str = "./openAIR"

def create_openair_lookup(openair_lookup: dict[str, str], cfg: Config):
    # Make a lookup table of openair files where the file.wav is the key and the value is the path.
    if openair_lookup is None:
        openair_lookup = {}

    for subfolder in os.listdir(cfg.openAIR):
        if not os.path.isdir(os.path.join(cfg.openAIR, subfolder)): continue
        for subsubfolder in os.listdir(os.path.join(cfg.openAIR, subfolder)):
            if not os.path.isdir(os.path.join(cfg.openAIR, subfolder, subsubfolder)): continue
            for file in os.listdir(os.path.join(cfg.openAIR, subfolder, subsubfolder)):
                if not file.endswith(".wav"): continue
                openair_lookup[file] = os.path.join(cfg.openAIR, subfolder, subsubfolder, file)
    return openair_lookup

def analyze_rir(file_path):
    # https://python-acoustics.github.io/python-acoustics/room.html 
    try:
        # Read the audio file with librosa
        signal, fs = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

    # Normalize signal to avoid overflow issues and convert to float64
    signal = signal.astype(np.float64)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    # Find the arrival time of the direct sound (peak of the impulse)
    t0 = np.argmax(np.abs(signal))

    # Calculate squared energy of the impulse response
    h2 = signal ** 2

    # Calculate sample indices for time windows
    ms50 = int(0.050 * fs)
    ms80 = int(0.080 * fs)
    # A standard 2.5ms window is often used to isolate direct sound energy for DRR
    drr_window = int(0.0025 * fs)

    # Total energy from the arrival time onwards
    total_energy = np.sum(h2[t0:])

    # ---------------------------------------------------------
    # 1. Clarity (C50 and C80)
    # Ratio of early energy to late energy, expressed in dB
    # ---------------------------------------------------------
    early_50 = np.sum(h2[t0:t0 + ms50])
    late_50 = np.sum(h2[t0 + ms50:])
    c50 = 10 * np.log10(early_50 / late_50) if late_50 > 0 else float('inf')

    early_80 = np.sum(h2[t0:t0 + ms80])
    late_80 = np.sum(h2[t0 + ms80:])
    c80 = 10 * np.log10(early_80 / late_80) if late_80 > 0 else float('inf')

    # ---------------------------------------------------------
    # 2. Definition (D50 - Deutlichkeit)
    # Ratio of early 50ms energy to total energy, expressed as %
    # ---------------------------------------------------------
    d50 = (early_50 / total_energy) * 100 if total_energy > 0 else 0

    # ---------------------------------------------------------
    # 3. Direct-to-Reverberant Ratio (DRR)
    # Ratio of direct sound to reverberant tail
    # ---------------------------------------------------------
    direct_start = max(0, t0 - drr_window)
    direct_end = min(len(h2), t0 + drr_window)
    direct_energy = np.sum(h2[direct_start:direct_end])
    reverb_energy = np.sum(h2) - direct_energy
    drr = 10 * np.log10(direct_energy / reverb_energy) if reverb_energy > 0 else float('inf')

    # ---------------------------------------------------------
    # 4. Reverberation Time (RT60)
    # Estimated using the Schroeder Integration method (T20 extrapolation)
    # ---------------------------------------------------------
    # Backward integration of the squared impulse response
    schroeder = np.cumsum(h2[::-1])[::-1]

    # Convert to decibels (add tiny epsilon to prevent log10(0))
    schroeder_db = 10 * np.log10(schroeder / np.max(schroeder) + 1e-10)

    try:
        # Find time to drop from -5 dB to -25 dB (T20 parameter)
        idx_minus_5 = np.where(schroeder_db <= -5.0)[0][0]
        idx_minus_25 = np.where(schroeder_db <= -25.0)[0][0]

        # Calculate T20 in seconds, then multiply by 3 to extrapolate to RT60
        t20_samples = idx_minus_25 - idx_minus_5
        rt60 = (t20_samples / fs) * 3
    except IndexError:
        # Triggers if the audio clip is too short to drop by 25dB
        rt60 = float('nan')

        # ---------------------------------------------------------
    # Generate CLI Report
    # ---------------------------------------------------------


    return file_path, fs, rt60, drr, c50, c80, d50


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a Room Impulse Response (WAV) to extract acoustic parameters.")
    parser.add_argument("--rir_file", type=str, help="Path to the Room Impulse Response (.wav) file")
    cfg = Config()

    args = parser.parse_args()

    openair_lookup = create_openair_lookup(None, cfg)
    if args.rir_file is None:
        with open("audio_stimuli.csv", "r") as f:
            csv_reader = csv.reader(f, delimiter=",")
            first_row = next(csv_reader)

            for row in tqdm(csv_reader, desc="Analyzing RIRs", unit="RIRs"):
                if row[0] in openair_lookup:
                    rir_path = openair_lookup[row[0]]
                else:
                    rir_path = os.path.join(cfg.MIT_AUDIO, row[0])

                # Add Measurements and Sample rate to audio_stimul.csv
                with open("audio_stimuli_with_measurements.csv", "a") as nf:
                    # Check if the file is empty or not
                    if os.stat("audio_stimuli_with_measurements.csv").st_size == 0:
                        nf.write(f"{first_row[0]},{first_row[1]},{first_row[2]},{first_row[3]},{first_row[4]},{first_row[5]},{first_row[6]},{first_row[7]},{first_row[8]},{first_row[9]},sample_rate,rt60,drr,c50,c80,d50\n")
                    nf.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]},{row[9]},{fs},{rt60:.4f},{drr:.4f},{c50:.4f},{c80:.4f},{d50:.4f}\n")
                    nf.flush()
    else:
        file_path, fs, rt60, drr, c50, c80, d50 = analyze_rir(args.rir_file)
        print("\n" + "=" * 45)
        print("          ACOUSTIC PARAMETER REPORT")
        print("=" * 45)
        print(f"File             : {file_path}")
        print(f"Sample Rate      : {fs} Hz")
        print("-" * 45)

        if np.isnan(rt60):
            print("RT60             : [Error] RIR decay < 25dB")
        else:
            print(f"RT60             : {rt60:.2f} seconds")

        print(f"DRR              : {drr:.2f} dB")
        print(f"Clarity (C50)    : {c50:.2f} dB")
        print(f"Clarity (C80)    : {c80:.2f} dB")
        print(f"Definition (D50) : {d50:.2f} %")
        print("=" * 45 + "\n")
