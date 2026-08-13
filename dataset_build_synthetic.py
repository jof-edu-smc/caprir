import json
import os
import csv
import time
from google import genai # type: ignore
from google.genai import types # type: ignore
from pydantic import BaseModel, Field
from dataclasses import dataclass
from dotenv import load_dotenv # type: ignore
from tqdm import tqdm

load_dotenv()

@dataclass
class Config:
    speech_dir: str = "./stimuli/speech"
    music_dir: str = "./stimuli/music"
    pop_dir: str = "./stimuli/clap"
    MIT_AUDIO: str = "./MIT_audio"
    openAIR: str = "./openAIR"
    run_openai: bool = True

    annotators_sheet = "annotators_table"
    captions_sheet = "captions_table"
    stimuli_sheet = "audio_stimuli"
    evaluations_sheet = "evaluations_table"
    metadata_file = "audio_stimuli.csv"
    spreadsheet_id = os.getenv("GOOGLE_SHEETS_ID")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

cfg = Config()

# 1. Define the output structure (forces the model to return exactly 5 captions)
class RIRCaptions(BaseModel):
    file_name: str
    captions: list[str] = Field(
        min_length=5,
        max_length=5,
        description="Five distinct acoustic captions describing the room reverberation, warmth, early reflections, brightness, and spaciousness."
    )

# 2. Initialize Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

OUTPUT_FILE = "gemini_rir_captions.json"
results = []

# System instruction to guide the acoustic evaluation
system_instruction = """
You are an expert audio engineer and acoustic analyst. 
Analyze the provided Room Impulse Response (RIR) audio file and generate 5 detailed, natural language captions no longer than 250 characters. 
Describe acoustic qualities such as reverberation time (RT60 feel), room size/volume, wall materials, early reflections, decay curve, spectral tilt (bright/dark), and spatial intimacy.
"""

# Make a lookup table of openair files where the file.wav is the key and the value is the path.
openair_lookup = {}
for subfolder in os.listdir(cfg.openAIR):
    if not os.path.isdir(os.path.join(cfg.openAIR, subfolder)): continue
    for subsubfolder in os.listdir(os.path.join(cfg.openAIR, subfolder)):
        if not os.path.isdir(os.path.join(cfg.openAIR, subfolder, subsubfolder)): continue
        for file in os.listdir(os.path.join(cfg.openAIR, subfolder, subsubfolder)):
            if not file.endswith(".wav"): continue
            openair_lookup[file] = os.path.join(cfg.openAIR, subfolder, subsubfolder, file)


with open(cfg.metadata_file, encoding="utf-8") as f:
    csv_reader = csv.reader(f, delimiter=",")
    next(csv_reader)

    for row in tqdm(csv_reader, desc="Processing RIRs", unit="RIRs"):

        file_name = row[0]
        if not file_name.endswith(('.wav', '.flac', '.mp3')):
            continue

        # Resolve correct path
        if file_name in openair_lookup:
            rir_path = openair_lookup[file_name]
        else:
            rir_path = os.path.join(cfg.MIT_AUDIO, file_name)

        # Check if file actually exists before attempting API call
        if not os.path.exists(rir_path):
            tqdm.write(f"Skipping: Local file not found at {rir_path}")
            continue

        audio_file = None  # Safely scope variable outside try block

        try:
            # 1. Upload file
            audio_file = client.files.upload(file=rir_path)

            # 2. Generate content
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[
                    audio_file,
                    f"Analyze the acoustic properties of this RIR ({file_name}) and generate a single word tag describing the location like (Hallway, Church, etc.)" 
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=RIRCaptions,
                    temperature=0.7
                )
            )

            # 3. Parse and append output
            if response.text:
                caption_data = json.loads(response.text)
                results.append(caption_data)
            else:
                tqdm.write(f"Warning: Empty response received for {file_name}")

        except Exception as e:
            # Use tqdm.write so error prints don't mess up the progress bar layout
            tqdm.write(f"Error processing {rir_path}: {e}")

        finally:
            # Safely attempt deletion only if upload succeeded
            if audio_file is not None:
                try:
                    client.files.delete(name=audio_file.name)
                except Exception as cleanup_err:
                    tqdm.write(f"Failed to delete remote file {audio_file.name}: {cleanup_err}")

        # Rate-limiting delay
        time.sleep(2)
# Save the final dataset
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Done! Dataset saved to {OUTPUT_FILE}")