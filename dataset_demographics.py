import os
import io
import csv
import json
from dotenv import load_dotenv # type: ignore
from dataclasses import dataclass
from google.oauth2.service_account import Credentials # type: ignore
import gspread # type: ignore
from openai import OpenAI # type: ignore
from pygments.lexers import data

@dataclass
class Config:
    speech_dir: str = "./stimuli/speech"
    music_dir: str = "./stimuli/music"
    pop_dir: str = "./stimuli/clap"
    MIT_AUDIO: str = "./MIT_audio"
    openAIR: str = "./openAIR"
    run_openai: bool = False
    output_file: str = "metadata_natural.csv"
    annotators_sheet = "annotators_table"
    captions_sheet = "captions_table"
    stimuli_sheet = "audio_stimuli"
    evaluations_sheet = "evaluations_table"
    from_json: bool = False
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

load_dotenv()
spreadsheet_id = os.getenv("GOOGLE_SHEETS_ID")
cfg = Config()

service_account_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
credentials = Credentials.from_service_account_info(service_account_info, scopes=cfg.scopes)
client = gspread.authorize(credentials)
spreadsheet = client.open_by_key(spreadsheet_id)

annotators_table = spreadsheet.worksheet(cfg.annotators_sheet).get_all_values()
# captions_table = spreadsheet.worksheet(cfg.captions_sheet).get_all_values()
# stimuli_sheet = spreadsheet.worksheet(cfg.stimuli_sheet).get_all_values()
# evaluations_table = spreadsheet.worksheet(cfg.evaluations_sheet).get_all_values()

# List columns
# [0,1,2,3,4,5,6,7,8,9,10]
# ['prolific_id', 'timestamp', 'phase', 'study_id', 'session_id', 'age_range', 'experience_in_audio', 'normal_hearing', 'understood_assignment', 'L-Acoustics']

unique_annotator_ids = set()
age_range_counts = {}
experience_level_counts = {}
normal_hearing_counts = {}

for row in annotators_table[1:]:
    annotator_id = row[0]
    print(f"Annotator ID: {annotator_id}")
    if annotator_id not in unique_annotator_ids:
        unique_annotator_ids.add(annotator_id)
        age_range = row[5]
        experience_level = row[6]
        normal_hearing = row[7]

        # Count age ranges
        if age_range in age_range_counts:
            age_range_counts[age_range] += 1
        else:
            age_range_counts[age_range] = 1

        # Count experience levels
        if experience_level in experience_level_counts:
            experience_level_counts[experience_level] += 1
        else:
            experience_level_counts[experience_level] = 1

        # Count normal hearing responses
        if normal_hearing in normal_hearing_counts:
            normal_hearing_counts[normal_hearing] += 1
        else:
            normal_hearing_counts[normal_hearing] = 1
            
# Print Total Number of Unique Annotators
print(f"Total Unique Annotators: {len(unique_annotator_ids)}")
# Print Age Range Counts
print("Age Range Counts:")
for age_range, count in age_range_counts.items():
    print(f"{age_range}: {count}")
# Print Experience Level Counts
print("Experience Level Counts:")
for experience_level, count in experience_level_counts.items():
    print(f"{experience_level}: {count}")
# Print Normal Hearing Counts
print("Normal Hearing Counts:")
for normal_hearing, count in normal_hearing_counts.items():
    print(f"{normal_hearing}: {count}")