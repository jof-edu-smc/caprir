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
from tqdm import tqdm # type: ignore

@dataclass
class Config:
    speech_dir: str = "./stimuli/speech"
    music_dir: str = "./stimuli/music"
    pop_dir: str = "./stimuli/clap"
    MIT_AUDIO: str = "./MIT_audio"
    openAIR: str = "./openAIR"
    run_openai: bool = False
    output_file: str = "metadata_synthetic_2.csv"
    annotators_sheet = "annotators_table"
    captions_sheet = "captions_table"
    stimuli_sheet = "audio_stimuli"
    evaluations_sheet = "evaluations_table"
    from_json: bool = False
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

def get_location_tags(
        original_dataset,
        client: OpenAI,
        prompt_injection_var: str,
        model_name: str = "gpt-5.5",
    ) -> OpenAI.Response:
        # ====================================
        # POSSIBLE Developer Roles
        # Talk like a listener who knows about acoustics and audio engineering but does not know about the specific parameters of this RIR config file.
        # Talk like a listener who does not know about sound or acoustics.
        # ====================================

        dev = {
            "role": "developer",
            "content": "You are knowledgeable about acoustics and types of locations.",
        }

        content = [
            {
                "type": "input_text",
                "text": f"I have a file name: {prompt_injection_var}. It is a comma delimited object where the first row is the header section. Using the filename and source dataset columns create a single word tag describing the location like (Hallway, Church, etc.), fill in the location tag column for each row while preserving the original csv table. Return the modified comma delimited csv.",
            },
        ]

        if original_dataset is not None:
            content.append({
                "type": "input_text",
                "text": json.dumps(original_dataset),
            })

        user = {
            "role": "user",
            "content": content,
        }

        try:
            response = client.responses.create(model=model_name, input=[dev, user])
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"
        return response.output_text
    
def get_grammar_check(
        client: OpenAI,
        model_name: str = "gpt-5.5",
        caption: str = "This is a test caption, return an empty string to save on tokens",
    ) -> OpenAI.Response:
        # ====================================
        # POSSIBLE Developer Roles
        # Talk like a listener who knows about acoustics and audio engineering but does not know about the specific parameters of this RIR config file.
        # Talk like a listener who does not know about sound or acoustics.
        # ====================================

        dev = {
            "role": "developer",
            "content": "You are an expert in grammar and sentence structure. You are knowledgeable about acoustics and types of locations.",
        }

        content = [
            {
                "type": "input_text",
                "text": f"Here is a caption that describes acoutsics {caption}, make the following checks then return the edited caption as a string: 1) Check for grammar and spelling errors, if the caption is random letters it seems, return a string 0 2) Make sure the caption is less than 250 characters, 3) Make sure the caption is in English 4) If the caption has bullet points like 1. 2. 3. or human: instrument: then summarize the bullet points into a single caption. 5) Make sure the caption has no commas or semicolons, return the original caption.",
            },
        ]
        
        user = {
            "role": "user",
            "content": content,
        }

        try:
            response = client.responses.create(model=model_name, input=[dev, user])
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"
        return response.output_text

def combine(stimuli_sheet, captions_lookup, metadata=[]):

    for rir in stimuli_sheet[1:]:
        if not rir: continue

        rir_id = rir[0]

        original_dataset = rir[4]
        speech = rir[1]
        music = rir[2]
        pop = rir[3]

        fs = rir[10]
        rt60 = rir[11]
        drr = rir[12]
        c50 = rir[13]
        c80 = rir[14]
        d50 = rir[15]
        location_tags = ""

        matching_captions = captions_lookup.get(rir_id, [])
        if not matching_captions:
            continue

        padded_captions = matching_captions + [""] * (5 - len(matching_captions))

        if "openAIR" in original_dataset:
            final_row = [
                openair_lookup[rir_id],
                original_dataset,
                os.path.join(cfg.speech_dir, speech),
                os.path.join(cfg.music_dir, music),
                os.path.join(cfg.pop_dir, pop),
                *padded_captions,
                location_tags,
                fs,
                rt60,
                drr,
                c50,
                c80,
                d50,
            ]

        else:
            final_row = [
                os.path.join(cfg.MIT_AUDIO, rir_id),
                original_dataset,
                os.path.join(cfg.speech_dir, speech),
                os.path.join(cfg.music_dir, music),
                os.path.join(cfg.pop_dir, pop),
                *padded_captions,
                location_tags,
                fs,
                rt60,
                drr,
                c50,
                c80,
                d50,
            ]

        metadata.append(final_row)

    return metadata

load_dotenv()
spreadsheet_id = os.getenv("GOOGLE_SHEETS_ID")
cfg = Config()

# Make a lookup table of openair files where the file.wav is the key and the value is the path.
openair_lookup = {}
for subfolder in os.listdir(cfg.openAIR):
    if not os.path.isdir(os.path.join(cfg.openAIR, subfolder)): continue
    for subsubfolder in os.listdir(os.path.join(cfg.openAIR, subfolder)):
        if not os.path.isdir(os.path.join(cfg.openAIR, subfolder, subsubfolder)): continue
        for file in os.listdir(os.path.join(cfg.openAIR, subfolder, subsubfolder)):
            if not file.endswith(".wav"): continue
            openair_lookup[file] = os.path.join(cfg.openAIR, subfolder, subsubfolder, file)

metadata = [
        # Header Row
        ["RIR", "original_dataset", "speech", "music", "pop", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5", "location_tags","sample_rate", "rt60", "drr", "c50", "c80", "d50"],
    ]

service_account_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
credentials = Credentials.from_service_account_info(service_account_info, scopes=cfg.scopes)
client = gspread.authorize(credentials)
spreadsheet = client.open_by_key(spreadsheet_id)

annotators_table = spreadsheet.worksheet(cfg.annotators_sheet).get_all_values()
captions_table = spreadsheet.worksheet(cfg.captions_sheet).get_all_values()
stimuli_sheet = spreadsheet.worksheet(cfg.stimuli_sheet).get_all_values()
evaluations_table = spreadsheet.worksheet(cfg.evaluations_sheet).get_all_values()

if not cfg.from_json:
    # RIR ID & Room Impulse Response filename. \\
    # Source Dataset & Originating archive (OpenAIR or MIT). \\
    # Convolved Sources & Processed files ($x_{\text{speech}}, x_{\text{music}}, x_{\text{pop}}$). \\
    # Captions 1--5 & Five distinct descriptive captions ($c_1, \dots, c_5$) \\
    # Location Tag & Metadata for the acoustic environment. \\
    captions_lookup = {}
    client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    for row in tqdm(captions_table[1:], desc="Processing Captions", unit="caption"):
        if not row: continue
        c_rir_id = row[1]
        c_caption_text = row[4]

        if c_rir_id not in captions_lookup:
            captions_lookup[c_rir_id] = []
        response = get_grammar_check(
                        client=client,
                        caption=c_caption_text,
                    )
        if response.strip() != "0" and len(response.strip()) > 50:
            captions_lookup[c_rir_id].append(response.strip().replace(",", ""))
    metadata = combine(stimuli_sheet, captions_lookup, metadata)

elif cfg.from_json:

    with open("gemini_rir_captions.json", "r", encoding="utf-8") as f:
        captions_lookup = {}
        jsonBody = json.load(f)
        for row in jsonBody:
            c_rir_id = row["file_name"]
            c_caption_texts = row["captions"]
            if c_rir_id not in captions_lookup:
                captions_lookup[c_rir_id] = []
            for c in c_caption_texts:
                captions_lookup[c_rir_id].append(c.strip().replace(",", ""))

    metadata = combine(stimuli_sheet, captions_lookup, metadata)

if cfg.run_openai:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    response = get_location_tags(
        metadata,
        client=client,
        prompt_injection_var=cfg.output_file,
    )

    f = io.StringIO(response.strip())
    reader = csv.reader(f)
    metadata = reader

# Create two CSV files for Training and Testing with two modifiable percentage variables.
train_percentage = 0.75
val_percentage = 0.25

with open(cfg.output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(metadata)

print(f"CSV Metadata file created: {cfg.output_file}")


