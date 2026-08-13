import os
import time
import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from torch.optim import AdamW # type: ignore
import librosa # type: ignore
import argparse
import transformers # type: ignore
from huggingface_hub import hf_hub_download # type: ignore
import importlib.util

# ---------------------------------------------------------
# 1. Dynamic Model Loading
# ---------------------------------------------------------
def load_custom_whisper_class(repo_id="MU-NLPC/whisper-tiny-audio-captioning"):
    path = hf_hub_download(repo_id, "model_class.py")
    spec = importlib.util.spec_from_file_location("model_class", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.WhisperForAudioCaptioning

# ---------------------------------------------------------
# 2. Dataset Definition
# ---------------------------------------------------------
class WhisperCaptionDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 feature_extractor,
                 style_prefix="clotho > caption: ",
                 device="cuda",
                 source="caprir"
        ):
        """
        data_list: [{'audio_path': '...', 'caption': '...'}, ...]
        """
        self.device = device
        self.data_list = []
        # Check if the first row is a header row, if so, remove it
        check_first_row = data[0].strip().split(",")
        if check_first_row[0].strip() == "RIR" or check_first_row[1].strip() == "original_dataset":
            data = data[1:]
        if source == "caprir":
            for row in data:
                row = row.strip().split(",")
                if len(row) != 11 and len(row) != 17:
                    raise ValueError(f"Invalid row format: {len(row)}: {row}")

                audio_files = [row[i] for i in [0, 2, 3, 4] if row[i] != ""]
                captions = [c for c in row[5:10] if c != ""]
                for audio_file in audio_files:
                    for caption in captions:
                        self.data_list.append({
                            "audio_path": audio_file,
                            "caption": caption,
                        })
                        
        elif source == "clotho":
            for row in data: 
                row = row.strip().split(",")
                if len(row) != 6:
                    raise ValueError(f"Invalid row format: {len(row)}: {row}")
                audio_file, caption = row[0], row[1]
                self.data_list.append({
                    "audio_path": os.join("clotho/evaluation", audio_file),
                    "caption": caption,
                })

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.style_prefix = style_prefix

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        audio_file = item['audio_path']
        raw_caption = item['caption']

        # 1. Load and process audio
        audio, sr = librosa.load(audio_file, sr=16000)
        audio = audio.astype("float32")

        # The feature extractor handles padding the audio to 30 seconds automatically
        # (which is what Whisper expects by default).
        features = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.squeeze(0)  # Remove batch dim for the Dataset

        # 2. Process the text caption
        # We prepend the style prefix directly to the target text so the model
        # learns to associate the prefix with the resulting caption style.
        full_caption = self.style_prefix + raw_caption

        tokenized_text = self.tokenizer(
            full_caption,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = tokenized_text["input_ids"].squeeze(0)

        # 3. Handle ignore_index for loss calculation
        # PyTorch CrossEntropy uses -100 to ignore padding tokens during loss calculation.
        # We replace all tokenizer pad tokens with -100.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_features": features,
            "labels": labels,
            "labels_text": full_caption
        }

# ---------------------------------------------------------
# 3. Collate Function
# ---------------------------------------------------------
def collate_fn(batch):
    # Since WhisperFeatureExtractor pads to 30s and we used padding="max_length"
    # for text, tensors are already uniformly sized. We just stack them.
    input_features = torch.stack([item["input_features"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    labels_text = [item["labels_text"] for item in batch]

    return {
        "input_features": input_features,
        "labels": labels,
        "labels_text": labels_text
    }
# ---------------------------------------------------------
# 3.5. Freeze parameters
# ---------------------------------------------------------
def freeze_all_except_fc1(model):
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        # Keep gradients ONLY for fc1 layers in both encoder and decoder
        if ".fc1." in name:
            # print(f"Unfreezing {name}")
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(
        f"Trainable params: {trainable_params:,} | "
        f"Total params: {all_param:,} | "
        f"Percentage trainable: {100 * trainable_params / all_param:.2f}%"
    )
    return model

# ---------------------------------------------------------
# 4. Main Training Loop
# ---------------------------------------------------------
def train(args):

    torch.manual_seed(42)
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # checkpoint = "openai/whisper-tiny"
    # model = transformers.WhisperForConditionalGeneration.from_pretrained(checkpoint).to(device)

    checkpoint = "MU-NLPC/whisper-tiny-audio-captioning"
    WhisperForAudioCaptioning = load_custom_whisper_class(checkpoint)
    model = WhisperForAudioCaptioning.from_pretrained(checkpoint).to(device)

    # Load Tokenizer, and Feature Extractor
    tokenizer = transformers.WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(checkpoint)

    model = freeze_all_except_fc1(model)
    model.train()
    print(f"Training on: {device} using Model: {checkpoint}")
    # Create dummy data for demonstration (Replace with actual dataset)
    with open(args.metadata, 'r') as f:
        metadata = f.readlines()

    # Setup Dataset & DataLoader
    dataset = WhisperCaptionDataset(metadata, tokenizer, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Setup Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Prepare caption style
    style_prefix = "clotho > caption: "
    style_prefix_tokens = tokenizer("", text_target=style_prefix, return_tensors="pt", add_special_tokens=False).labels

    # Training Loop
    for epoch in range(args.epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            # --- FORWARD PASS ---
            # Standard Hugging Face Seq2Seq models will automatically calculate
            # Cross-Entropy loss if 'labels' are provided.
            outputs = model(
                input_features=input_features,
                forced_ac_decoder_ids=style_prefix_tokens,
                labels=labels
            )

            # --- LOSS CALCULATION ---
            loss = outputs.loss

            # --- BACKWARD PASS ---
            loss.backward()

            # Gradient clipping is often highly recommended for Seq2Seq models
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            print(
                f"Epoch [{epoch + 1}/{args.epochs}] | Batch [{batch_idx + 1}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch + 1} Complete | Average Loss: {avg_loss:.4f} ---")

    # Save the fine-tuned model
    save_dir = "./whisper-finetuned"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    feature_extractor.save_pretrained(save_dir)
    print("Model saved to", save_dir)


if __name__ == "__main__":
    # track time taken for training
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate (usually lower for Seq2Seq)")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to the metadata CSV file for the dataset")
    args = parser.parse_args()

    train(args)
    time_taken = time.time() - start_time
    minutes, seconds = divmod(time_taken, 60)
    print(f"Training completed in {int(minutes)} minutes and {int(seconds)} seconds.")