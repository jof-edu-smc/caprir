import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import Dataset, DataLoader, random_split  # type: ignore
from transformers import AutoModel, PreTrainedTokenizerFast  # type: ignore
from torch.optim import AdamW  # type: ignore
from torch.optim.lr_scheduler import LambdaLR  # type: ignore
import torchaudio  # type: ignore
import librosa  # type: ignore
import argparse
import time

# ---------------------------------------------------------
# 1. Dataset Definition
# ---------------------------------------------------------
class AudioCaptionDataset(Dataset):
    def __init__(self, data, tokenizer, sample_rate, device="cuda", source="caprir"):
        """
        data_list: ["RIR", "original_dataset", "speech", "music", "pop", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5", "location_tags"],
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
                            "temporal_tag": 3
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
                    "temporal_tag": 3
                })
                
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        audio_file = item['audio_path']
        caption = item['caption']
        temporal_tag = item.get('temporal_tag', 0)

        # Reusing audio loading logic
        if audio_file.endswith(".flac"):
            wav, sr = librosa.load(audio_file, sr=16000)
            wav = torch.from_numpy(wav).unsqueeze(0)
        else:
            wav, sr = torchaudio.load(audio_file)

        wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.size(0) > 1:
            wav = wav.mean(0).unsqueeze(0)

        min_samples = int(0.5 * self.sample_rate)
        if wav.size(1) < min_samples:
            pad_len = min_samples - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        # Build target sequence with explicit <start>/<end> and <pad>
        START, END, PAD, MAX_LEN = 1, 2, 0, 50
        body = self.tokenizer(caption, add_special_tokens=False)["input_ids"]
        body = body[: MAX_LEN - 2]  # leave room for start + end
        ids = [START] + body + [END]
        ids = ids + [PAD] * (MAX_LEN - len(ids))  # right-pad to fixed length

        # input_ids will be our labels for CrossEntropy
        labels = torch.tensor(ids, dtype=torch.long)

        return {
            "audio": wav.squeeze(0),  # Remove channel dim for batching
            "audio_length": wav.size(1),
            "temporal_tag": temporal_tag,
            "labels": labels,
            "labels_text": caption
        }


# ---------------------------------------------------------
# 2. Collate Function
# ---------------------------------------------------------
def collate_fn(batch):
    # Find max audio length in this batch to pad
    max_audio_len = max([item['audio_length'] for item in batch])

    audios = []
    audio_lengths = []
    temporal_tags = []
    labels = []
    labels_text = []

    for item in batch:
        pad_amount = max_audio_len - item['audio_length']
        padded_audio = torch.nn.functional.pad(item['audio'], (0, pad_amount))

        audios.append(padded_audio)
        audio_lengths.append(item['audio_length'])
        temporal_tags.append(item['temporal_tag'])
        labels.append(item['labels'])
        labels_text.append(item['labels_text'])
    return {
        "audio": torch.stack(audios),
        "audio_length": torch.tensor(audio_lengths),
        "temporal_tag": torch.tensor(temporal_tags),
        "labels": torch.stack(labels),
        "labels_text": labels_text
    }


def verify_pad_and_indices(tokenizer, model, dataset=None):
    EXPECTED = {"pad": 0, "start": 1, "end": 2}

    print("=== Tokenizer special tokens ===")
    print(f"pad_token   = {tokenizer.pad_token!r} -> id {tokenizer.pad_token_id}")
    print(f"bos_token   = {tokenizer.bos_token!r} -> id {tokenizer.bos_token_id}")
    print(f"eos_token   = {tokenizer.eos_token!r} -> id {tokenizer.eos_token_id}")
    print(f"unk_token   = {tokenizer.unk_token!r} -> id {tokenizer.unk_token_id}")
    print(f"vocab_size  = {tokenizer.vocab_size} | model vocab = {model.config.vocab_size}")

    assert tokenizer.pad_token_id == EXPECTED["pad"], (
        f"PAD id {tokenizer.pad_token_id} != model pad_idx {EXPECTED['pad']}. "
        f"Loss masking will be wrong."
    )

    sample = "a dog is barking in the background"
    if dataset is None:
        enc = tokenizer(
            sample, padding="max_length", max_length=50,
            truncation=True, return_tensors="pt"
        )
        ids = enc["input_ids"].squeeze(0)
    else:
        enc = dataset.__getitem__(0)["labels"]
        ids = enc.squeeze(0)

    print("\n=== Sample encoding ===")
    print("first 10 ids:", ids[:10].tolist())
    print("last 10 ids :", ids[-10:].tolist())
    print("decoded     :", tokenizer.decode(ids))

    real_len = int((ids != EXPECTED["pad"]).sum())
    print(f"non-pad length: {real_len}")
    print(f"starts with start_idx({EXPECTED['start']})? {ids[0].item() == EXPECTED['start']}")
    print(f"contains end_idx({EXPECTED['end']})?       {EXPECTED['end'] in ids.tolist()}")
    pad_region_ok = bool((ids[real_len:] == EXPECTED["pad"]).all())
    print(f"tail is all pad?                            {pad_region_ok}")
    assert pad_region_ok, "Padding region contains non-pad tokens."

    assert ids.max() < model.config.vocab_size, (
        f"token id {ids.max().item()} >= vocab_size {model.config.vocab_size}"
    )
    print("\n✅ PAD token and index layout look consistent with the model.")


# ---------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------
def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Model & Tokenizer
    model = AutoModel.from_pretrained(
        "wsntxxn/cnn14rnn-tempgru-audiocaps-captioning",
        trust_remote_code=True
    ).to(device)

    for param in model.sed_model.parameters():
        param.requires_grad = False

    tokenizer = PreTrainedTokenizerFast.from_pretrained("wsntxxn/audiocaps-simple-tokenizer")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
    pad_token_id = tokenizer.pad_token_id

    model.train()

    with open(args.metadata, 'r') as f:
        metadata = f.readlines()

    dataset = AudioCaptionDataset(metadata, tokenizer, model.config.sample_rate, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    verify_pad_and_indices(tokenizer, model, dataset=dataset)

    # Setup Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Setup Loss Function with Label Smoothing (alpha = 0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)

    # Setup Learning Rate Scheduler: Linear Warm-up + Exponential Decay
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warm-up steps
    decay_gamma = 0.99  # Exponential decay factor per step post-warmup

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Exponential decay after warm-up
        return max(0.01, decay_gamma ** (current_step - warmup_steps))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Scheduled Sampling parameters (1.0 -> 0.7)
    ss_start = 1.0
    ss_end = 0.7

    print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    # Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0

        # Calculate ss_ratio for the current epoch (linearly decreasing from 1.0 to 0.7)
        if args.epochs > 1:
            current_ss_ratio = ss_start - (ss_start - ss_end) * (epoch / (args.epochs - 1))
        else:
            current_ss_ratio = ss_start

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            audio = batch['audio'].to(device)
            audio_length = batch['audio_length'].tolist()
            temporal_tag = batch['temporal_tag'].tolist()
            labels = batch['labels'].to(device)

            mel_spec = model.melspec_extractor(audio)
            log_mel_spec = model.db_transform(mel_spec)

            cap_len = (labels != pad_token_id).sum(dim=1)

            input_dict = {
                "lms": log_mel_spec,
                "wav_len": audio_length,
                "temporal_tag": torch.as_tensor(temporal_tag).to(device),
                "mode": "train",
                "cap": labels,
                "cap_len": cap_len,
                "ss_ratio": current_ss_ratio,  # Dynamic scheduled sampling ratio
            }
            # print lengths of the input_dict
            # print(f"Input dict values: {len(input_dict['lms']), len(input_dict['wav_len']), len(input_dict['temporal_tag']), len(input_dict['mode']), len(input_dict['cap']), len(input_dict['cap_len']), input_dict['ss_ratio']}")

            output = model.cap_model(input_dict)
            logits = output["logit"]

            targets = labels[:, 1:]  # Shift target tokens by 1
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] | "
                    f"Batch [{batch_idx + 1}/{len(dataloader)}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"SS Ratio: {current_ss_ratio:.3f} | "
                    f"LR: {current_lr:.6f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f} | SS Ratio: {current_ss_ratio:.3f} ---")

    # Save fine-tuned model and tokenizer
    print("Saving fine-tuned model...")
    model.save_pretrained("./wstxxn-finetuned")
    tokenizer.save_pretrained("./wstxxn-finetuned")

if __name__ == "__main__":
    # track time taken for training
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Peak learning rate (default: 5e-4)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata CSV file")
    args = parser.parse_args()

    train(args)
    time_taken = time.time() - start_time
    minutes, seconds = divmod(time_taken, 60)
    print(f"Training completed in {int(minutes)} minutes and {int(seconds)} seconds.")