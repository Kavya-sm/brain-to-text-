import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import h5py
import os
from collections import Counter
import json

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

TARGET_FEATURE_DIM = 512

def standardize_features(features, target_dim=TARGET_FEATURE_DIM):
    if features.shape[0] == target_dim:
        return features
    if features.shape[0] > target_dim:
        return features[:target_dim]
    else:
        padded = np.zeros((target_dim, features.shape[1]))
        padded[:features.shape[0]] = features
        return padded

def custom_collate_fn(batch):
    max_seq_len = max(item['neural_features'].shape[0] for item in batch)
    
    padded_neural = []
    input_lengths = []
    
    for item in batch:
        seq_len = item['neural_features'].shape[0]
        padded = torch.zeros(max_seq_len, TARGET_FEATURE_DIM)
        padded[:seq_len] = item['neural_features']
        padded_neural.append(padded)
        input_lengths.append(seq_len)
    
    neural_features = torch.stack(padded_neural)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    
    # Handle text targets
    max_text_len = max(len(item['text_targets']) for item in batch)
    padded_texts = []
    text_lengths = []
    
    for item in batch:
        text_len = len(item['text_targets'])
        padded = torch.full((max_text_len,), 0, dtype=torch.long)
        padded[:text_len] = item['text_targets']
        padded_texts.append(padded)
        text_lengths.append(text_len)
    
    text_targets = torch.stack(padded_texts)
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    
    return {
        'neural_features': neural_features,
        'text_targets': text_targets,
        'input_lengths': input_lengths,
        'target_lengths': text_lengths,
    }

class BrainToTextDataset(Dataset):
    def __init__(self, data_files, max_length=2000, vocab_size=5000, is_training=True, build_vocab=True, vocab=None):
        self.data_files = data_files
        self.max_length = max_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        
        self.data = self.load_all_data()
        
        if build_vocab:
            if is_training:
                self.tokenizer = self.build_vocabulary()
            elif vocab is not None:
                self.tokenizer = vocab
            else:
                self.tokenizer = self.load_vocabulary()
        else:
            self.tokenizer = vocab if vocab is not None else {}
            
    def load_h5py_file(self, file_path):
        data = {
            'neural_features': [],
            'sentence_label': [],
        }
        
        try:
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                for key in keys:
                    g = f[key]
                    
                    neural_features = g['input_features'][:]
                    sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None

                    data['neural_features'].append(neural_features)
                    data['sentence_label'].append(sentence_label)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            
        return data
    
    def load_all_data(self):
        all_data = {
            'neural_features': [],
            'sentence_label': [],
        }
        
        for file_path in self.data_files:
            print(f"Loading {file_path}...")
            file_data = self.load_h5py_file(file_path)
            
            standardized_features = []
            for features in file_data['neural_features']:
                standardized = standardize_features(features, TARGET_FEATURE_DIM)
                standardized_features.append(standardized)
            
            all_data['neural_features'].extend(standardized_features)
            all_data['sentence_label'].extend(file_data['sentence_label'])
            
        print(f"Loaded {len(all_data['neural_features'])} samples")
        print(f"Feature dimensions after standardization: {512}")
        
        return all_data
    
    def build_vocabulary(self):
        all_sentences = []
        for sentence in self.data['sentence_label']:
            if sentence is not None:
                try:
                    if isinstance(sentence, bytes):
                        sentence_str = sentence.decode('utf-8')
                    else:
                        sentence_str = str(sentence)
                    all_sentences.append(sentence_str)
                except:
                    continue
        
        # Use character-level tokenization for better learning
        char_counter = Counter()
        for sentence in all_sentences:
            # Clean and lowercase the text
            clean_sentence = sentence.lower().strip()
            char_counter.update(clean_sentence)
        
        # Add essential punctuation and space
        essential_chars = [' ', '.', ',', '!', '?', "'", '"', '-', ';', ':', '\n', '\t']
        for char in essential_chars:
            if char not in char_counter:
                char_counter[char] = 1
        
        # Build vocabulary with most common characters
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        for idx, (char, _) in enumerate(char_counter.most_common(self.vocab_size - 4)):
            vocab[char] = idx + 4
            
        self.vocab = vocab
        print(f"Vocabulary built with {len(vocab)} tokens")
        print(f"Sample tokens: {list(vocab.keys())[:20]}")
        return vocab
    
    def load_vocabulary(self):
        if hasattr(self, 'vocab'):
            return self.vocab
        else:
            vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
            print("Using default vocabulary")
            return vocab
    
    def text_to_tokens(self, text):
        if text is None or self.tokenizer is None:
            return torch.tensor([1, 2])
        
        try:
            if isinstance(text, bytes):
                text_str = text.decode('utf-8')
            else:
                text_str = str(text)
            
            # Character-level tokenization
            clean_text = text_str.lower().strip()
            tokens = [self.tokenizer.get(char, self.tokenizer['<unk>']) for char in clean_text]
            return torch.tensor([self.tokenizer['<sos>']] + tokens + [self.tokenizer['<eos>']])
        except:
            return torch.tensor([1, 2])
    
    def __len__(self):
        return len(self.data['neural_features'])
    
    def __getitem__(self, idx):
        neural_features = self.data['neural_features'][idx]
        sentence_label = self.data['sentence_label'][idx]
        
        neural_features = neural_features.T
        
        if neural_features.shape[0] > self.max_length:
            neural_features = neural_features[:self.max_length]
        
        neural_tensor = torch.FloatTensor(neural_features)
        text_tokens = self.text_to_tokens(sentence_label)
        
        return {
            'neural_features': neural_tensor,
            'text_targets': text_tokens,
            'input_lengths': neural_tensor.shape[0],
            'sentence_label': sentence_label
        }

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, encoder_dim=256, num_layers=2, dropout=0.2):
        super(SimpleEncoder, self).__init__()
        
        # Simple input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Simple LSTM instead of transformer
        self.lstm = nn.LSTM(
            encoder_dim, 
            encoder_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Simpler unidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        
        # Apply conv layers
        x = x.transpose(1, 2)  # (batch_size, encoder_dim, seq_len)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, encoder_dim)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        return lstm_out

class SimpleDecoder(nn.Module):
    def __init__(self, encoder_dim, vocab_size, hidden_dim=512, num_layers=2, dropout=0.2):
        super(SimpleDecoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
        # LSTM input size: hidden_dim (embedding) only, no attention concatenation
        self.lstm = nn.LSTM(
            hidden_dim,  # Only embedding, no attention context
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (separate from LSTM input)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        
        # Output layer combines LSTM output and attention context
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # LSTM_out + attention_context
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Project encoder outputs
        encoder_outputs_proj = self.encoder_proj(encoder_outputs)
        
        # Initialize hidden state
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.hidden_dim, device=device)
        
        hidden = (h0, c0)
        
        if targets is not None:
            # Training mode
            seq_len = targets.size(1)
            outputs = []
            
            # Embed entire sequence
            embedded = self.embedding(targets)
            embedded = self.dropout(embedded)
            
            for t in range(seq_len):
                # Get current input
                if t == 0:
                    input_emb = embedded[:, t:t+1]
                else:
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        input_emb = embedded[:, t:t+1]
                    else:
                        input_emb = self.embedding(next_token)
                
                # LSTM forward with just embedding
                lstm_out, hidden = self.lstm(input_emb, hidden)
                
                # Apply attention using LSTM output
                attn_out, _ = self.attention(lstm_out, encoder_outputs_proj, encoder_outputs_proj)
                
                # Combine LSTM output with attention context for final prediction
                combined = torch.cat([lstm_out, attn_out], dim=-1)
                
                # Output projection
                output = self.output_layer(combined.squeeze(1))
                outputs.append(output)
                
                # Get next token for teacher forcing
                next_token = output.argmax(-1).unsqueeze(1)
            
            return torch.stack(outputs, dim=1)
        else:
            # Inference mode
            outputs = []
            input_token = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)  # <sos>
            
            for step in range(max_length):
                # Embed input
                embedded = self.embedding(input_token)
                embedded = self.dropout(embedded)
                
                # LSTM forward
                lstm_out, hidden = self.lstm(embedded, hidden)
                
                # Apply attention
                attn_out, _ = self.attention(lstm_out, encoder_outputs_proj, encoder_outputs_proj)
                
                # Combine for output
                combined = torch.cat([lstm_out, attn_out], dim=-1)
                
                # Output projection
                output = self.output_layer(combined.squeeze(1))
                outputs.append(output)
                
                # Sample next token with temperature
                temperature = 0.8
                logits = output / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Remove special tokens from sampling
                probs[:, 0] = 0  # <pad>
                probs[:, 1] = 0  # <sos>
                if step < 5:  # Don't allow <eos> in first 5 steps
                    probs[:, 2] = 0
                
                # Renormalize
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, 1)
                
                # Stop if all sequences generated <eos>
                if (next_token == 2).all() and step >= 5:
                    break
                
                input_token = next_token
            
            return torch.stack(outputs, dim=1) if outputs else torch.zeros(batch_size, 1, self.output_layer[-1].out_features, device=device)

class EEGToTextModel(nn.Module):
    def __init__(self, input_dim=512, encoder_dim=256, vocab_size=5000, dropout=0.2):
        super(EEGToTextModel, self).__init__()
        
        self.encoder = SimpleEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=2,
            dropout=dropout
        )
        
        self.decoder = SimpleDecoder(
            encoder_dim=encoder_dim,
            vocab_size=vocab_size,
            hidden_dim=512,
            num_layers=2,
            dropout=dropout
        )
        
    def forward(self, eeg_input, target_text=None, teacher_forcing_ratio=0.5):
        encoder_outputs = self.encoder(eeg_input)
        decoder_output = self.decoder(encoder_outputs, target_text, teacher_forcing_ratio)
        
        return {
            'decoder_output': decoder_output
        }

class EEGToTextTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Higher learning rate for faster learning
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Standard cross entropy without label smoothing initially
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            eeg_data = batch['neural_features'].to(self.device)
            text_targets = batch['text_targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Start with high teacher forcing, gradually reduce
            teacher_forcing_ratio = max(0.9 - epoch * 0.05, 0.5)
            
            outputs = self.model(
                eeg_data, 
                text_targets[:, :-1], 
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            decoder_output = outputs['decoder_output']
            loss = self.ce_loss(
                decoder_output.reshape(-1, decoder_output.size(-1)),
                text_targets[:, 1:].reshape(-1)
            )
            
            loss.backward()
            
            # Gentle gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
                
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                eeg_data = batch['neural_features'].to(self.device)
                text_targets = batch['text_targets'].to(self.device)
                
                outputs = self.model(eeg_data, text_targets[:, :-1], teacher_forcing_ratio=1.0)
                
                decoder_output = outputs['decoder_output']
                loss = self.ce_loss(
                    decoder_output.reshape(-1, decoder_output.size(-1)),
                    text_targets[:, 1:].reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        patience = 8
        patience_counter = 0
        
        print("Starting training...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = {
                    'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch
                }
                print('✓ Best model saved!')
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'No improvement for {patience_counter} epochs')
                
                if patience_counter >= patience:
                    print(f'Early stopping after {epoch + 1} epochs')
                    break
            
            print('-' * 60)

def get_data_files(base_path):
    sessions = [
        't15.2023.09.29',
        't15.2023.12.03', 
        't15.2024.03.17',
        't15.2025.04.13'
    ]
    
    train_files = []
    val_files = []
    test_files = []
    
    for session in sessions:
        session_path = os.path.join(base_path, session)
        if os.path.exists(session_path):
            train_file = os.path.join(session_path, 'data_train.hdf5')
            val_file = os.path.join(session_path, 'data_val.hdf5') 
            test_file = os.path.join(session_path, 'data_test.hdf5')
            
            if os.path.exists(train_file):
                train_files.append(train_file)
            if os.path.exists(val_file):
                val_files.append(val_file)
            if os.path.exists(test_file):
                test_files.append(test_file)
    
    print(f"Found {len(train_files)} train files, {len(val_files)} val files, {len(test_files)} test files")
    return train_files, val_files, test_files

class EEGToTextPipeline:
    def __init__(self, encoder_dim=256, vocab_size=100):
        self.model = EEGToTextModel(
            input_dim=TARGET_FEATURE_DIM,
            encoder_dim=encoder_dim,
            vocab_size=vocab_size
        )
        self.vocab_size = vocab_size
        self.idx_to_char = None
        self.trainer = None
        self.tokenizer = None
        
    def train(self, base_data_path, num_epochs=20, batch_size=32, max_length=2000):
        train_files, val_files, _ = get_data_files(base_data_path)
        
        if not train_files:
            print("No training files found!")
            return
            
        print("Creating datasets...")
        train_dataset = BrainToTextDataset(train_files, max_length=max_length, vocab_size=self.vocab_size, is_training=True)
        val_dataset = BrainToTextDataset(val_files, max_length=max_length, vocab_size=self.vocab_size, is_training=False, build_vocab=False, vocab=train_dataset.tokenizer)
        
        self.idx_to_char = {v: k for k, v in train_dataset.tokenizer.items()}
        self.tokenizer = train_dataset.tokenizer
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=0, collate_fn=custom_collate_fn)
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        print(f"Vocabulary size: {len(self.tokenizer)}")
        
        self.trainer = EEGToTextTrainer(self.model, train_loader, val_loader, device)
        self.trainer.train(num_epochs)
        
    def load_best_model(self):
        if hasattr(self.trainer, 'best_model_state'):
            self.model.load_state_dict(self.trainer.best_model_state['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            print("Best model loaded!")
        else:
            print("No best model found!")
        
    def tokens_to_text(self, tokens):
        if self.idx_to_char is None:
            return "Vocabulary not loaded"
        
        text = ''.join([self.idx_to_char.get(t.item(), '') for t in tokens])
        text = text.replace('<sos>', '').replace('<eos>', '').replace('<pad>', '').replace('<unk>', '')
        return text.strip()
    
    def predict_with_confidence(self, eeg_input, temperature=0.8, max_length=50):
        if self.model is None:
            return "", 0.0
        
        self.model.eval()
        with torch.no_grad():
            if len(eeg_input.shape) == 2:
                eeg_input = eeg_input.unsqueeze(0)
            eeg_input = eeg_input.to(device)
            
            outputs = self.model(eeg_input)
            decoder_output = outputs['decoder_output'][0]
            
            predictions = []
            confidences = []
            
            for t in range(min(decoder_output.size(0), max_length)):
                logits = decoder_output[t]
                probs = F.softmax(logits / temperature, dim=-1)
                
                # Remove special tokens from consideration
                probs[0] = 0  # <pad>
                probs[1] = 0  # <sos>
                if t < 5:  # Don't allow <eos> early
                    probs[2] = 0
                
                # Renormalize
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                
                predicted_idx = torch.multinomial(probs, 1).item()
                confidence = probs[predicted_idx].item()
                
                if predicted_idx == 2 and t >= 5:  # <eos>
                    break
                    
                if predicted_idx not in [0, 1, 2]:  # Skip special tokens
                    predictions.append(predicted_idx)
                    confidences.append(confidence)
            
            prediction = self.tokens_to_text(torch.tensor(predictions))
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return prediction, avg_confidence

def main():
    BASE_DATA_PATH = "/Users/kavya/Downloads/hdf5_data_final"
    
    pipeline = EEGToTextPipeline(
        encoder_dim=256,
        vocab_size=100  # Smaller vocabulary for character-level
    )
    
    print("Starting training with simplified model...")
    pipeline.train(BASE_DATA_PATH, num_epochs=20, batch_size=32)
    
    pipeline.load_best_model()
    
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    train_files, val_files, _ = get_data_files(BASE_DATA_PATH)
    if val_files:
        test_dataset = BrainToTextDataset(val_files[:1], max_length=2000, vocab_size=pipeline.vocab_size, is_training=False, build_vocab=False, vocab=pipeline.tokenizer)
        if len(test_dataset) > 0:
            for i in range(min(5, len(test_dataset))):
                sample = test_dataset[i]
                prediction, confidence = pipeline.predict_with_confidence(sample['neural_features'], temperature=0.8)
                if 'sentence_label' in sample and sample['sentence_label'] is not None:
                    try:
                        ground_truth = sample['sentence_label'].decode('utf-8') if isinstance(sample['sentence_label'], bytes) else str(sample['sentence_label'])
                        print(f"\nSample {i+1}:")
                        print(f"  Predicted: '{prediction}'")
                        print(f"  Truth:     '{ground_truth}'")
                        print(f"  Confidence: {confidence:.3f}")
                    except:
                        print(f"\nSample {i+1}: '{prediction}' (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()
