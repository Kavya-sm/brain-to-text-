# brain_to_text_trainer.py
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os
import random
from collections import Counter

# ==================== CONFIGURATION ====================
DATA_DIR = "/Users/kavya/Downloads/hdf5_data_final"
BATCH_SIZE = 8  # Reduced for stability
HIDDEN_DIM = 128  # Reduced for stability
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# ==================== DATA LOADING & PROCESSING ====================

def load_h5py_file(file_path):
    """Load HDF5 file containing neural data and labels"""
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f"Loading {len(keys)} trials from {file_path}")

            for key in keys:
                g = f[key]
                neural_features = g['input_features'][:]
                n_time_steps = g.attrs['n_time_steps']
                seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
                seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
                sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
                session = g.attrs['session']
                block_num = g.attrs['block_num']
                trial_num = g.attrs['trial_num']

                data['neural_features'].append(neural_features)
                data['n_time_steps'].append(n_time_steps)
                data['seq_class_ids'].append(seq_class_ids)
                data['seq_len'].append(seq_len)
                data['sentence_label'].append(sentence_label)
                data['session'].append(session)
                data['block_num'].append(block_num)
                data['trial_num'].append(trial_num)
                
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
        
    return data

def extract_sentence(sentence_array):
    """Extract string from sentence byte array"""
    if sentence_array is None:
        return ""
    try:
        end_idx = np.argwhere(sentence_array == 0)
        if len(end_idx) > 0:
            end_idx = end_idx[0, 0]
        else:
            end_idx = len(sentence_array)
        sentence = ''.join(chr(c) for c in sentence_array[:end_idx])
        return sentence
    except:
        return ""

def build_vocabulary(data):
    """Build vocabulary from all sentences in the data"""
    all_sentences = []
    for i in range(len(data['sentence_label'])):
        sentence = extract_sentence(data['sentence_label'][i])
        if sentence:
            all_sentences.append(sentence.lower())
    
    # Count word frequencies
    word_counts = Counter()
    for sentence in all_sentences:
        words = sentence.split()
        word_counts.update(words)
    
    # Create vocabulary (most common words + special tokens)
    vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + [word for word, count in word_counts.most_common(500)]  # Reduced vocab size
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    
    print(f"Built vocabulary with {len(vocab)} words")
    print(f"Most common words: {list(word_counts.most_common(10))}")
    
    return word_to_idx, idx_to_word, vocab

def sentence_to_tensor(sentence, word_to_idx, max_length=30):  # Reduced max_length
    """Convert sentence to tensor of word indices"""
    words = sentence.lower().split()
    # Add SOS and EOS tokens
    indices = [word_to_idx['<SOS>']] + [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words] + [word_to_idx['<EOS>']]
    
    # Pad to max_length
    if len(indices) < max_length:
        indices.extend([word_to_idx['<PAD>']] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
        indices[-1] = word_to_idx['<EOS>']  # Ensure EOS at end
    
    return torch.tensor(indices, dtype=torch.long)

# ==================== DATASET CLASS ====================

class BrainToTextDataset(Dataset):
    def __init__(self, data_dict, word_to_idx, max_length=30):
        self.data = data_dict
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data['neural_features'])
    
    def __getitem__(self, idx):
        neural_features = torch.FloatTensor(self.data['neural_features'][idx])
        sentence = extract_sentence(self.data['sentence_label'][idx])
        target_tensor = sentence_to_tensor(sentence, self.word_to_idx, self.max_length)
        
        return {
            'neural_features': neural_features,
            'target_sentence': target_tensor,
            'sentence_text': sentence,
            'seq_len': self.data['n_time_steps'][idx],
            'session': self.data['session'][idx],
            'block_num': self.data['block_num'][idx],
            'trial_num': self.data['trial_num'][idx],
        }

def custom_collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    neural_features = [item['neural_features'] for item in batch]
    target_sentences = torch.stack([item['target_sentence'] for item in batch])
    sentence_texts = [item['sentence_text'] for item in batch]
    seq_lens = [item['seq_len'] for item in batch]
    sessions = [item['session'] for item in batch]
    block_nums = [item['block_num'] for item in batch]
    trial_nums = [item['trial_num'] for item in batch]
    
    # Pad neural features
    neural_features_padded = pad_sequence(neural_features, batch_first=True, padding_value=0)
    neural_lengths = torch.tensor([item.shape[0] for item in neural_features])
    
    return {
        'neural_features': neural_features_padded,
        'neural_lengths': neural_lengths,
        'target_sentences': target_sentences,
        'sentence_texts': sentence_texts,
        'seq_lens': torch.tensor(seq_lens),
        'sessions': sessions,
        'block_nums': block_nums,
        'trial_nums': trial_nums,
    }

# ==================== SIMPLIFIED MODEL ARCHITECTURE ====================

class SimpleBrainToTextModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, vocab_size=500, num_layers=1, dropout=0.2):
        super(SimpleBrainToTextModel, self).__init__()
        
        # Encoder - processes neural data (simpler, no bidirectional)
        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False  # Simplified - no bidirectional
        )
        
        # Decoder - generates text
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,  # Same as encoder
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, neural_input, neural_lengths, target_sentences=None):
        batch_size = neural_input.size(0)
        
        # Encode neural data
        packed_input = pack_padded_sequence(neural_input, neural_lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoder_outputs, hidden = self.encoder_gru(packed_input)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)
        
        if target_sentences is not None:
            # Training mode - use teacher forcing
            target_embedded = self.decoder_embedding(target_sentences)
            decoder_output, _ = self.decoder_gru(target_embedded, hidden)
            output = self.output_layer(self.dropout(decoder_output))
            return output
        else:
            # Inference mode - generate sequence
            max_length = 30
            outputs = []
            
            # Start with SOS token
            decoder_input = torch.tensor([self.word_to_idx['<SOS>']] * batch_size, device=neural_input.device).unsqueeze(1)
            decoder_hidden = hidden
            
            for t in range(max_length):
                decoder_embedded = self.decoder_embedding(decoder_input)
                decoder_output, decoder_hidden = self.decoder_gru(decoder_embedded, decoder_hidden)
                output = self.output_layer(self.dropout(decoder_output))
                outputs.append(output)
                
                # Get next input (greedy)
                _, topi = output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
                
            return torch.cat(outputs, dim=1)

# ==================== TRAINING & PREDICTION ====================

def train_model():
    """Train the model to predict actual sentences"""
    print("Initializing Training...")
    print("=" * 60)
    
    # Find training data
    sessions = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not sessions:
        print("No sessions found!")
        return None, None
    
    # Load training data from first session
    session = sessions[0]
    train_file = os.path.join(DATA_DIR, session, "data_train.hdf5")
    
    if not os.path.exists(train_file):
        print(f"No training data found for {session}")
        return None, None
    
    print(f"Loading training data from {session}...")
    train_data = load_h5py_file(train_file)
    if not train_data:
        return None, None
    
    # Build vocabulary
    word_to_idx, idx_to_word, vocab = build_vocabulary(train_data)
    vocab_size = len(vocab)
    
    # Create dataset and dataloader
    dataset = BrainToTextDataset(train_data, word_to_idx)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleBrainToTextModel(
        input_dim=512,
        hidden_dim=HIDDEN_DIM,
        vocab_size=vocab_size,
        num_layers=1,  # Single layer for simplicity
        dropout=0.2
    )
    model.word_to_idx = word_to_idx
    model.idx_to_word = idx_to_word
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                neural_features = batch['neural_features'].to(device)
                neural_lengths = batch['neural_lengths'].to(device)
                target_sentences = batch['target_sentences'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(neural_features, neural_lengths, target_sentences)
                
                # Calculate loss (ignore padding)
                loss = criterion(
                    outputs[:, :-1].contiguous().view(-1, vocab_size),
                    target_sentences[:, 1:].contiguous().view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 5 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")
            
            # Show predictions after each epoch
            show_predictions(model, device, dataloader, idx_to_word)
    
    return model, idx_to_word

def show_predictions(model, device, dataloader, idx_to_word, num_samples=3):
    """Show model predictions vs ground truth"""
    model.eval()
    print("\nCurrent Predictions:")
    print("-" * 40)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:  # Just show first batch
                break
                
            neural_features = batch['neural_features'].to(device)
            neural_lengths = batch['neural_lengths'].to(device)
            sentence_texts = batch['sentence_texts']
            
            # Generate predictions
            outputs = model(neural_features, neural_lengths)
            _, predicted_indices = outputs.max(dim=-1)
            
            # Convert to text
            for i in range(min(num_samples, len(sentence_texts))):
                # Get predicted sentence
                pred_indices = predicted_indices[i].cpu().numpy()
                pred_words = []
                for idx in pred_indices:
                    word = idx_to_word[idx]
                    if word == '<EOS>':
                        break
                    if word not in ['<SOS>', '<PAD>']:
                        pred_words.append(word)
                predicted_sentence = ' '.join(pred_words) if pred_words else '<no prediction>'
                
                # Calculate confidence (simple softmax of first few words)
                if len(pred_words) > 0:
                    probs = torch.softmax(outputs[i, :len(pred_words)], dim=-1)
                    confidence = probs.max(dim=-1)[0].mean().item()
                else:
                    confidence = 0.0
                
                # Print results
                print(f"Predicted: '{predicted_sentence}' (confidence: {confidence:.3f})")
                print(f"Truth:     '{sentence_texts[i]}'")
                print()

def evaluate_on_validation(model, idx_to_word):
    """Evaluate trained model on validation data"""
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SET PREDICTIONS")
    print("=" * 60)
    
    # Find validation data
    sessions = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    session = sessions[0]
    val_file = os.path.join(DATA_DIR, session, "data_val.hdf5")
    
    if not os.path.exists(val_file):
        print("No validation data found")
        return
    
    # Load validation data
    val_data = load_h5py_file(val_file)
    if not val_data:
        return
    
    # Create validation dataset
    word_to_idx = model.word_to_idx
    val_dataset = BrainToTextDataset(val_data, word_to_idx)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    device = next(model.parameters()).device
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            neural_features = batch['neural_features'].to(device)
            neural_lengths = batch['neural_lengths'].to(device)
            sentence_texts = batch['sentence_texts']
            
            outputs = model(neural_features, neural_lengths)
            _, predicted_indices = outputs.max(dim=-1)
            
            for i in range(min(8, len(sentence_texts))):
                # Convert prediction to text
                pred_indices = predicted_indices[i].cpu().numpy()
                pred_words = []
                for idx in pred_indices:
                    word = idx_to_word[idx]
                    if word == '<EOS>':
                        break
                    if word not in ['<SOS>', '<PAD>']:
                        pred_words.append(word)
                predicted_sentence = ' '.join(pred_words) if pred_words else '<no prediction>'
                
                # Calculate confidence
                if len(pred_words) > 0:
                    probs = torch.softmax(outputs[i, :len(pred_words)], dim=-1)
                    confidence = probs.max(dim=-1)[0].mean().item()
                else:
                    confidence = 0.0
                
                print(f"Predicted: '{predicted_sentence}' (confidence: {confidence:.3f})")
                print(f"Truth:     '{sentence_texts[i]}'")
                print()
            
            break  # Only show first batch

# ==================== MAIN EXECUTION ====================

def main():
    """Main function to train model and show predictions"""
    print("Brain-to-Text Model Training")
    print("=" * 60)
    
    # Train the model
    model, idx_to_word = train_model()
    
    if model is None:
        print("Training failed!")
        return
    
    # Show final validation predictions
    evaluate_on_validation(model, idx_to_word)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("The model has been trained to predict actual ground truth sentences!")
    print("You can see the predictions above with confidence scores.")

if __name__ == "__main__":
    main()
