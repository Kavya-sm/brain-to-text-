# eeg_to_text_complete_fixed.py
"""
Complete EEG to Text model with all features except PCA - FIXED VERSION
"""
import os
import math
import json
import wave
import pywt
import numpy as np
from collections import Counter, defaultdict, deque
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.stats import permutation_test

# ---------------------------
# Config
# ---------------------------
@dataclass
class ModelConfig:
    # Feature extraction
    target_feature_dim: int = 512
    sampling_rate: int = 256
    use_wavelet: bool = True
    wavelet_level: int = 4
    wavelet_family: str = 'db4'
    
    # Architecture
    d_model: int = 256
    enc_layers: int = 2
    nhead: int = 4
    d_ff: int = 512
    use_bilstm: bool = True
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 20
    teacher_forcing_ratio: float = 0.8
    dropout_rate: float = 0.1
    
    # Decoding
    beam_width: int = 5
    n_best: int = 3
    max_len: int = 50
    temperature: float = 0.8
    
    # Vocabulary
    vocab_size: int = 200
    phoneme_vocab_size: int = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = ModelConfig()

# ---------------------------
# Feature Extraction (PSD + Wavelet)
# ---------------------------
class SpectralFeatureExtractor:
    def __init__(self, sampling_rate=256, target_dim=512, use_wavelet=True):
        self.sampling_rate = sampling_rate
        self.target_dim = target_dim
        self.use_wavelet = use_wavelet
        
    def compute_psd(self, eeg_data):
        """Compute Power Spectral Density features"""
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
            
        psd_features = []
        for channel in eeg_data:
            # Compute PSD using Welch's method
            freqs, psd = signal.welch(channel, fs=self.sampling_rate, nperseg=min(256, len(channel)))
            
            # Extract bands: delta (0.5-4Hz), theta (4-8Hz), alpha (8-13Hz), beta (13-30Hz), gamma (30-50Hz)
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
            
            band_features = []
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                    band_features.append(band_power)
                else:
                    band_features.append(0.0)
            
            # Add spectral centroid and spread
            if np.sum(psd) > 0:
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
            else:
                spectral_centroid = 0.0
                spectral_spread = 0.0
                
            band_features.extend([spectral_centroid, spectral_spread])
            psd_features.extend(band_features)
            
        return np.array(psd_features)
    
    def compute_wavelet_features(self, eeg_data, wavelet='db4', level=4):
        """Compute Wavelet Transform features"""
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
            
        wavelet_features = []
        for channel in eeg_data:
            try:
                # Multi-level wavelet decomposition
                coeffs = pywt.wavedec(channel, wavelet, level=min(level, pywt.dwt_max_level(len(channel), pywt.Wavelet(wavelet))))
                
                channel_features = []
                for coeff in coeffs:
                    # Statistical features from coefficients
                    if len(coeff) > 0:
                        channel_features.extend([
                            np.mean(coeff),
                            np.std(coeff),
                            np.median(coeff),
                            np.max(np.abs(coeff)),
                            np.sum(coeff ** 2)  # Energy
                        ])
                    else:
                        channel_features.extend([0.0] * 5)
                        
                wavelet_features.extend(channel_features)
            except:
                # Fallback: zero features if wavelet fails
                wavelet_features.extend([0.0] * (5 * (level + 1)))
                
        return np.array(wavelet_features)
    
    def extract_features(self, eeg_data):
        """Extract combined PSD + Wavelet features"""
        features = []
        
        # PSD features
        psd_feats = self.compute_psd(eeg_data)
        features.extend(psd_feats)
        
        # Wavelet features (if enabled)
        if self.use_wavelet:
            wavelet_feats = self.compute_wavelet_features(eeg_data)
            features.extend(wavelet_feats)
        
        # Ensure fixed dimension
        if len(features) > self.target_dim:
            features = features[:self.target_dim]
        elif len(features) < self.target_dim:
            features = np.pad(features, (0, self.target_dim - len(features)))
            
        return np.array(features, dtype=np.float32)

# ---------------------------
# Language Model for Rescoring
# ---------------------------
class NGramLanguageModel:
    def __init__(self, n=2):  # Changed to n=2 for better laptop performance
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        
    def train(self, texts):
        """Train n-gram model on text corpus"""
        for text in texts:
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            text = str(text).lower().strip()
            
            words = text.split()
            if len(words) == 0:
                continue
                
            # Add sentence boundaries
            words = ['<s>'] + words + ['</s>']
            self.vocab.update(words)
            
            # Build n-grams
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i + self.n - 1])
                next_word = words[i + self.n - 1]
                self.ngrams[context][next_word] += 1
                
    def score_sequence(self, words):
        """Score a sequence using n-gram probabilities"""
        if len(words) == 0:
            return 0.0
            
        words = ['<s>'] + words + ['</s>']
        total_log_prob = 0.0
        word_count = 0
        
        for i in range(len(words) - self.n + 1):
            context = tuple(words[i:i + self.n - 1])
            target = words[i + self.n - 1]
            
            if context in self.ngrams and target in self.ngrams[context]:
                total_count = sum(self.ngrams[context].values())
                prob = self.ngrams[context][target] / total_count
                total_log_prob += math.log(prob + 1e-10)  # Avoid log(0)
                word_count += 1
                
        return total_log_prob / max(1, word_count)
    
    def rescore(self, hypotheses):
        """Rescore hypotheses using language model"""
        scored_hyps = []
        for hyp_text in hypotheses:
            words = hyp_text.lower().split()
            lm_score = self.score_sequence(words)
            scored_hyps.append((hyp_text, lm_score))
            
        # Sort by LM score (higher is better)
        scored_hyps.sort(key=lambda x: x[1], reverse=True)
        return [hyp[0] for hyp in scored_hyps]

# ---------------------------
# Dataset with Enhanced Features
# ---------------------------
class BrainToTextDataset(Dataset):
    def __init__(self, files: List[str], config: ModelConfig, max_length=2000, 
                 is_training=True, build_vocab=True, vocab=None):
        self.files = files
        self.config = config
        self.max_length = max_length
        self.is_training = is_training
        self.feature_extractor = SpectralFeatureExtractor(
            sampling_rate=config.sampling_rate,
            target_dim=config.target_feature_dim,
            use_wavelet=config.use_wavelet
        )

        self.data = self._load_all()
        
        if build_vocab:
            if is_training:
                self.vocab = self._build_vocab()
                self.phoneme_vocab = self._build_phoneme_vocab()
            else:
                self.vocab = vocab if vocab is not None else {'<pad>':0,'<sos>':1,'<eos>':2,'<unk>':3}
                self.phoneme_vocab = {'<pad>':0, '<unk>':1}
        else:
            self.vocab = vocab if vocab is not None else {'<pad>':0,'<sos>':1,'<eos>':2,'<unk>':3}
            self.phoneme_vocab = {'<pad>':0, '<unk>':1}
            
        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        self.inv_phoneme_vocab = {v:k for k,v in self.phoneme_vocab.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")

    def _extract_enhanced_features(self, raw_features):
        """Extract PSD + Wavelet features from raw EEG"""
        # If already processed, return as is
        if raw_features.shape[0] == self.config.target_feature_dim:
            return raw_features.T
            
        # Extract spectral features
        enhanced_features = []
        for time_point in range(raw_features.shape[1]):
            time_slice = raw_features[:, time_point]
            features = self.feature_extractor.extract_features(time_slice)
            enhanced_features.append(features)
            
        return np.array(enhanced_features, dtype=np.float32)

    def _load_h5(self, path):
        data = {'neural': [], 'sentence_label': [], 'phonemes': []}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                g = f[k]
                nf = g['input_features'][:]
                nf = np.asarray(nf, dtype=np.float32)
                
                # Extract enhanced features
                enhanced_features = self._extract_enhanced_features(nf)
                data['neural'].append(enhanced_features)
                
                s = g.attrs.get('sentence_label', "")
                data['sentence_label'].append(s)
                
                seq_class_ids = g.get('seq_class_ids', None)
                if seq_class_ids is not None:
                    try:
                        data['phonemes'].append(np.asarray(seq_class_ids[:], dtype=np.int64))
                    except:
                        data['phonemes'].append(None)
                else:
                    data['phonemes'].append(None)
        return data

    def _load_all(self):
        out = {'neural': [], 'sentence_label': [], 'phonemes': []}
        for p in self.files:
            print("Loading", p)
            try:
                dat = self._load_h5(p)
            except Exception as e:
                print("Error loading", p, e)
                continue
            out['neural'].extend(dat['neural'])
            out['sentence_label'].extend(dat['sentence_label'])
            out['phonemes'].extend(dat['phonemes'])
        print(f"Total samples: {len(out['neural'])}")
        return out

    def _build_vocab(self):
        counter = Counter()
        for s in self.data['sentence_label']:
            if s is None or s == "":
                continue
            if isinstance(s, bytes):
                s = s.decode('utf-8', errors='ignore')
            s = str(s).lower().strip()
            if s:
                counter.update(list(s))
        
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        
        # Add common characters
        common_chars = [' ', '.', ',', '!', '?', "'", '"', '-', ';', ':']
        for ch in common_chars:
            if ch not in vocab:
                vocab[ch] = len(vocab)
        
        # Add letters by frequency
        for ch, count in counter.most_common():
            if ch not in vocab:
                vocab[ch] = len(vocab)
            if len(vocab) >= self.config.vocab_size:
                break
        
        print(f"Built vocab size: {len(vocab)}")
        return vocab

    def _build_phoneme_vocab(self):
        counter = Counter()
        for phon_seq in self.data['phonemes']:
            if phon_seq is not None:
                counter.update(phon_seq)
        
        phoneme_vocab = {'<pad>': 0, '<unk>': 1}
        
        for i, (phon_id, count) in enumerate(counter.most_common(self.config.phoneme_vocab_size - 2)):
            phoneme_vocab[int(phon_id)] = i + 2
        
        print(f"Built phoneme vocab size: {len(phoneme_vocab)}")
        return phoneme_vocab

    def text_to_tokens(self, text):
        if text is None or text == "":
            return torch.tensor([self.vocab['<sos>'], self.vocab['<eos>']], dtype=torch.long)
        
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        text = str(text).lower().strip()
        
        if not text:
            return torch.tensor([self.vocab['<sos>'], self.vocab['<eos>']], dtype=torch.long)
            
        toks = [self.vocab.get(ch, self.vocab['<unk>']) for ch in text]
        return torch.tensor([self.vocab['<sos>']] + toks + [self.vocab['<eos>']], dtype=torch.long)

    def phonemes_to_tokens(self, phonemes):
        if phonemes is None:
            return None
        phon_tokens = [self.phoneme_vocab.get(int(ph), self.phoneme_vocab['<unk>']) for ph in phonemes]
        return torch.tensor(phon_tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data['neural'])

    def __getitem__(self, idx):
        feats = self.data['neural'][idx]
        if feats.shape[0] > self.max_length:
            feats = feats[:self.max_length]
        feats = feats.astype(np.float32)
        
        tokens = self.text_to_tokens(self.data['sentence_label'][idx])
        
        phon = None
        if self.data['phonemes'][idx] is not None:
            phon = self.phonemes_to_tokens(self.data['phonemes'][idx])
        
        return {
            'features': torch.from_numpy(feats), 
            'tokens': tokens, 
            'phonemes': phon, 
            'raw': self.data['sentence_label'][idx]
        }

# ---------------------------
# Model Architecture
# ---------------------------
class ConvSubsampling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x)
        return x.transpose(1,2)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        attn_out, attn_w = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_w

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, num_layers=2, nhead=4, d_ff=512, use_bilstm=True, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.subsample = ConvSubsampling(d_model, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.bilstm = nn.LSTM(d_model, d_model//2, num_layers=1, batch_first=True, bidirectional=True)
            
    def forward(self, x, lengths=None):
        x = self.input_proj(x)
        x = self.subsample(x)
        
        key_padding_mask = None
        if lengths is not None:
            T = x.size(1)
            reduced_len = (lengths + 3) // 4
            key_padding_mask = torch.arange(T, device=x.device).unsqueeze(0) >= reduced_len.unsqueeze(1)
        
        attn_maps = []
        for block in self.blocks:
            x, attn_w = block(x, key_padding_mask=key_padding_mask)
            attn_maps.append(attn_w)
            
        if self.use_bilstm:
            x, _ = self.bilstm(x)
            
        return x, attn_maps

class AttentionDecoder(nn.Module):
    def __init__(self, enc_dim, embed_dim, hidden_dim, vocab_size, phoneme_dim=0, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTMCell(embed_dim + enc_dim, hidden_dim)
        self.enc_proj = nn.Linear(enc_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)
        self.phoneme_head = nn.Linear(hidden_dim, phoneme_dim) if phoneme_dim > 0 else None
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim

    def forward(self, encoder_outputs, encoder_mask=None, targets=None, teacher_forcing_ratio=0.5, max_len=100):
        B, T_enc, _ = encoder_outputs.size()
        device = encoder_outputs.device
        
        encoder_proj = self.enc_proj(encoder_outputs)
        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros(B, self.hidden_dim, device=device)
        
        outputs = []
        phon_outputs = []
        
        if targets is not None and targets.size(1) > 1:
            seq_len = targets.size(1)
            input_tok = targets[:, 0]
            
            for t in range(seq_len - 1):
                input_tok = input_tok.clamp(0, self.embedding.num_embeddings - 1)
                emb = self.embedding(input_tok)

                q = self.query_proj(h).unsqueeze(1)
                score = torch.tanh(encoder_proj + q)
                score = self.v(score).squeeze(-1)
                
                if encoder_mask is not None:
                    score = score.masked_fill(encoder_mask, float('-inf'))
                    
                attn_weights = F.softmax(score, dim=1)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
                
                lstm_input = torch.cat([emb, context], dim=1)
                h, c = self.lstm(lstm_input, (h, c))
                
                logits = self.out_proj(self.dropout(h))
                outputs.append(logits.unsqueeze(1))
                
                if self.phoneme_head is not None:
                    phon_logits = self.phoneme_head(self.dropout(h))
                    phon_outputs.append(phon_logits.unsqueeze(1))
                
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                if teacher_force and t + 1 < seq_len:
                    input_tok = targets[:, t+1]
                else:
                    if torch.rand(1).item() < 0.1:
                        probs = F.softmax(logits, dim=-1)
                        input_tok = torch.multinomial(probs, 1).squeeze(-1)
                    else:
                        input_tok = logits.argmax(dim=-1)
                    
            outputs = torch.cat(outputs, dim=1) if outputs else torch.zeros(B, 0, self.out_proj.out_features, device=device)
            phon_outputs = torch.cat(phon_outputs, dim=1) if phon_outputs else None
            
        else:
            input_tok = torch.full((B,), 1, dtype=torch.long, device=device)
            for step in range(max_len):
                input_tok = input_tok.clamp(0, self.embedding.num_embeddings - 1)
                emb = self.embedding(input_tok)
                
                q = self.query_proj(h).unsqueeze(1)
                score = torch.tanh(encoder_proj + q)
                score = self.v(score).squeeze(-1)
                
                if encoder_mask is not None:
                    score = score.masked_fill(encoder_mask, float('-inf'))
                    
                attn_weights = F.softmax(score, dim=1)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
                
                lstm_input = torch.cat([emb, context], dim=1)
                h, c = self.lstm(lstm_input, (h, c))
                
                logits = self.out_proj(self.dropout(h))
                
                temperature = 0.8
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                input_tok = torch.multinomial(probs, 1).squeeze(-1)
                
                outputs.append(logits.unsqueeze(1))
                
                if self.phoneme_head is not None:
                    phon_logits = self.phoneme_head(self.dropout(h))
                    phon_outputs.append(phon_logits.unsqueeze(1))
                
                if (input_tok == 2).all():
                    break
                    
            outputs = torch.cat(outputs, dim=1) if outputs else torch.zeros(B, 0, self.out_proj.out_features, device=device)
            phon_outputs = torch.cat(phon_outputs, dim=1) if phon_outputs else None
            
        return outputs, phon_outputs

class EEGToTextModel(nn.Module):
    def __init__(self, input_dim=512, d_model=256, enc_layers=2, nhead=4, d_ff=512, use_bilstm=True,
                 embed_dim=128, dec_hidden=256, vocab_size=5000, phoneme_dim=0, dropout=0.1):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim=input_dim, d_model=d_model, num_layers=enc_layers, 
                                      nhead=nhead, d_ff=d_ff, use_bilstm=use_bilstm, dropout=dropout)
        self.decoder = AttentionDecoder(enc_dim=d_model, embed_dim=embed_dim, hidden_dim=dec_hidden, 
                                      vocab_size=vocab_size, phoneme_dim=phoneme_dim, dropout=dropout)
        self.vocab_size = vocab_size

    def forward(self, feats, lengths=None, targets=None, teacher_forcing_ratio=0.5, max_len=100):
        enc_out, attn_maps = self.encoder(feats, lengths)
        
        enc_mask = None
        if lengths is not None:
            T_enc = enc_out.size(1)
            reduced = (lengths + 3) // 4
            enc_mask = torch.arange(T_enc, device=feats.device).unsqueeze(0) >= reduced.unsqueeze(1)
            
        logits, phon_logits = self.decoder(enc_out, encoder_mask=enc_mask, targets=targets, 
                                         teacher_forcing_ratio=teacher_forcing_ratio, max_len=max_len)
        return logits, phon_logits, attn_maps

# ---------------------------
# Advanced Decoding: Beam Search + N-best
# ---------------------------
class BeamSearchNode:
    def __init__(self, hidden_state, cell_state, sequence, log_prob, length, attn_weights):
        self.hidden = hidden_state
        self.cell = cell_state
        self.sequence = sequence  # List of token IDs
        self.log_prob = log_prob
        self.length = length
        self.attn_weights = attn_weights
        
    def eval(self, alpha=1.0):
        # Length normalization
        return self.log_prob / (self.length ** alpha)

def beam_search_decode(model, features, lengths, vocab, beam_width=5, max_len=50, device=DEVICE):
    """Beam search decoding for better sequence generation"""
    model.eval()
    with torch.no_grad():
        # Encode features
        enc_out, _ = model.encoder(features.unsqueeze(0), lengths)
        T_enc = enc_out.size(1)
        
        # Initialize beam
        start_node = BeamSearchNode(
            hidden_state=torch.zeros(1, model.decoder.hidden_dim, device=device),
            cell_state=torch.zeros(1, model.decoder.hidden_dim, device=device),
            sequence=[vocab['<sos>']],
            log_prob=0.0,
            length=1,
            attn_weights=[]
        )
        
        beams = [start_node]
        finished = []
        
        for step in range(max_len):
            new_beams = []
            
            for node in beams:
                if node.sequence[-1] == vocab['<eos>']:
                    finished.append(node)
                    continue
                    
                # Prepare decoder inputs
                input_tok = torch.tensor([node.sequence[-1]], device=device)
                emb = model.decoder.embedding(input_tok)
                
                # Compute attention
                encoder_proj = model.decoder.enc_proj(enc_out)
                q = model.decoder.query_proj(node.hidden).unsqueeze(1)
                score = torch.tanh(encoder_proj + q)
                score = model.decoder.v(score).squeeze(-1)
                attn_weights = F.softmax(score, dim=1)
                context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)
                
                # LSTM step
                lstm_input = torch.cat([emb, context], dim=1)
                h, c = model.decoder.lstm(lstm_input, (node.hidden, node.cell))
                
                # Get log probabilities
                logits = model.decoder.out_proj(model.decoder.dropout(h))
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top k candidates
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    token = topk_indices[0, i].item()
                    log_prob = node.log_prob + topk_log_probs[0, i].item()
                    
                    new_node = BeamSearchNode(
                        hidden_state=h,
                        cell_state=c,
                        sequence=node.sequence + [token],
                        log_prob=log_prob,
                        length=node.length + 1,
                        attn_weights=node.attn_weights + [attn_weights.cpu()]
                    )
                    new_beams.append(new_node)
            
            # Keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x.eval())[-beam_width:]
            
            # Early stopping if all beams finished
            if all(node.sequence[-1] == vocab['<eos>'] for node in beams):
                finished.extend(beams)
                break
        
        # Add any remaining beams to finished
        finished.extend(beams)
        
        # Return n-best sequences
        finished.sort(key=lambda x: x.eval(), reverse=True)
        return finished

def generate_n_best(model, features, lengths, vocab, n_best=3, beam_width=5, max_len=50, device=DEVICE):
    """Generate N-best hypotheses using beam search"""
    beams = beam_search_decode(model, features, lengths, vocab, beam_width, max_len, device)
    
    n_best_sequences = []
    for i, beam in enumerate(beams[:n_best]):
        # Convert tokens to text
        tokens = beam.sequence[1:]  # Remove <sos>
        text = tokens_to_text(tokens, {v: k for k, v in vocab.items()})
        n_best_sequences.append({
            'text': text,
            'score': beam.eval(),
            'tokens': tokens,
            'attention': beam.attn_weights
        })
    
    return n_best_sequences

# ---------------------------
# Explainability Tools
# ---------------------------
class ModelExplainer:
    def __init__(self, model, vocab, device=DEVICE):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.inv_vocab = {v: k for k, v in vocab.items()}
    
    def aggregate_attention(self, dataloader, num_samples=100):
        """Aggregate attention maps across samples"""
        self.model.eval()
        all_attention_maps = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i * batch['features'].size(0) >= num_samples:
                    break
                    
                feats = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                
                # Get attention maps from encoder
                _, attn_maps = self.model.encoder(feats, lengths)
                
                # Average across layers and heads
                for sample_attn in attn_maps:
                    # attn_maps shape: [layers, (attn_weights)]
                    layer_attentions = []
                    for layer_attn in sample_attn:
                        # Average across attention heads
                        if layer_attn.dim() == 4:  # Multi-head attention
                            head_avg = layer_attn.mean(dim=1)  # Average over heads
                            layer_attentions.append(head_avg.cpu().numpy())
                    
                    if layer_attentions:
                        all_attention_maps.append(np.array(layer_attentions))
        
        if all_attention_maps:
            # Average across samples and layers
            avg_attention = np.mean([np.mean(sample, axis=0) for sample in all_attention_maps], axis=0)
            return avg_attention
        return None
    
    def compute_permutation_importance(self, dataset, num_permutations=50):  # Reduced for speed
        """Compute feature importance using permutation"""
        original_predictions = []
        permuted_predictions = []
        
        # Get baseline predictions
        for i in range(min(20, len(dataset))):  # Reduced samples
            sample = dataset[i]
            feats = sample['features'].unsqueeze(0).to(self.device)
            length = torch.tensor([feats.shape[1]], device=self.device)
            
            tokens = greedy_decode_with_temperature(self.model, feats.squeeze(0), length.item(), 
                                                  self.vocab, device=self.device)
            original_predictions.append(tokens)
        
        # Permutation test for each feature dimension
        feature_importance = np.zeros(CONFIG.target_feature_dim)
        
        for feature_idx in range(min(50, CONFIG.target_feature_dim)):  # Test only first 50 features
            permuted_scores = []
            
            for _ in range(num_permutations):
                # Create permuted dataset
                permuted_correct = 0
                total = 0
                
                for i in range(min(10, len(dataset))):  # Reduced samples
                    sample = dataset[i]
                    feats = sample['features'].clone()
                    
                    # Permute the specific feature dimension
                    permuted_feats = feats.clone()
                    permuted_feats[:, feature_idx] = feats[torch.randperm(feats.shape[0]), feature_idx]
                    
                    # Get prediction with permuted features
                    permuted_feats = permuted_feats.unsqueeze(0).to(self.device)
                    length = torch.tensor([permuted_feats.shape[1]], device=self.device)
                    
                    tokens = greedy_decode_with_temperature(self.model, permuted_feats.squeeze(0), 
                                                          length.item(), self.vocab, device=self.device)
                    
                    # Simple accuracy measure (compare if any token matches)
                    orig_tokens = original_predictions[i]
                    if len(tokens) > 0 and len(orig_tokens) > 0:
                        if tokens[0] == orig_tokens[0]:  # Simple first token match
                            permuted_correct += 1
                    total += 1
                
                if total > 0:
                    permuted_scores.append(permuted_correct / total)
            
            if permuted_scores:
                # Importance = baseline_accuracy - permuted_accuracy
                baseline_acc = 0.5  # Approximate baseline
                feature_importance[feature_idx] = baseline_acc - np.mean(permuted_scores)
        
        return feature_importance
    
    def visualize_attention(self, attention_weights, input_features, output_text):
        """Create attention visualization data"""
        if attention_weights is None or len(attention_weights) == 0:
            return None
            
        # Average across layers if multiple layers
        if isinstance(attention_weights, list):
            avg_attention = np.mean([attn.cpu().numpy() for attn in attention_weights], axis=0)
        else:
            avg_attention = attention_weights.cpu().numpy()
        
        # Create time-aligned attention
        time_steps = min(input_features.shape[0], avg_attention.shape[0])
        attention_data = {
            'time_steps': list(range(time_steps)),
            'attention_weights': avg_attention[:time_steps].tolist(),
            'output_tokens': output_text.split(),
            'feature_magnitude': np.mean(np.abs(input_features.cpu().numpy()), axis=1)[:time_steps].tolist()
        }
        
        return attention_data

# ---------------------------
# Complete Training Pipeline
# ---------------------------
def collate_fn(batch):
    max_seq = max(x['features'].shape[0] for x in batch)
    feat_dim = batch[0]['features'].shape[1]
    B = len(batch)
    
    feats = torch.zeros(B, max_seq, feat_dim, dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.long)
    
    max_tlen = max(x['tokens'].shape[0] for x in batch)
    tokens = torch.full((B, max_tlen), fill_value=0, dtype=torch.long)
    
    phon_present = any(x['phonemes'] is not None for x in batch)
    max_ph_len = max((x['phonemes'].shape[0] if x['phonemes'] is not None else 0) for x in batch) if phon_present else 0
    phonemes = torch.full((B, max_ph_len), fill_value=-1, dtype=torch.long) if phon_present else None
    
    raw_texts = []
    
    for i, x in enumerate(batch):
        L = x['features'].shape[0]
        feats[i, :L] = x['features']
        lengths[i] = L
        
        tlen = x['tokens'].shape[0]
        tokens[i, :tlen] = x['tokens']
        
        if phon_present and x['phonemes'] is not None:
            plen = x['phonemes'].shape[0]
            phonemes[i, :plen] = x['phonemes']
        
        raw_texts.append(x.get('raw', ''))
    
    out = {'features': feats, 'lengths': lengths, 'tokens': tokens, 'raw_texts': raw_texts}
    if phonemes is not None:
        out['phonemes'] = phonemes
    return out

class CompleteTrainer:
    def __init__(self, model, optimizer, vocab, config, device=DEVICE):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.vocab = vocab
        self.config = config
        self.device = device
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        self.crit = nn.CrossEntropyLoss(ignore_index=0)
        self.phoneme_crit = nn.CrossEntropyLoss(ignore_index=-1)
        self.best_val_loss = float('inf')
        
        # Initialize explainer
        self.explainer = ModelExplainer(model, vocab, device)
        
        # Initialize language model
        self.language_model = NGramLanguageModel(n=2)  # Use n=2 for better laptop performance

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        cnt = 0
        
        for batch_idx, batch in enumerate(loader):
            feats = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            tokens = tokens.clamp(0, self.model.vocab_size - 1)
            
            phonemes = batch.get('phonemes', None)
            if phonemes is not None:
                phonemes = phonemes.to(self.device)
            
            self.optimizer.zero_grad()
            logits, phon_logits, _ = self.model(feats, lengths, targets=tokens, 
                                              teacher_forcing_ratio=self.config.teacher_forcing_ratio)
            
            if logits.size(1) == 0:
                continue
                
            V = logits.size(-1)
            target_len = min(logits.size(1), tokens.size(1)-1)
            target_tokens = tokens[:, 1:1+target_len].reshape(-1)
            logits = logits[:, :target_len, :].contiguous().view(-1, V)
            
            loss = self.crit(logits, target_tokens)
            
            if phon_logits is not None and phonemes is not None:
                P = phon_logits.size(-1)
                ph_target_len = min(phon_logits.size(1), phonemes.size(1))
                ph_target = phonemes[:, :ph_target_len].reshape(-1)
                phon_logits = phon_logits[:, :ph_target_len, :].contiguous().view(-1, P)
                loss_ph = self.phoneme_crit(phon_logits, ph_target)
                loss = loss + 0.3 * loss_ph
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            cnt += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        return total_loss / max(1, cnt)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        cnt = 0
        
        with torch.no_grad():
            for batch in loader:
                feats = batch['features'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                tokens = tokens.clamp(0, self.model.vocab_size - 1)
                
                phonemes = batch.get('phonemes', None)
                if phonemes is not None:
                    phonemes = phonemes.to(self.device)
                    
                logits, phon_logits, _ = self.model(feats, lengths, targets=tokens, teacher_forcing_ratio=1.0)
                
                if logits.size(1) == 0:
                    continue
                    
                V = logits.size(-1)
                target_len = min(logits.size(1), tokens.size(1)-1)
                target_tokens = tokens[:, 1:1+target_len].reshape(-1)
                logits = logits[:, :target_len, :].contiguous().view(-1, V)
                
                loss = self.crit(logits, target_tokens)
                
                if phon_logits is not None and phonemes is not None:
                    P = phon_logits.size(-1)
                    ph_target_len = min(phon_logits.size(1), phonemes.size(1))
                    ph_target = phonemes[:, :ph_target_len].reshape(-1)
                    phon_logits = phon_logits[:, :ph_target_len, :].contiguous().view(-1, P)
                    loss_ph = self.phoneme_crit(phon_logits, ph_target)
                    loss = loss + 0.3 * loss_ph
                    
                total_loss += loss.item()
                cnt += 1
                
        return total_loss / max(1, cnt)

    def train_language_model(self, train_dataset):
        """Train n-gram language model on training texts"""
        # FIXED: train_dataset.data['sentence_label'] is already a list of strings
        texts = [text for text in train_dataset.data['sentence_label'] if text is not None and text != ""]
        self.language_model.train(texts)
        print("✓ Trained n-gram language model")

    def generate_with_rescoring(self, features, lengths, n_best=3):
        """Generate text with LM rescoring"""
        # Generate N-best hypotheses
        n_best_hyps = generate_n_best(self.model, features, lengths, self.vocab, 
                                    n_best=n_best, beam_width=self.config.beam_width,
                                    max_len=self.config.max_len, device=self.device)
        
        # Extract texts for rescoring
        hypotheses = [hyp['text'] for hyp in n_best_hyps]
        
        # Rescore with language model
        rescored_hypotheses = self.language_model.rescore(hypotheses)
        
        return rescored_hypotheses

    def analyze_model(self, dataset, num_samples=50):
        """Complete model analysis with explainability"""
        print("🔍 Analyzing model...")
        
        # Aggregate attention
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        avg_attention = self.explainer.aggregate_attention(dataloader, num_samples)
        
        # Compute feature importance
        feature_importance = self.explainer.compute_permutation_importance(dataset)
        
        analysis_results = {
            'average_attention': avg_attention.tolist() if avg_attention is not None else None,
            'feature_importance': feature_importance.tolist(),
            'top_features': np.argsort(feature_importance)[-10:].tolist()  # Top 10 important features
        }
        
        print("✓ Model analysis completed")
        return analysis_results

# ---------------------------
# Utility Functions
# ---------------------------
def greedy_decode_with_temperature(model, feats, length, vocab, max_len=50, temperature=0.8, device=DEVICE):
    model.eval()
    with torch.no_grad():
        feats = feats.unsqueeze(0).to(device)
        lengths = torch.tensor([length], device=device)
        
        enc_out, _ = model.encoder(feats, lengths)
        T_enc = enc_out.size(1)
        reduced = (lengths + 3) // 4
        enc_mask = torch.arange(T_enc, device=device).unsqueeze(0) >= reduced.unsqueeze(1)
        
        logits, _ = model.decoder(enc_out, encoder_mask=enc_mask, targets=None, max_len=max_len)
        
        if temperature > 0 and logits.size(1) > 0:
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            tokens = torch.multinomial(probs.squeeze(0), 1).squeeze(-1).cpu().numpy().tolist()
        else:
            tokens = logits.argmax(dim=-1).squeeze(0).cpu().numpy().tolist() if logits.size(1) > 0 else []
            
        return tokens

def tokens_to_text(tokens, inv_vocab):
    chars = []
    for t in tokens:
        if t == 2:  # <eos>
            break
        if t not in (0, 1, 2):
            chars.append(inv_vocab.get(t, ''))
    text = ''.join(chars)
    text = ' '.join(text.split())
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text

def save_model_safe(model, vocab, phoneme_vocab, epoch, val_loss, filename):
    try:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'phoneme_vocab': phoneme_vocab,
            'epoch': epoch,
            'val_loss': val_loss,
            'config': CONFIG
        }, filename)
        print(f"✓ Model saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

# ---------------------------
# Main Function
# ---------------------------
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
    for s in sessions:
        sp = os.path.join(base_path, s)
        if os.path.exists(sp):
            tr = os.path.join(sp, 'data_train.hdf5')
            va = os.path.join(sp, 'data_val.hdf5')
            te = os.path.join(sp, 'data_test.hdf5')
            if os.path.exists(tr): train_files.append(tr)
            if os.path.exists(va): val_files.append(va)
            if os.path.exists(te): test_files.append(te)
    return train_files, val_files, test_files

def main():
    BASE = "/Users/kavya/Downloads/hdf5_data_final"
    train_files, val_files, test_files = get_data_files(BASE)
    
    if not train_files:
        print("No train files found")
        return
    
    print("🚀 Loading datasets with enhanced feature extraction...")
    train_ds = BrainToTextDataset(train_files, CONFIG, max_length=2000, is_training=True)
    val_ds = BrainToTextDataset(val_files, CONFIG, max_length=2000, is_training=False, vocab=train_ds.vocab)
    
    print(f"✅ Training samples: {len(train_ds)}")
    print(f"✅ Validation samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CONFIG.batch_size, shuffle=False, collate_fn=collate_fn)

    print("🤖 Initializing complete model...")
    model = EEGToTextModel(
        input_dim=CONFIG.target_feature_dim, 
        d_model=CONFIG.d_model, 
        enc_layers=CONFIG.enc_layers,
        nhead=CONFIG.nhead, 
        d_ff=CONFIG.d_ff, 
        use_bilstm=CONFIG.use_bilstm,
        embed_dim=128, 
        dec_hidden=256, 
        vocab_size=len(train_ds.vocab),
        phoneme_dim=len(train_ds.phoneme_vocab),
        dropout=CONFIG.dropout_rate
    )
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate, weight_decay=1e-5)
    trainer = CompleteTrainer(model, optimizer, train_ds.vocab, CONFIG, DEVICE)

    # Train language model
    trainer.train_language_model(train_ds)

    print(f"🎯 Starting training for {CONFIG.num_epochs} epochs...")
    for epoch in range(1, CONFIG.num_epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader)
        
        print(f"📊 Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            save_success = save_model_safe(
                model, train_ds.vocab, train_ds.phoneme_vocab, 
                epoch, val_loss, 'complete_eeg2text_model.pth'
            )
        
        # Demo advanced features every 5 epochs
        if epoch % 5 == 0 and len(val_ds) > 0:
            print("\n--- 🧪 DEMO: Advanced Features ---")
            sample = val_ds[0]
            feats = sample['features']
            length = feats.shape[0]
            
            print(f"📝 Input: {sample['raw']}")
            
            # 1. Basic temperature sampling
            tokens = greedy_decode_with_temperature(model, feats, length, train_ds.vocab, 
                                                  temperature=0.8, device=DEVICE)
            basic_text = tokens_to_text(tokens, train_ds.inv_vocab)
            print(f"🌡️  Basic (temp=0.8): {basic_text}")
            
            # 2. N-best with beam search
            print("🔍 N-best with beam search:")
            n_best_results = generate_n_best(model, feats, torch.tensor([length]), train_ds.vocab, 
                                           n_best=3, beam_width=5, device=DEVICE)
            for i, result in enumerate(n_best_results):
                print(f"   {i+1}. {result['text']} (score: {result['score']:.3f})")
            
            # 3. LM rescoring
            print("📚 LM-rescored:")
            rescored = trainer.generate_with_rescoring(feats, torch.tensor([length]), n_best=3)
            for i, text in enumerate(rescored):
                print(f"   {i+1}. {text}")
            
            print("--- 🏁 End Demo ---\n")
    
    # Final model analysis
    print("🔬 Running final model analysis...")
    analysis = trainer.analyze_model(val_ds)
    print(f"📈 Top important features: {analysis['top_features']}")
    
    print("\n🎉 Training completed with all advanced features!")

if __name__ == "__main__":
    main()
