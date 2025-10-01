

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

import re
import unicodedata
import math
import time
import os

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH = "data/fra.txt"
MODEL_SAVE_PATH = "models/transformer_en_fr.pth"
NUM_EXAMPLES = 237838
EPOCHS = 5

# Model Hyperparameters
EMBEDDING_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
FF_DIM = 512
DROPOUT_RATE = 0.1

# Training Parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Translation Parameters
MAX_LENGTH = 50

# Special Tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# -----------------------------------------------------------------------------
# 2. DATA PREPROCESSING
# -----------------------------------------------------------------------------
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    w = f"{SOS_TOKEN} {w} {EOS_TOKEN}"
    return w

def create_dataset(path, num_examples):
    try:
        lines = open(path, encoding='UTF-8').read().strip().split('\n')
        num_lines = min(num_examples, len(lines))
        word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]]
                      for l in lines[:num_lines]]
        return zip(*word_pairs)
    except FileNotFoundError:
        print(f"Error: The data file was not found at {path}")
        print("Please make sure you have a 'data' folder with 'fra.txt' inside.")
        exit()

class LanguageIndex:
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.create_index()

    def create_index(self):
        vocab = set()
        for phrase in self.lang:
            vocab.update(phrase.split(' '))
        self.word2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        current_idx = 4
        for word in sorted(list(vocab)):
            if word not in self.word2idx:
                self.word2idx[word] = current_idx
                current_idx += 1
        self.idx2word = {index: word for word, index in self.word2idx.items()}

# -----------------------------------------------------------------------------
# 3. TRANSFORMER MODEL DEFINITION
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, embed_dim, num_heads,
                 num_encoder_layers, num_decoder_layers, ff_dim, dropout_rate):
        super(TransformerModel, self).__init__()
        self.embedding_input = nn.Embedding(input_vocab_size, embed_dim, padding_idx=0)
        self.embedding_target = nn.Embedding(target_vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim, dropout=dropout_rate
        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_embed = self.dropout(self.positional_encoding(self.embedding_input(src)))
        tgt_embed = self.dropout(self.positional_encoding(self.embedding_target(tgt)))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(src.device)
        output = self.transformer(
            src_embed, tgt_embed, tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc_out(output)

# -----------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def train_model():
    print("--- Starting Model Training ---")
    if not os.path.exists("models"):
        os.makedirs("models")

    input_lang_docs, target_lang_docs = create_dataset(DATA_PATH, NUM_EXAMPLES)
    input_lang = LanguageIndex(input_lang_docs)
    target_lang = LanguageIndex(target_lang_docs)

    input_tensor = [[input_lang.word2idx.get(s, input_lang.word2idx[UNK_TOKEN]) for s in sen.split(' ')] for sen in input_lang_docs]
    target_tensor = [[target_lang.word2idx.get(s, target_lang.word2idx[UNK_TOKEN]) for s in sen.split(' ')] for sen in target_lang_docs]

    input_tensor_padded = pad_sequence([torch.tensor(it) for it in input_tensor], batch_first=True, padding_value=input_lang.word2idx[PAD_TOKEN])
    target_tensor_padded = pad_sequence([torch.tensor(tt) for tt in target_tensor], batch_first=True, padding_value=target_lang.word2idx[PAD_TOKEN])

    dataset = TensorDataset(input_tensor_padded, target_tensor_padded)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_vocab_size = len(input_lang.word2idx)
    target_vocab_size = len(target_lang.word2idx)

    model = TransformerModel(
        input_vocab_size, target_vocab_size, EMBEDDING_DIM, NUM_HEADS,
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FF_DIM, DROPOUT_RATE
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=target_lang.word2idx[PAD_TOKEN])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        for (input_batch, target_batch) in dataloader:
            input_batch = input_batch.transpose(0, 1).to(device)
            target_batch = target_batch.transpose(0, 1).to(device)
            optimizer.zero_grad()
            target_input = target_batch[:-1, :]
            target_output = target_batch[1:, :]
            src_padding_mask = (input_batch == input_lang.word2idx[PAD_TOKEN]).transpose(0, 1)
            tgt_padding_mask = (target_input == target_lang.word2idx[PAD_TOKEN]).transpose(0, 1)
            output = model(input_batch, target_input, src_padding_mask, tgt_padding_mask, src_padding_mask)
            loss = criterion(output.reshape(-1, output.shape[-1]), target_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(dataloader)
        end_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s")

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_lang_word2idx': input_lang.word2idx,
        'target_lang_word2idx': target_lang.word2idx,
        'target_lang_idx2word': target_lang.idx2word,
    }, MODEL_SAVE_PATH)
    print(f"--- Training Complete. Model saved to {MODEL_SAVE_PATH} ---")

def translate_sentence(model, sentence, input_lang_word2idx, target_lang_word2idx, target_lang_idx2word, device):
    model.eval()
    processed_sentence = preprocess_sentence(sentence)
    input_tokens = [input_lang_word2idx.get(s, input_lang_word2idx[UNK_TOKEN]) for s in processed_sentence.split(' ')]
    input_tensor = torch.LongTensor(input_tokens).unsqueeze(1).to(device)
    
    start_token_idx = target_lang_word2idx[SOS_TOKEN]
    end_token_idx = target_lang_word2idx[EOS_TOKEN]
    
    output_tokens = [start_token_idx]

    for _ in range(MAX_LENGTH):
        target_tensor = torch.LongTensor(output_tokens).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(input_tensor, target_tensor, None, None, None)
        predicted_index = output.argmax(2)[-1, :].item()
        output_tokens.append(predicted_index)
        if predicted_index == end_token_idx:
            break

    output_words = [target_lang_idx2word.get(idx, UNK_TOKEN) for idx in output_tokens]
    return " ".join(output_words[1:-1])

def start_translation_cli():
    print("--- Starting Translation Mode ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        print("Please train the model first by running the script.")
        return

    input_lang_word2idx = checkpoint['input_lang_word2idx']
    target_lang_word2idx = checkpoint['target_lang_word2idx']
    target_lang_idx2word = checkpoint['target_lang_idx2word']
    
    input_vocab_size = len(input_lang_word2idx)
    target_vocab_size = len(target_lang_word2idx)

    model = TransformerModel(
        input_vocab_size, target_vocab_size, EMBEDDING_DIM, NUM_HEADS,
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FF_DIM, DROPOUT_RATE
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print("\nModel loaded. Enter an English sentence to translate (or 'quit' to exit).")
    
    while True:
        try:
            input_sentence = input("> ")
            if input_sentence.lower() == 'quit':
                break
            # THIS IS THE CORRECTED LINE. It now uses the variables loaded from the checkpoint.
            translation = translate_sentence(model, input_sentence, input_lang_word2idx, target_lang_word2idx, target_lang_idx2word, device)
            print(f"Translation: {translation}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Trained model not found.")
        train_model()
        print("\nRun the script again to start translating sentences.")
    else:
        start_translation_cli()
