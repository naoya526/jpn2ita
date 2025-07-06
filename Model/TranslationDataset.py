import torch
from torch.utils.data import Dataset
def download_scentence():
    eng_sentences = []
    with open("text-eng.txt", "r") as f:
        for line in f:
            line = line.strip().replace('<sos>', '').replace('<eos>', '').strip()
            if line:
                eng_sentences.append(line)

    ita_sentences = []
    with open("text-ita.txt", "r") as f:
        for line in f:
            line = line.strip().replace('<sos>', '').replace('<eos>', '').strip()
            if line:
                ita_sentences.append(line)
    return eng_sentences, ita_sentences


class TranslationDataset(Dataset):
    def __init__(self, data_pair, eng_vocab, ita_vocab, seq_len=64):
        self.data_pair = data_pair
        self.eng_vocab = eng_vocab  # {word: id} dictionary
        self.ita_vocab = ita_vocab  # {word: id} dictionary
        self.seq_len = seq_len
        
        # Create reverse mappings
        self.eng_id_to_word = {v: k for k, v in eng_vocab.items()}
        self.ita_id_to_word = {v: k for k, v in ita_vocab.items()}
        
        # Special token IDs
        self.eng_pad_id = eng_vocab.get('[PAD]', 0)
        self.eng_cls_id = eng_vocab.get('[CLS]', 1)
        self.eng_sep_id = eng_vocab.get('[SEP]', 2)
        self.eng_unk_id = eng_vocab.get('[UNK]', 3)
        
        self.ita_pad_id = ita_vocab.get('[PAD]', 0)
        self.ita_cls_id = ita_vocab.get('[CLS]', 1)
        self.ita_sep_id = ita_vocab.get('[SEP]', 2)
        self.ita_unk_id = ita_vocab.get('[UNK]', 3)

    def __len__(self):
        return len(self.data_pair)

    def tokenize_sentence(self, sentence, vocab, unk_id):
        """Convert sentence to list of token IDs"""
        words = sentence.lower().split()  # Simple word tokenization
        token_ids = []
        for word in words:
            # Remove punctuation and get token ID
            word = word.strip('.,!?;:"()[]{}')
            token_id = vocab.get(word, unk_id)
            token_ids.append(token_id)
        return token_ids

    def __getitem__(self, item):
        eng_sentence, ita_sentence = self.data_pair[item]

        # Tokenize English sentence to word IDs
        eng_token_ids = self.tokenize_sentence(eng_sentence, self.eng_vocab, self.eng_unk_id)
        
        # Add CLS and SEP tokens, then pad/truncate
        eng_token_ids = [self.eng_cls_id] + eng_token_ids + [self.eng_sep_id]
        if len(eng_token_ids) > self.seq_len:
            eng_token_ids = eng_token_ids[:self.seq_len]
        else:
            eng_token_ids += [self.eng_pad_id] * (self.seq_len - len(eng_token_ids))
        
        # Create attention mask for English
        eng_attention_mask = [1 if token_id != self.eng_pad_id else 0 for token_id in eng_token_ids]

        # Tokenize Italian sentence to word IDs
        ita_token_ids = self.tokenize_sentence(ita_sentence, self.ita_vocab, self.ita_unk_id)
        
        # Add CLS and SEP tokens, then pad/truncate
        ita_full_ids = [self.ita_cls_id] + ita_token_ids + [self.ita_sep_id]
        if len(ita_full_ids) > self.seq_len:
            ita_full_ids = ita_full_ids[:self.seq_len]
        else:
            ita_full_ids += [self.ita_pad_id] * (self.seq_len - len(ita_full_ids))

        # Create decoder input (shift right: [CLS] + original[:-1])
        decoder_input_ids = [self.ita_cls_id] + ita_full_ids[:-1]
        
        # Target is the original sequence (what we want to predict)
        ita_target_ids = ita_full_ids

        # Create attention mask for Italian
        ita_attention_mask = [1 if token_id != self.ita_pad_id else 0 for token_id in ita_full_ids]

        # Create causal mask for decoder self-attention
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
        causal_mask = causal_mask.masked_fill(~causal_mask, 0.0)

        output = {
            "eng_ids": torch.tensor(eng_token_ids, dtype=torch.long),
            "eng_mask": torch.tensor(eng_attention_mask, dtype=torch.long),
            "eng_token_type_ids": torch.zeros(self.seq_len, dtype=torch.long),
            "ita_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "ita_mask": torch.tensor(ita_attention_mask, dtype=torch.long),
            "ita_token_type_ids": torch.zeros(self.seq_len, dtype=torch.long),
            "ita_causal_mask": causal_mask,
            "ita_target_ids": torch.tensor(ita_target_ids, dtype=torch.long),
        }

        return output