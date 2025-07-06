import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from collections import defaultdict

from TranslationDataset import TranslationDataset, download_sentence, build_vocabulary
from Decoder import BertTranslationModel

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs=5):
    """Train the BertTranslationModel"""
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_iterator = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in train_iterator:
            # Move batch to device
            eng_ids = batch['eng_ids'].to(device)
            eng_mask = batch['eng_mask'].to(device)
            eng_token_type_ids = batch['eng_token_type_ids'].to(device)
            ita_ids = batch['ita_ids'].to(device)
            ita_mask = batch['ita_mask'].to(device)
            ita_target_ids = batch['ita_target_ids'].to(device)
            ita_token_type_ids = batch['ita_token_type_ids'].to(device)
            ita_causal_mask = batch['ita_causal_mask'].to(device)
            
            # Prepare masks for the model forward pass
            batch_size, ita_seq_len = ita_ids.shape
            _, eng_seq_len = eng_ids.shape
            num_heads = model.decoder_blocks[0].cross_attention.num_heads
            
            # Eng mask for decoder cross-attention
            eng_cross_mask = eng_mask.unsqueeze(1).unsqueeze(2)
            eng_cross_mask = eng_cross_mask.expand(batch_size, num_heads, ita_seq_len, eng_seq_len)
            eng_cross_mask = (1.0 - eng_cross_mask.float()) * -1e9
            
            # Ita causal mask for decoder self-attention
            ita_causal_mask = ita_causal_mask.unsqueeze(1)
            ita_causal_mask = ita_causal_mask.expand(batch_size, num_heads, ita_seq_len, ita_seq_len)
            
            # Forward pass
            logits = model(
                eng_ids=eng_ids,
                ita_ids=ita_ids,
                eng_mask=eng_cross_mask,
                ita_mask=ita_causal_mask,
                eng_token_type_ids=eng_token_type_ids,
                ita_token_type_ids=ita_token_type_ids
            )
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), ita_target_ids.view(-1))
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_iterator.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_iterator = tqdm.tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")
        
        with torch.no_grad():
            for batch in val_iterator:
                # Move batch to device
                eng_ids = batch['eng_ids'].to(device)
                eng_mask = batch['eng_mask'].to(device)
                eng_token_type_ids = batch['eng_token_type_ids'].to(device)
                ita_ids = batch['ita_ids'].to(device)
                ita_mask = batch['ita_mask'].to(device)
                ita_target_ids = batch['ita_target_ids'].to(device)
                ita_token_type_ids = batch['ita_token_type_ids'].to(device)
                ita_causal_mask = batch['ita_causal_mask'].to(device)
                
                # Prepare masks (same as training)
                batch_size, ita_seq_len = ita_ids.shape
                _, eng_seq_len = eng_ids.shape
                num_heads = model.decoder_blocks[0].cross_attention.num_heads
                
                eng_cross_mask = eng_mask.unsqueeze(1).unsqueeze(2)
                eng_cross_mask = eng_cross_mask.expand(batch_size, num_heads, ita_seq_len, eng_seq_len)
                eng_cross_mask = (1.0 - eng_cross_mask.float()) * -1e9
                
                ita_causal_mask = ita_causal_mask.unsqueeze(1)
                ita_causal_mask = ita_causal_mask.expand(batch_size, num_heads, ita_seq_len, ita_seq_len)
                
                # Forward pass
                logits = model(
                    eng_ids=eng_ids,
                    ita_ids=ita_ids,
                    eng_mask=eng_cross_mask,
                    ita_mask=ita_causal_mask,
                    eng_token_type_ids=eng_token_type_ids,
                    ita_token_type_ids=ita_token_type_ids
                )
                
                # Calculate validation loss
                val_loss = criterion(logits.view(-1, logits.size(-1)), ita_target_ids.view(-1))
                total_val_loss += val_loss.item()
                val_iterator.set_postfix({'val_loss': val_loss.item()})
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")
        print("-" * 50)

def main():
    # Configuration
    MAXLEN = 10
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading dataset...")
    eng_sentences, ita_sentences = download_sentence()
    data_pair = list(zip(eng_sentences, ita_sentences))
    
    # Build vocabularies
    print("Building vocabularies...")
    eng_vocab = build_vocabulary(eng_sentences, min_freq=2)
    ita_vocab = build_vocabulary(ita_sentences, min_freq=2)
    
    print(f"English vocabulary size: {len(eng_vocab)}")
    print(f"Italian vocabulary size: {len(ita_vocab)}")
    
    # Split data
    train_size = int(0.8 * len(data_pair))
    train_data = data_pair[:train_size]
    val_data = data_pair[train_size:]
    
    print(f"Total pairs: {len(data_pair)}")
    print(f"Training pairs: {len(train_data)}")
    print(f"Validation pairs: {len(val_data)}")
    
    # Create datasets and dataloaders
    train_dataset = TranslationDataset(train_data, eng_vocab, ita_vocab, seq_len=MAXLEN)
    val_dataset = TranslationDataset(val_data, eng_vocab, ita_vocab, seq_len=MAXLEN)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    
    # Initialize model
    print("Initializing model...")
    model = BertTranslationModel(
        ita_vocab_size=len(ita_vocab),
        eng_vocab_size=len(eng_vocab),
        max_seq_len=MAXLEN,
        d_model=768,
        num_layers=6,
        num_heads=12,
        dropout=0.1
    )
    
    model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=ita_vocab['[PAD]'])
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Loss function:", criterion)
    print("Optimizer:", optimizer)
    
    # Train the model
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, EPOCHS)
    
    # Save the model
    model_save_path = "bert_translation_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Test with a sample
    print("\n=== Testing with sample translations ===")
    model.eval()
    test_samples = [
        ("Hello world", "Ciao mondo"),
        ("Good morning", "Buongiorno"),
        ("Thank you", "Grazie")
    ]
    
    with torch.no_grad():
        for eng_text, expected_ita in test_samples:
            print(f"English: {eng_text}")
            print(f"Expected Italian: {expected_ita}")
            
            # Simple tokenization for testing
            eng_words = eng_text.lower().split()
            eng_ids = [eng_vocab.get(word.strip('.,!?;:"()[]{}'), eng_vocab['[UNK]']) for word in eng_words]
            eng_ids = [eng_vocab['[CLS]']] + eng_ids + [eng_vocab['[SEP]']]
            
            if len(eng_ids) < MAXLEN:
                eng_ids += [eng_vocab['[PAD]']] * (MAXLEN - len(eng_ids))
            else:
                eng_ids = eng_ids[:MAXLEN]
            
            eng_tensor = torch.tensor([eng_ids], device=device)
            eng_mask = torch.tensor([[1 if id != eng_vocab['[PAD]'] else 0 for id in eng_ids]], device=device)
            
            # Generate prediction (simplified)
            ita_input = torch.tensor([[ita_vocab['[CLS]']] + [ita_vocab['[PAD]']] * (MAXLEN-1)], device=device)
            
            batch_size, ita_seq_len = ita_input.shape
            _, eng_seq_len = eng_tensor.shape
            num_heads = model.decoder_blocks[0].cross_attention.num_heads
            
            eng_cross_mask = eng_mask.unsqueeze(1).unsqueeze(2)
            eng_cross_mask = eng_cross_mask.expand(batch_size, num_heads, ita_seq_len, eng_seq_len)
            eng_cross_mask = (1.0 - eng_cross_mask.float()) * -1e9
            
            causal_mask = torch.triu(torch.ones(MAXLEN, MAXLEN), diagonal=1).bool()
            causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
            causal_mask = causal_mask.masked_fill(~causal_mask, 0.0)
            ita_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, ita_seq_len, ita_seq_len).to(device)
            
            logits = model(
                eng_ids=eng_tensor,
                ita_ids=ita_input,
                eng_mask=eng_cross_mask,
                ita_mask=ita_causal_mask,
                eng_token_type_ids=torch.zeros_like(eng_tensor),
                ita_token_type_ids=torch.zeros_like(ita_input)
            )
            
            predictions = torch.argmax(logits[0], dim=-1)
            
            # Decode prediction
            ita_id_to_word = {v: k for k, v in ita_vocab.items()}
            predicted_words = [ita_id_to_word.get(token_id.item(), '[UNK]') for token_id in predictions[:5]]
            predicted_text = ' '.join([w for w in predicted_words if w not in ['[PAD]', '[CLS]', '[SEP]']])
            
            print(f"Predicted Italian: {predicted_text}")
            print()

if __name__ == "__main__":
    main()