import torch
import torch.nn.functional as F
from Decoder import BertTranslationModel

def create_causal_mask(seq_len, batch_size=1):
    """
    ✅ 修正: Causal maskを正しい形状で作成
    """
    # (seq_len, seq_len) のマスクを作成
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, -1e9)
    
    # ✅ 修正: repeatを使用してバッチ次元を追加
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    mask = mask.repeat(batch_size, 1, 1, 1)  # expand → repeat
    
    print(f"  🔍 Causal mask形状: {mask.shape}")
    return mask

def create_padding_mask(ids):
    """
    ✅ 修正: パディングマスクを正しい形状で作成
    """
    batch_size, seq_len = ids.shape
    
    # パディング位置を特定 (0 = パディング)
    mask = (ids != 0).float()  # (batch, seq_len)
    
    # ✅ 修正: 正確な形状変換
    mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    mask = (1.0 - mask) * -1e9
    
    print(f"  🔍 Padding mask形状: {mask.shape}")
    print(f"  🔍 マスク範囲: [{mask.min():.1f}, {mask.max():.1f}]")
    return mask

def create_simple_masks(batch_size, seq_len):
    """
    ✅ 新規: 簡単なマスク作成（デバッグ用）
    """
    # 全て有効なマスク（マスクなし状態）
    mask = torch.zeros(batch_size, 1, 1, seq_len)
    return mask

def test_translation_model():
    """
    翻訳モデルのテスト（修正版）
    """
    print("🧪 翻訳モデルテスト開始")
    print("=" * 40)
    
    # ✅ 修正: より小さなモデルでテスト
    model = BertTranslationModel(
        ita_vocab_size=1000,   # 小さくして安全に
        eng_vocab_size=1200,   # 小さくして安全に
        max_seq_len=64,        # 小さくして安全に
        d_model=128,           # 小さくして高速化
        num_layers=2,          # 小さくして高速化
        num_heads=4            # 小さくして高速化
    )
    
    # サンプルデータ
    batch_size = 2
    eng_len, ita_len = 6, 4  # ✅ 修正: より小さなサイズ
    
    print(f"📋 設定:")
    print(f"  batch_size: {batch_size}")
    print(f"  english_len: {eng_len}, italian_len: {ita_len}")
    print(f"  model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 英語入力（Encoder用）
    eng_ids = torch.randint(1, 999, (batch_size, eng_len))  # ✅ 安全な範囲
    eng_ids[:, 0] = 101   # [CLS]
    eng_ids[:, -1] = 102  # [SEP]
    
    # イタリア語入力（Decoder用 - Teacher Forcing）
    ita_ids = torch.randint(1, 999, (batch_size, ita_len))  # ✅ 安全な範囲
    ita_ids[:, 0] = 101   # [CLS]
    ita_ids[:, -1] = 102  # [SEP]
    
    print(f"📥 入力データ:")
    print(f"  eng_ids: {eng_ids.shape}, range: [{eng_ids.min()}, {eng_ids.max()}]")
    print(f"  ita_ids: {ita_ids.shape}, range: [{ita_ids.min()}, {ita_ids.max()}]")
    print(f"  eng_ids[0]: {eng_ids[0].tolist()}")
    print(f"  ita_ids[0]: {ita_ids[0].tolist()}")
    
    # ✅ マスク作成（修正版）
    print(f"\n🎭 マスク作成:")
    try:
        eng_mask = create_padding_mask(eng_ids)        # 英語のパディングマスク
        ita_mask = create_causal_mask(ita_len, batch_size)  # イタリア語のCausalマスク
        
        print(f"  eng_mask: {eng_mask.shape}")
        print(f"  ita_mask: {ita_mask.shape}")
        
        # ✅ 形状が正しいかチェック
        assert eng_mask.dim() == 4, f"英語マスクの次元が間違い: {eng_mask.dim()} != 4"
        assert ita_mask.dim() == 4, f"イタリア語マスクの次元が間違い: {ita_mask.dim()} != 4"
        
    except Exception as e:
        print(f"⚠️  マスク作成でエラー: {e}")
        print("  → 簡単なマスクを使用します")
        eng_mask = create_simple_masks(batch_size, eng_len)
        ita_mask = create_simple_masks(batch_size, ita_len)
    
    # Forward pass
    print(f"\n🚀 Forward pass実行...")
    try:
        with torch.no_grad():
            # ✅ token_type_idsを削除してシンプルに
            logits = model(
                eng_ids=eng_ids,
                ita_ids=ita_ids,
                eng_mask=eng_mask,
                ita_mask=ita_mask,
                eng_token_type_ids=None,  # ✅ Noneで簡単に
                ita_token_type_ids=None,  # ✅ Noneで簡単に
            )
        
        print(f"📤 出力結果:")
        print(f"  logits.shape: {logits.shape}")
        print(f"  期待する形状: ({batch_size}, {ita_len}, 1000)")
        print(f"  logits統計: mean={logits.mean():.4f}, std={logits.std():.4f}")
        
        # 形状チェック
        expected_shape = (batch_size, ita_len, 1000)
        assert logits.shape == expected_shape, f"形状エラー: {logits.shape} != {expected_shape}"
        
        # NaN/Infチェック
        assert not torch.isnan(logits).any(), "NaNが検出されました"
        assert not torch.isinf(logits).any(), "Infが検出されました"
        
        print(f"✅ テスト成功! 翻訳モデルが正常に動作しています。")
        
        # ✅ 予測結果の確認
        predicted_tokens = torch.argmax(logits, dim=-1)
        print(f"\n🎯 予測結果:")
        print(f"  predicted_tokens[0]: {predicted_tokens[0].tolist()}")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

def test_step_by_step():
    """
    段階別テスト（修正版）
    """
    print(f"\n🔍 段階別テスト:")
    
    model = BertTranslationModel(
        ita_vocab_size=5000,
        eng_vocab_size=6000,
        max_seq_len=32,
        d_model=64,
        num_layers=1,
        num_heads=2
    )
    
    # ✅ 修正: 非常に小さなデータ
    eng_ids = torch.tensor([[101, 123, 456, 102]])  # (1, 4)
    ita_ids = torch.tensor([[101, 789, 102, 0]])    # (1, 4)
    
    print(f"  Encoder入力: {eng_ids.shape} = {eng_ids[0].tolist()}")
    print(f"  Decoder入力: {ita_ids.shape} = {ita_ids[0].tolist()}")
    
    # ✅ マスク作成（簡単版）
    eng_mask = create_simple_masks(1, 4)
    ita_mask = create_simple_masks(1, 4)
    
    print(f"  eng_mask: {eng_mask.shape}")
    print(f"  ita_mask: {ita_mask.shape}")
    
    # 段階別実行
    with torch.no_grad():
        try:
            # Encoderのみテスト
            print(f"\n  📝 Encoderテスト:")
            encoder_output = model.encoder(eng_ids, eng_mask)
            print(f"    Encoder出力: {encoder_output.shape}")
            print(f"    統計: mean={encoder_output.mean():.4f}")
            
            # Decoder embeddingのみテスト  
            print(f"\n  📝 Decoder Embeddingテスト:")
            decoder_emb = model.decoder_embeddings(ita_ids)
            print(f"    Decoder embedding: {decoder_emb.shape}")
            print(f"    統計: mean={decoder_emb.mean():.4f}")
            
            # 全体テスト
            print(f"\n  📝 全体テスト:")
            logits = model(eng_ids, ita_ids, eng_mask, ita_mask)
            print(f"    最終出力: {logits.shape}")
            print(f"    統計: mean={logits.mean():.4f}")
            
        except Exception as e:
            print(f"    ❌ 段階別テストエラー: {e}")
            import traceback
            traceback.print_exc()

def test_mask_shapes():
    """
    ✅ 新規: マスクの形状を詳細テスト
    """
    print(f"\n🔬 マスク形状テスト:")
    
    batch_size, seq_len = 2, 4
    
    # テストデータ
    ids = torch.tensor([
        [101, 123, 456, 102],
        [101, 789, 102, 0]    # 最後がパディング
    ])
    
    print(f"  入力IDs: {ids.shape}")
    print(f"  ids[0]: {ids[0].tolist()}")
    print(f"  ids[1]: {ids[1].tolist()}")
    
    # パディングマスク
    padding_mask = create_padding_mask(ids)
    print(f"  padding_mask: {padding_mask.shape}")
    print(f"  padding_mask[0,0,0]: {padding_mask[0,0,0].tolist()}")
    print(f"  padding_mask[1,0,0]: {padding_mask[1,0,0].tolist()}")
    
    # Causalマスク
    causal_mask = create_causal_mask(seq_len, batch_size)
    print(f"  causal_mask: {causal_mask.shape}")
    print(f"  causal_mask[0,0]: \n{causal_mask[0,0]}")

def main():
    test_mask_shapes()      # ✅ まずマスクをテスト
    test_step_by_step()     # ✅ 段階別テスト
    test_translation_model() # ✅ 全体テスト

if __name__ == "__main__":
    main()