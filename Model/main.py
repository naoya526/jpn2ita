import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from Encoder import EncoderBlock, BertEmbeddings, Bert
from Decoder import CrossAttention, DecoderBlock


# main

def exp_Transformer():
    d_model = 512
    seq = 10
    batch_size = 2
    num_heads=2
    d_ff=10

    x = torch.randn(batch_size, seq, d_model)  # (batch, seq, d_model)
    print("start")
    func = EncoderBlock(d_model, num_heads, d_ff)
    #func = ShowMultiHeadAttention(d_model, num_heads)
    out = func(x)
    assert out.shape == x.shape
    return 0

def test_bert_embeddings():
    """
    BertEmbeddingsの軽量テスト
    """
    print("🧪 BertEmbeddings テスト開始")
    print("=" * 50)
    
    # パラメータ設定
    vocab_size = 1000
    d_model = 128  # 軽量化（通常は768）
    max_seq_len = 64
    batch_size = 2
    seq_len = 8
    
    # モデル初期化
    embeddings = BertEmbeddings(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    
    print(f"📋 設定:")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - d_model: {d_model}")
    print(f"  - max_seq_len: {max_seq_len}")
    print(f"  - batch_size: {batch_size}, seq_len: {seq_len}")
    
    # テストデータ作成
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    token_type_ids[:, seq_len//2:] = 1  # 後半を文B（セグメント1）に
    
    print(f"\n📥 入力データ:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  token_type_ids: {token_type_ids.shape}")
    print(f"  input_ids[0]: {input_ids[0].tolist()}")
    print(f"  token_type_ids[0]: {token_type_ids[0].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        output = embeddings(input_ids, token_type_ids)
    
    print(f"\n📤 出力:")
    print(f"  output shape: {output.shape}")
    print(f"  output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  output mean: {output.mean():.3f}")
    print(f"  output std: {output.std():.3f}")
    
    # 各埋め込みの個別テスト
    print(f"\n🔬 個別埋め込みテスト:")
    
    # Token Embeddings
    token_emb = embeddings.token(input_ids)
    print(f"  Token embeddings: {token_emb.shape}")
    
    # Position Embeddings
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    pos_emb = embeddings.position(position_ids)
    print(f"  Position embeddings: {pos_emb.shape}")
    
    # Segment Embeddings
    seg_emb = embeddings.segment(token_type_ids)
    print(f"  Segment embeddings: {seg_emb.shape}")
    
    # 手動計算との比較
    manual_sum = token_emb + pos_emb + seg_emb
    manual_output = embeddings.layer_norm(manual_sum)
    
    print(f"\n✅ 検証:")
    print(f"  手動計算との差: {torch.abs(output - manual_output).max():.6f}")
    
    # セグメント埋め込みの効果確認
    print(f"\n🎯 セグメント埋め込みの効果:")
    seg_0_emb = embeddings.segment.weight[0]  # セグメント0の埋め込み
    seg_1_emb = embeddings.segment.weight[1]  # セグメント1の埋め込み
    similarity = F.cosine_similarity(seg_0_emb.unsqueeze(0), seg_1_emb.unsqueeze(0))
    print(f"  セグメント0 vs 1 類似度: {similarity.item():.3f}")
    print(f"  (学習前なのでランダム値)")
    
    print(f"\n🎉 BertEmbeddings テスト完了！")
    
    return output

def test_edge_cases():
    """
    エッジケースのテスト
    """
    print("\n🚨 エッジケーステスト")
    print("=" * 40)
    
    vocab_size = 100
    d_model = 64
    embeddings = BertEmbeddings(vocab_size, d_model, max_seq_len=32, dropout=0.0)
    
    # テスト1: 最小サイズ
    print("1️⃣ 最小サイズテスト (1x1)")
    input_ids = torch.tensor([[5]])
    output = embeddings(input_ids)
    print(f"   出力形状: {output.shape}")
    assert output.shape == (1, 1, d_model), "形状が正しくありません"
    
    # テスト2: パディングトークン
    print("2️⃣ パディングトークンテスト")
    input_ids = torch.tensor([[0, 1, 2, 0, 0]])  # 0はパディング
    output = embeddings(input_ids)
    print(f"   パディング位置の出力: {output[0, 0].sum():.3f}")
    
    # テスト3: token_type_ids未指定
    print("3️⃣ token_type_ids未指定テスト")
    input_ids = torch.tensor([[1, 2, 3, 4]])
    output1 = embeddings(input_ids)  # token_type_ids=None
    output2 = embeddings(input_ids, torch.zeros_like(input_ids))
    diff = torch.abs(output1 - output2).max()
    print(f"   差分: {diff:.6f}")
    assert diff < 1e-6, "token_type_ids=Noneの処理が正しくありません"
    
    print("✅ 全てのエッジケーステスト通過！")
def test_bert_masking():
    """
    BERTのマスキング動作をテスト
    """
    print("🧪 BERT Masking テスト")
    print("=" * 40)
    
    # 小さなBERTモデル
    bert = Bert(vocab_size=1000, d_model=128, num_layers=2, num_heads=4)
    
    # テストデータ: 異なる長さの文
    input_ids = torch.tensor([
        [101, 123, 578, 102,    0,    0],  # 短い文
        [101, 111, 222, 333, 444, 102]   # 長い文
    ])
    
    # 手動でattention_maskを作成
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],  # 最初の4語のみ
        [1, 1, 1, 1, 1, 1]   # 全語有効
    ])
    
    print(f"📥 入力:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = bert(input_ids, attention_mask)
    
    print(f"📤 出力:")
    print(f"  output: {output.shape}")
    
    # パディング位置の出力確認
    padding_output = output[0, 4:, :]  # 最初のサンプルのパディング部分
    valid_output = output[0, :4, :]    # 有効部分
    
    print(f"📊 パディング位置の統計:")
    print(f"  パディング部分の平均: {padding_output.mean():.6f}")
    print(f"  有効部分の平均: {valid_output.mean():.6f}")
    print(f"  (パディング部分もAttentionの影響を受ける)")

def test_cross_attention():
    """
    CrossAttentionの動作をテスト
    Query: English hidden states
    Key, Value: Italian hidden states
    """
    print("🧪 CrossAttention テスト開始")
    print("=" * 50)
    
    # パラメータ設定
    batch_size = 2
    eng_len = 6  # 英語の系列長
    ita_len = 8  # イタリア語の系列長
    d_model = 128
    num_heads = 8
    
    print(f"📋 設定:")
    print(f"  batch_size: {batch_size}")
    print(f"  english_len: {eng_len}, italian_len: {ita_len}")
    print(f"  d_model: {d_model}, num_heads: {num_heads}")
    
    # CrossAttentionモデル初期化
    cross_attention = CrossAttention(d_model, num_heads, dropout=0.1)
    
    # ダミーデータ作成
    # 英語の隠れ状態（Query用）
    english_hidden = torch.randn(batch_size, eng_len, d_model)
    # イタリア語の隠れ状態（Key, Value用）
    italian_hidden = torch.randn(batch_size, ita_len, d_model)
    
    print(f"\n📥 入力データ:")
    print(f"  english_hidden: {english_hidden.shape}")
    print(f"  italian_hidden: {italian_hidden.shape}")
    print(f"  english_mean: {english_hidden.mean():.4f}")
    print(f"  italian_mean: {italian_hidden.mean():.4f}")
    
    # マスク作成（イタリア語の一部をマスク）
    italian_mask = torch.ones(batch_size, ita_len)
    italian_mask[0, -2:] = 0  # 最初のサンプルの最後2語をマスク
    italian_mask[1, -3:] = 0  # 2番目のサンプルの最後3語をマスク
    
    # マスク形状変換: (batch, ita_len) → (batch, 1, 1, ita_len)
    extended_mask = italian_mask.unsqueeze(1).unsqueeze(2)
    extended_mask = (1.0 - extended_mask) * -1e9
    
    print(f"\n🎭 マスク情報:")
    print(f"  italian_mask[0]: {italian_mask[0].tolist()}")
    print(f"  italian_mask[1]: {italian_mask[1].tolist()}")
    print(f"  extended_mask.shape: {extended_mask.shape}")
    
    # Forward pass
    print(f"\n🚀 CrossAttention実行...")
    with torch.no_grad():
        output = cross_attention(english_hidden, italian_hidden, extended_mask)
    
    print(f"\n📤 出力結果:")
    print(f"  output.shape: {output.shape}")
    print(f"  output.mean: {output.mean():.4f}")
    print(f"  output.std: {output.std():.4f}")
    print(f"  output.min: {output.min():.4f}")
    print(f"  output.max: {output.max():.4f}")
    
    # 形状チェック
    expected_shape = (batch_size, eng_len, d_model)
    assert output.shape == expected_shape, f"形状エラー: {output.shape} != {expected_shape}"
    
    # NaN/Inf チェック
    assert not torch.isnan(output).any(), "NaNが検出されました"
    assert not torch.isinf(output).any(), "Infが検出されました"
    
    print(f"\n✅ 基本チェック通過!")
    
    # 詳細検証: マスクの効果確認
    print(f"\n🔍 詳細検証:")
    
    # マスクなしでの実行
    output_no_mask = cross_attention(english_hidden, italian_hidden, mask=None)
    
    # 差分確認
    diff = torch.abs(output - output_no_mask).mean()
    print(f"  マスクあり vs なし の差分: {diff:.6f}")
    
    if diff > 1e-6:
        print(f"  ✅ マスクが正しく適用されています")
    else:
        print(f"  ⚠️  マスクの効果が小さいです")
    
    # Attention重みの確認（内部実装をテスト）
    print(f"\n🎯 内部動作確認:")
    
    # Query, Key, Value の生成をテスト
    query = cross_attention.query(english_hidden)
    key = cross_attention.key(italian_hidden)
    value = cross_attention.value(italian_hidden)
    
    print(f"  query.shape: {query.shape}")
    print(f"  key.shape: {key.shape}")
    print(f"  value.shape: {value.shape}")
    
    # 各ヘッドでの処理をテスト
    query_heads = query.view(batch_size, eng_len, num_heads, d_model // num_heads)
    key_heads = key.view(batch_size, ita_len, num_heads, d_model // num_heads)
    
    print(f"  query_heads.shape: {query_heads.shape}")
    print(f"  key_heads.shape: {key_heads.shape}")
    
    # Attention scoresの計算テスト
    query_heads = query_heads.permute(0, 2, 1, 3)  # (batch, heads, eng_len, head_dim)
    key_heads = key_heads.permute(0, 2, 1, 3)      # (batch, heads, ita_len, head_dim)
    
    scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
    print(f"  attention_scores.shape: {scores.shape}")
    print(f"  expected: ({batch_size}, {num_heads}, {eng_len}, {ita_len})")
    
    # Softmax後の重みの確認
    weights = F.softmax(scores, dim=-1)
    print(f"  attention_weights sum: {weights.sum(dim=-1).mean():.4f} (should be ~1.0)")
    
    print(f"\n🎉 CrossAttention テスト完了!")
    return output

def test_cross_attention_edge_cases():
    """
    CrossAttentionのエッジケーステスト
    """
    print("\n🚨 CrossAttention エッジケーステスト")
    print("=" * 40)
    
    d_model = 64
    num_heads = 4
    cross_attention = CrossAttention(d_model, num_heads, dropout=0.0)
    
    # テスト1: 最小サイズ
    print("1️⃣ 最小サイズテスト (1x1)")
    english = torch.randn(1, 1, d_model)
    italian = torch.randn(1, 1, d_model)
    output = cross_attention(english, italian)
    assert output.shape == (1, 1, d_model), "最小サイズテスト失敗"
    print("   ✅ 通過")
    
    # テスト2: 異なる系列長
    print("2️⃣ 異なる系列長テスト")
    english = torch.randn(2, 3, d_model)  # 英語: 3語
    italian = torch.randn(2, 7, d_model)  # イタリア語: 7語
    output = cross_attention(english, italian)
    assert output.shape == (2, 3, d_model), "異なる系列長テスト失敗"
    print("   ✅ 通過")
    
    # テスト3: 完全マスク
    print("3️⃣ 完全マスクテスト")
    english = torch.randn(1, 4, d_model)
    italian = torch.randn(1, 4, d_model)
    full_mask = torch.full((1, 1, 1, 4), -1e9)  # 全てマスク
    output = cross_attention(english, italian, full_mask)
    assert not torch.isnan(output).any(), "完全マスクでNaN発生"
    print("   ✅ 通過")
    
    # テスト4: 勾配フロー確認
    print("4️⃣ 勾配フローテスト")
    english = torch.randn(1, 2, d_model, requires_grad=True)
    italian = torch.randn(1, 3, d_model, requires_grad=True)
    output = cross_attention(english, italian)
    loss = output.sum()
    loss.backward()
    
    assert english.grad is not None, "英語の勾配が計算されていません"
    assert italian.grad is not None, "イタリア語の勾配が計算されていません"
    print("   ✅ 通過")
    
    print("✅ 全エッジケーステスト通過!")

def test_decoder_block():
    """
    DecoderBlockの動作をテスト
    Self-Attention + Cross-Attention + Feed Forward の統合テスト
    """
    print("🧪 DecoderBlock テスト開始")
    print("=" * 50)
    
    # パラメータ設定
    batch_size = 2
    eng_len = 6    # 英語（Decoder入力）の系列長
    ita_len = 8    # イタリア語（Encoder出力）の系列長
    d_model = 128
    num_heads = 8
    d_ff = 512
    
    print(f"📋 設定:")
    print(f"  batch_size: {batch_size}")
    print(f"  english_len: {eng_len}, italian_len: {ita_len}")
    print(f"  d_model: {d_model}, num_heads: {num_heads}, d_ff: {d_ff}")
    
    # DecoderBlock初期化
    decoder_block = DecoderBlock(d_model, num_heads, d_ff, dropout=0.1)
    
    # ダミーデータ作成
    english_hidden = torch.randn(batch_size, eng_len, d_model)    # Decoder入力
    italian_encoder = torch.randn(batch_size, ita_len, d_model)   # Encoder出力
    
    print(f"\n📥 入力データ:")
    print(f"  english_hidden: {english_hidden.shape}")
    print(f"  italian_encoder: {italian_encoder.shape}")
    print(f"  english_mean: {english_hidden.mean():.4f}")
    print(f"  italian_mean: {italian_encoder.mean():.4f}")
    
    # マスク作成
    # Self-attention用Causal Mask（未来を見ないように）
    causal_mask = torch.triu(torch.ones(eng_len, eng_len), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, -1e9)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, eng_len, eng_len)
    
    # Cross-attention用マスク（イタリア語の一部をマスク）
    cross_mask = torch.ones(batch_size, ita_len)
    cross_mask[0, -2:] = 0  # 最初のサンプルの最後2語をマスク
    cross_mask[1, -3:] = 0  # 2番目のサンプルの最後3語をマスク
    cross_mask = cross_mask.unsqueeze(1).unsqueeze(2)
    cross_mask = (1.0 - cross_mask) * -1e9
    
    print(f"\n🎭 マスク情報:")
    print(f"  causal_mask.shape: {causal_mask.shape}")
    print(f"  cross_mask.shape: {cross_mask.shape}")
    print(f"  cross_mask[0,0,0]: {cross_mask[0,0,0].tolist()}")
    
    # Forward pass
    print(f"\n🚀 DecoderBlock実行...")
    with torch.no_grad():
        output = decoder_block(
            x=english_hidden,
            encoder_output=italian_encoder,
            self_mask=causal_mask,
            cross_mask=cross_mask
        )
    
    print(f"\n📤 出力結果:")
    print(f"  output.shape: {output.shape}")
    print(f"  output.mean: {output.mean():.4f}")
    print(f"  output.std: {output.std():.4f}")
    print(f"  output.min: {output.min():.4f}")
    print(f"  output.max: {output.max():.4f}")
    
    # 形状チェック
    expected_shape = (batch_size, eng_len, d_model)
    assert output.shape == expected_shape, f"形状エラー: {output.shape} != {expected_shape}"
    
    # NaN/Inf チェック
    assert not torch.isnan(output).any(), "NaNが検出されました"
    assert not torch.isinf(output).any(), "Infが検出されました"
    
    print(f"\n✅ 基本チェック通過!")
    
    # 段階別動作確認
    test_decoder_step_by_step(decoder_block, english_hidden, italian_encoder, causal_mask, cross_mask)
    
    print(f"\n🎉 DecoderBlock テスト完了!")
    return output

def test_decoder_step_by_step(decoder_block, english_hidden, italian_encoder, causal_mask, cross_mask):
    """
    DecoderBlockの各ステップを詳細確認
    """
    print(f"\n🔍 段階別動作確認:")
    
    with torch.no_grad():
        x = english_hidden.clone()
        
        # Step 1: Self-Attention
        residual = x
        x_norm = decoder_block.layer_norm1(x)
        x_self_attn = decoder_block.self_attention(x_norm, mask=causal_mask)
        x = decoder_block.dropout(x_self_attn) + residual
        
        print(f"  Step 1 (Self-Attention):")
        print(f"    input_mean: {english_hidden.mean():.4f}")
        print(f"    output_mean: {x.mean():.4f}")
        print(f"    change: {torch.abs(x - english_hidden).mean():.4f}")
        
        # Step 2: Cross-Attention
        residual = x
        x_norm = decoder_block.layer_norm2(x)
        x_cross_attn = decoder_block.cross_attention(
            query_input=x_norm,
            key_value_input=italian_encoder,
            mask=cross_mask
        )
        x = decoder_block.dropout(x_cross_attn) + residual
        
        print(f"  Step 2 (Cross-Attention):")
        print(f"    output_mean: {x.mean():.4f}")
        print(f"    italian_influence: {torch.abs(x_cross_attn).mean():.4f}")
        
        # Step 3: Feed Forward
        residual = x
        x_norm = decoder_block.layer_norm3(x)
        x_ffn = decoder_block.ffn(x_norm)
        x = decoder_block.dropout(x_ffn) + residual
        
        print(f"  Step 3 (Feed Forward):")
        print(f"    output_mean: {x.mean():.4f}")
        print(f"    ffn_change: {torch.abs(x_ffn).mean():.4f}")
        
        # 最終結果との比較
        full_output = decoder_block(english_hidden, italian_encoder, causal_mask, cross_mask)
        diff = torch.abs(x - full_output).max()
        print(f"  ✅ 段階別 vs 一括実行の差分: {diff:.8f}")

def test_decoder_masking_effects():
    """
    DecoderBlockでのマスクの効果を確認
    """
    print(f"\n🎭 マスク効果テスト:")
    
    batch_size, eng_len, ita_len, d_model = 1, 4, 6, 64
    decoder_block = DecoderBlock(d_model, 4, 256, dropout=0.0)
    
    english_hidden = torch.randn(batch_size, eng_len, d_model)
    italian_encoder = torch.randn(batch_size, ita_len, d_model)
    
    # テスト1: マスクなし
    output_no_mask = decoder_block(english_hidden, italian_encoder)
    
    # テスト2: Causal Maskのみ
    causal_mask = torch.triu(torch.ones(eng_len, eng_len), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, -1e9)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    output_causal = decoder_block(english_hidden, italian_encoder, self_mask=causal_mask)
    
    # テスト3: Cross Maskのみ
    cross_mask = torch.zeros(batch_size, 1, 1, ita_len)
    cross_mask[0, 0, 0, -2:] = -1e9  # 最後2語をマスク
    
    output_cross = decoder_block(english_hidden, italian_encoder, cross_mask=cross_mask)
    
    # テスト4: 両方のマスク
    output_both = decoder_block(english_hidden, italian_encoder, causal_mask, cross_mask)
    
    print(f"  マスクなし vs Causal: {torch.abs(output_no_mask - output_causal).mean():.6f}")
    print(f"  マスクなし vs Cross: {torch.abs(output_no_mask - output_cross).mean():.6f}")
    print(f"  マスクなし vs 両方: {torch.abs(output_no_mask - output_both).mean():.6f}")

def test_decoder_edge_cases():
    """
    DecoderBlockのエッジケーステスト
    """
    print(f"\n🚨 DecoderBlock エッジケーステスト:")
    
    d_model = 64
    decoder_block = DecoderBlock(d_model, 4, 256, dropout=0.0)
    
    # テスト1: 最小サイズ
    print("1️⃣ 最小サイズテスト")
    english = torch.randn(1, 1, d_model)
    italian = torch.randn(1, 1, d_model)
    output = decoder_block(english, italian)
    assert output.shape == (1, 1, d_model), "最小サイズテスト失敗"
    print("   ✅ 通過")
    
    # テスト2: 勾配フロー
    print("2️⃣ 勾配フローテスト")
    english = torch.randn(1, 3, d_model, requires_grad=True)
    italian = torch.randn(1, 4, d_model, requires_grad=True)
    output = decoder_block(english, italian)
    loss = output.sum()
    loss.backward()
    assert english.grad is not None and italian.grad is not None, "勾配計算失敗"
    print("   ✅ 通過")
    
    print("✅ 全エッジケーステスト通過!")

def main():
    """
    DecoderBlock総合テスト
    """
    print("🚀 DecoderBlock 総合テスト開始")
    print("=" * 60)
    
    try:
        # 基本テスト
        output = test_decoder_block()
        
        # マスク効果テスト
        test_decoder_masking_effects()
        
        # エッジケーステスト
        test_decoder_edge_cases()
        
        print(f"\n🏆 全テスト完了! DecoderBlockは正常に動作しています。")
        
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
def main():
    #test_bert_embeddings()
    #test_edge_cases()
    #test_bert_masking()
    print("🚀 CrossAttention 総合テスト開始")
    print("=" * 60)
    
    try:
        # 基本テスト
        output = test_cross_attention()
        
        # エッジケーステスト
        test_cross_attention_edge_cases()
        
        print(f"\n🏆 全テスト完了! CrossAttentionは正常に動作しています。")
        
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""



def visualize_tensor_transformation():
    """
    テンソル変形の可視化
    """
    print("🔍 Multi-Head Attention のテンソル変形を可視化")
    print("=" * 60)
    
    # 例: 小さなサイズで実験
    batch_size, seq_len, d_model = 2, 4, 8
    num_heads, head_dim = 2, 4  # d_model = num_heads * head_dim
    
    # ダミーデータ
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"入力 x: {x.shape}")
    
    # 線形変換（簡略化のため、恒等変換）
    query = x
    print(f"Q (線形変換後): {query.shape}")
    
    # Step 1: view操作
    query_reshaped = query.view(batch_size, seq_len, num_heads, head_dim)
    print(f"Q (view後): {query_reshaped.shape}")
    print("→ d_modelをnum_heads個のhead_dimに分割")
    
    # Step 2: permute操作  
    query_permuted = query_reshaped.permute(0, 2, 1, 3)
    print(f"Q (permute後): {query_permuted.shape}")
    print("→ ヘッド次元を前に移動（並列処理のため）")
    
    # 実際の計算例
    key_permuted = query_permuted  # 簡略化
    
    # 行列積の実行
    scores = torch.matmul(query_permuted, key_permuted.transpose(-2, -1))
    print(f"scores: {scores.shape}")
    print("→ (batch, num_heads, seq_len, seq_len) のattention matrix")
    
    print("\n💡 重要ポイント:")
    print("- permute後は (batch, num_heads, seq_len, head_dim)")
    print("- 最後の2次元 (seq_len, head_dim) で行列積")
    print("- num_heads分を並列で計算！")

# 実行例（コメントアウトを外して試してみてください）
# visualize_tensor_transformation()

def compare_ffn_activations():
    """
    最後の活性化ありなしの比較実験
    """
    print("🧪 FFN最終層の活性化関数比較実験")
    print("=" * 50)
    
    d_model, d_ff, batch_size, seq_len = 512, 2048, 2, 10
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. 標準的なFFN（最後に活性化なし）
    class StandardFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.gelu = nn.GELU()
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.gelu(x)        # 中間層のみ活性化
            x = self.linear2(x)     # 最後は線形
            return x
    
    # 2. 最後にも活性化を入れたFFN（実験用）
    class ActivatedFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.gelu = nn.GELU()
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.gelu(x)        # 中間層活性化
            x = self.linear2(x)
            x = self.gelu(x)        # 最後にも活性化（実験）
            return x
    
    standard_ffn = StandardFFN()
    activated_ffn = ActivatedFFN()
    
    # 出力の比較
    with torch.no_grad():
        standard_out = standard_ffn(x)
        activated_out = activated_ffn(x)
        
        print(f"入力の範囲: [{x.min():.3f}, {x.max():.3f}]")
        print(f"標準FFN出力: [{standard_out.min():.3f}, {standard_out.max():.3f}]")
        print(f"活性化FFN出力: [{activated_out.min():.3f}, {activated_out.max():.3f}]")
        
        # 負の値の割合
        standard_neg_ratio = (standard_out < 0).float().mean()
        activated_neg_ratio = (activated_out < 0).float().mean()
        
        print(f"\n📊 負の値の割合:")
        print(f"標準FFN: {standard_neg_ratio:.1%}")
        print(f"活性化FFN: {activated_neg_ratio:.1%}")
        
        print(f"\n💡 結論:")
        print(f"- 標準FFNは正負両方の値を出力（表現力が高い）")
        print(f"- 活性化FFNは正の値のみ（表現力が制限される）")

# 実行例（コメントアウトを外してください）
# compare_ffn_activations()

# 📚 他のアーキテクチャとの比較
"""
🔍 他のニューラルネットワークとの違い:

1. **CNN (畳み込みニューラルネット)**
   - 各層でReLU活性化が一般的
   - 特徴抽出が目的

2. **RNN/LSTM**
   - 隠れ状態でtanh/sigmoidを使用
   - ゲート機構で制御

3. **Transformer/BERT**
   - 中間層のみGELU活性化
   - 最終層は線形（Residual + LayerNorm）

4. **MLP (多層パーセプトロン)**
   - 通常は各層で活性化
   - 分類問題では最後にSoftmax

🎯 Transformerが特殊な理由:
- Residual Connection の存在
- LayerNormalization の後処理
- Attention機構との協調
"""
