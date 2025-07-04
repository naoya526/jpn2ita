import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader

# I tried to implement a BERT model from scratch using PyTorch.

# 1. まず基本的なビルディングブロックから
class ShowMultiHeadAttention(nn.Module):
    """
    ヒント: 
    - Query, Key, Value の3つの線形変換が必要
    - スケールドドット積アテンション: softmax(QK^T / sqrt(d_k))V
    - 複数のヘッドを並列実行後、concatenate
    - 最終的な線形変換で出力次元を調整
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__() # 親クラスの初期化
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 各ヘッドの次元
        self.dropout = torch.nn.Dropout(dropout)
        
        # Q, K, V の線形変換（修正：torch.nn.linear → torch.nn.Linear）
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        
        # 最終的な出力変換
        self.out_proj = torch.nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # ステップ1: Q, K, V を線形変換で生成
        query = self.query(x)  # (batch, seq_len, d_model)
        key = self.key(x)      # (batch, seq_len, d_model)
        value = self.value(x)  # (batch, seq_len, d_model)
        
        print(f"Before View: Q={query.shape}, K={key.shape}, V={value.shape}")
        
        # ステップ2: Multi-Head用に次元を変形
        # 理由1: d_model を num_heads 個のヘッドに分割
        # (batch, seq_len, d_model) → (batch, seq_len, num_heads, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)  # 修正: query.shape → batch_size
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        print(f"After View: Q={query.shape}, K={key.shape}, V={value.shape}")
        
        # ステップ3: 次元の順序を変更（これが質問のポイント！）
        # 理由2: バッチ並列処理のため (batch, num_heads, seq_len, head_dim)
        query = query.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        
        print(f"After Permute: Q={query.shape}, K={key.shape}, V={value.shape}")
        
        # ステップ4: Scaled Dot-Product Attention
        # scores = Q @ K^T / sqrt(d_k)
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # → (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        print(f"scores: {scores.shape}")
        
        # ステップ5: マスク処理（オプション）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # ステップ6: Softmax + Dropout
        weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        weights = self.dropout(weights)
        
        # ステップ7: Value との積
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # → (batch, num_heads, seq_len, head_dim)
        context = torch.matmul(weights, value)
        print(f"context: {context.shape}")
        
        # ステップ8: ヘッドを結合して元の形状に戻す
        # (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim)
        context = context.permute(0, 2, 1, 3)
        # → (batch, seq_len, d_model)
        context = context.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        print(f"Result: {context.shape}")
        
        # ステップ9: 最終的な線形変換
        return self.out_proj(context)  # 修正: output_linear → out_proj
class MultiHeadAttention(nn.Module):
    """
    ヒント: 
    - Query, Key, Value の3つの線形変換が必要
    - スケールドドット積アテンション: softmax(QK^T / sqrt(d_k))V
    - 複数のヘッドを並列実行後、concatenate
    - 最終的な線形変換で出力次元を調整
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__() # 親クラスの初期化
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 各ヘッドの次元
        self.dropout = torch.nn.Dropout(dropout)
        
        # Q, K, V の線形変換（修正：torch.nn.linear → torch.nn.Linear）
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        
        # 最終的な出力変換
        self.out_proj = torch.nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # ステップ1: Q, K, V を線形変換で生成
        query = self.query(x)  # (batch, seq_len, d_model)
        key = self.key(x)      # (batch, seq_len, d_model)
        value = self.value(x)  # (batch, seq_len, d_model)
        
        # ステップ2: Multi-Head用に次元を変形
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)  # 修正: query.shape → batch_size
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # ステップ3: 次元の順序を変更（これが質問のポイント！）
        query = query.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
                
        # ステップ4: Scaled Dot-Product Attention
        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # ステップ5: マスク処理（オプション）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # ステップ6: Softmax + Dropout
        weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        weights = self.dropout(weights)
        # ステップ7: Value との積
        context = torch.matmul(weights, value)    
        # ステップ8: ヘッドを結合して元の形状に戻す
        context = context.permute(0, 2, 1, 3)
        # → (batch, seq_len, d_model)
        context = context.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
    
        # ステップ9: 最終的な線形変換
        return self.out_proj(context)  # 修正: output_linear → out_proj
    
class PositionwiseFeedForward(nn.Module):
    """
    ヒント:
    - 2層のフィードフォワードネットワーク
    - 中間層では次元を拡張（通常4倍）
    - GELU活性化関数を使用
    - ドロップアウトも忘れずに
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 入力次元 → 中間次元
        self.linear2 = nn.Linear(d_ff, d_model)  # 中間次元
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)          
        x = self.gelu(x)             
        x = self.dropout(x)          
        x = self.linear2(x)          
        x = self.dropout(x)            
        return x  
      

class TransformerBlock(nn.Module):
    """
    ヒント:
    - Multi-Head Attention + Residual Connection + Layer Norm
    - Feed Forward + Residual Connection + Layer Norm
    - 順序に注意: Pre-LN vs Post-LN
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model,num_heads)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model,d_ff)
    def forward(self, x, mask=None):
        #Attention block
        #TODO implement transformer block
        residual = x
        #print("Took Residual...",x.shape)
        x = self.layer_norm1(x)
        #print("calculating layer norm...",x.shape)
        x = self.dropout(self.attention(x,mask))
        #print("calculating Attention...",x.shape)
        x = x + residual
        #print("calculating Residual Connection...",x.shape)
        #ffnn
        residual = x
        x = self.layer_norm2(x)
        #print("calculating layer norm...",x.shape)
        x = self.dropout(self.ffn(x))
        #print("calculating ffn...",x.shape)
        x = x + residual
        return x


class BertEmbeddings(nn.Module):
    """
    ヒント:
    - Token Embeddings (語彙サイズ × d_model)
    - Position Embeddings (最大系列長 × d_model) 
    - Segment Embeddings (2 × d_model, NSPタスク用)
    - 3つを足し合わせてLayerNormとDropout
    """
    def __init__(self, vocab_size, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        # TODO: 3種類の埋め込みを実装
        self.d_model = d_model
        self.token = torch.nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = torch.nn.Embedding(max_seq_len, d_model)
        self.segment = torch.nn.Embedding(2, d_model)  # 2つのセグメント（0と1）
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #Embedding: Lookup table that keep meaning vector of words
    def forward(self, input_ids, token_type_ids=None):
        # TODO: 埋め込みの計算を実装
        batch_size, seq_len = input_ids.shape
        # Step 1: Token Embeddings
        token_embeddings = self.token(input_ids)
        # Step 2: Position Embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)  # 🔧 バッチ次元を拡張
        position_embeddings = self.position(position_ids)
        # Step 3: Segment Embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)  # 全て0（単一文）
        segment_embeddings = self.segment(token_type_ids)  # (batch, seq_len, d_model)
        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.dropout(self.layer_norm(embeddings))

        return embeddings
        
class Bert(nn.Module):
    """
    BERT実装の最終形
    
    学習のヒント:
    1. 論文を読んで全体像を理解
    2. 小さな部品から実装（Attention → FFN → Block → Full Model）
    3. 各層で print(tensor.shape) してサイズを確認
    4. 簡単なダミーデータでテスト
    5. 事前学習は計算量が大きいので、小さいモデルから開始
    
    重要な概念:
    - Bidirectional: 左右両方向の文脈を見る
    - Masked Language Model: ランダムにマスクした単語を予測
    - Next Sentence Prediction: 2つの文が連続するかを予測
    - Attention Weights: どの単語に注目しているかの可視化
    """
    
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072, max_seq_len=512, dropout=0.1):
        super().__init__()
        # TODO: BERTの全体構造を実装
        pass
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # TODO: BERT全体のforward passを実装
        pass

# main

def exp_Transformer():
    d_model = 512
    seq = 10
    batch_size = 2
    num_heads=2
    d_ff=10

    x = torch.randn(batch_size, seq, d_model)  # (batch, seq, d_model)
    print("start")
    func = TransformerBlock(d_model, num_heads, d_ff)
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

def main():
    test_bert_embeddings()
    test_edge_cases()
    return 0

main()



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
