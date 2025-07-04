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
        # 🔍 FeedForward の正しい順序と活性化について
        
        # 1. 第1層: d_model → d_ff (次元拡張)
        x = self.linear1(x)          # 線形変換
        x = self.gelu(x)             # 活性化関数 (中間層のみ)
        x = self.dropout(x)          # ドロップアウト
        
        # 2. 第2層: d_ff → d_model (次元復元)
        x = self.linear2(x)          # 線形変換
        x = self.dropout(x)          # ドロップアウト
        
        
        return x  # 出力次元は元のd_modelに戻す
        
        # 💡 なぜ最後に活性化しないのか？
        # 1. Residual Connection: x + FFN(x) で加算するため
        # 2. 表現の柔軟性: 負の値も重要な情報
        # 3. Transformer設計: 最終的にはLayerNormが正規化
        

class TransformerBlock(nn.Module):
    """
    ヒント:
    - Multi-Head Attention + Residual Connection + Layer Norm
    - Feed Forward + Residual Connection + Layer Norm
    - 順序に注意: Pre-LN vs Post-LN
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: アテンション、FFN、正規化層を組み合わせ
        pass
    
    def forward(self, x, mask=None):
        # TODO: Transformer blockの処理を実装
        pass

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
        pass
    
    def forward(self, input_ids, token_type_ids=None):
        # TODO: 埋め込みの計算を実装
        pass

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
def main():
    x = torch.randn(2, 10, 512)  # (batch, seq, d_model)
    attn = ShowMultiHeadAttention(512, 8)
    out = attn(x)
    assert out.shape == x.shape
    return 0

main()

# 実装のチェックポイント:
"""
🔍 段階的なテスト方法:

1. MultiHeadAttention単体テスト:
   x = torch.randn(2, 10, 512)  # (batch, seq, d_model)
   attn = MultiHeadAttention(512, 8)
   out = attn(x)
   assert out.shape == x.shape

2. TransformerBlock単体テスト:
   block = TransformerBlock(512, 8, 2048)
   out = block(x)
   assert out.shape == x.shape

3. BERT全体テスト:
   bert = Bert(vocab_size=30000, d_model=512, num_layers=6, num_heads=8)
   input_ids = torch.randint(0, 30000, (2, 50))
   out = bert(input_ids)
   print(f"Output shape: {out.shape}")

📚 参考になる実装のヒント:
- "Attention Is All You Need" (Transformer原論文)
- "BERT: Pre-training of Deep Bidirectional Transformers" (BERT論文)
- PyTorchの nn.TransformerEncoder の実装を参考に（ただしコピーしない）

🎯 実装後の課題:
1. 小さなデータセットでMLMタスクを学習
2. Attention weightsの可視化
3. [CLS]トークンの分類性能評価
"""

# ==========================================
# 🛠️ 実装の道しるべ - 詳細ガイド
# ==========================================

"""
📖 STEP 1: Multi-Head Attention を理解・実装

核心的な質問:
Q1. なぜ「Multi」Head なのか？
→ 異なる種類の関係性（構文的、意味的）を並列で捉えるため

Q2. Scaled Dot-Product Attention の式は？
→ Attention(Q,K,V) = softmax(QK^T / √d_k)V

Q3. Maskingはなぜ必要？
→ PADトークンへのattentionを防ぐため（-infを代入してsoftmax→0）

実装のポイント:
- d_model を num_heads で割り切れる必要がある
- reshape操作で(batch, seq, num_heads, head_dim)に変形
- Einstein notation (torch.einsum) が便利
- attention_weightsを保存すると可視化に便利
"""

class AttentionVisualization:
    """
    Attention重みの可視化用ユーティリティ
    
    使用例:
    viz = AttentionVisualization()
    viz.plot_attention(attention_weights, tokens, head_idx=0)
    """
    @staticmethod
    def plot_attention(attention_weights, tokens, head_idx=0):
        # TODO: matplotlib/seabornでheatmap描画
        # ヒント: attention_weights.shape = (batch, num_heads, seq_len, seq_len)
        pass

"""
📖 STEP 2: Position Embeddings の重要性

核心的な質問:
Q1. なぜ位置情報が必要？
→ Transformerは順序を考慮しない。「I love you」と「You love I」が同じになってしまう

Q2. 学習可能 vs 固定式 のどちらを選ぶ？
→ BERTは学習可能。GPTは固定式（sin/cos）。どちらでも実装可能

Q3. 絶対位置 vs 相対位置？
→ BERTは絶対位置。最近の研究では相対位置も注目

実装のコツ:
- max_seq_lenまでの位置埋め込みを事前に作成
- position_ids = torch.arange(seq_len) で位置インデックス生成
"""

"""
📖 STEP 3: Layer Normalization の配置

核心的な質問:  
Q1. Pre-LN vs Post-LN の違いは？
→ Pre-LN: 学習が安定、勾配消失が起きにくい（最近の主流）
→ Post-LN: 原論文通り、しかし深い層で不安定

Q2. BatchNorm vs LayerNorm？
→ LayerNorm: 系列長が可変でも安定
→ BatchNorm: バッチサイズに依存

実装例:
# Pre-LN (推奨)
residual = x
x = self.layer_norm1(x)
x = self.attention(x)
x = x + residual

# Post-LN (原論文)
residual = x  
x = self.attention(x)
x = self.layer_norm1(x + residual)
"""

"""
📖 STEP 4: BERTの事前学習タスク

MLM (Masked Language Model):
- 15%の単語をマスク: 80%→[MASK], 10%→ランダム単語, 10%→そのまま
- CrossEntropyLossでマスクされた位置のみを予測

NSP (Next Sentence Prediction):  
- 50%は連続する文、50%はランダムな文
- [CLS]トークンの表現で二値分類

実装のヒント:
class BertForPreTraining(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.mlm_head = nn.Linear(bert_model.d_model, bert_model.vocab_size)
        self.nsp_head = nn.Linear(bert_model.d_model, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # TODO: MLMとNSPの損失を計算
        pass
"""

"""
📖 STEP 5: デバッグとテストの戦略

🐛 よくあるバグと対処法:

1. Shape Mismatch:
   print(f"Expected: {expected_shape}, Got: {tensor.shape}")
   
2. Gradient Explosion/Vanishing:
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   
3. NaN Loss:
   assert not torch.isnan(loss), f"NaN loss detected at step {step}"
   
4. Memory Error:
   torch.cuda.empty_cache()  # GPU memory cleanup
   
🧪 単体テストの例:
def test_attention():
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    attn = MultiHeadAttention(d_model, num_heads=8)
    out = attn(x)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert not torch.isnan(out).any(), "NaN detected in output"
    assert not torch.isinf(out).any(), "Inf detected in output"
    
    print("✅ MultiHeadAttention test passed!")

📊 パフォーマンス測定:
def benchmark_model(model, input_shape, num_runs=100):
    import time
    
    dummy_input = torch.randn(*input_shape)
    model.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    avg_time = (time.time() - start_time) / num_runs
    print(f"Average inference time: {avg_time*1000:.2f}ms")
"""

# ==========================================
# 🎯 実装完了後の発展課題
# ==========================================

"""
1. 🔥 Advanced Techniques:
   - RoPE (Rotary Position Embedding)
   - Flash Attention (メモリ効率化)
   - LoRA (Low-Rank Adaptation)

2. 📈 性能改善:
   - Mixed Precision Training (torch.cuda.amp)
   - Gradient Checkpointing (メモリ節約)
   - DataParallel / DistributedDataParallel

3. 🔍 分析・可視化:
   - Attention Pattern の可視化
   - Embedding Space の可視化 (t-SNE, UMAP)
   - Layer-wise Learning Rate Decay

4. 🚀 実用化:
   - ONNX Export (推論最適化)
   - TensorRT Optimization
   - Quantization (8bit/16bit)
"""

# 🤔 なぜ view + permute が必要なのか？詳細解説
"""
質問: 「viewとpermuteで次元を変えているのはなぜ？」

💡 答え: **効率的なバッチ並列処理のため**

🔍 具体例で理解しよう:
d_model=512, num_heads=8, head_dim=64 とする

1️⃣ 元の形状:
   Q, K, V: (batch=2, seq_len=10, d_model=512)

2️⃣ view操作の理由:
   512次元を8つのヘッドに分割
   (2, 10, 512) → (2, 10, 8, 64)
   ↑ 512 = 8 × 64 に分解

3️⃣ permute操作の理由:
   (2, 10, 8, 64) → (2, 8, 10, 64)
   ↑ ヘッド次元を前に持ってくる

4️⃣ なぜこの順序？
   行列積: (2, 8, 10, 64) @ (2, 8, 64, 10) → (2, 8, 10, 10)
   ↑ PyTorchは最後の2次元で行列積を計算
   ↑ 8つのヘッドを**並列**で処理できる！

❌ もしpermuteしなかったら:
   (2, 10, 8, 64) の形状では、ヘッド次元が最後から2番目にあり、
   行列積が複雑になってしまう

✅ permuteすることで:
   - 8つのヘッドを同時に処理
   - GPUでの並列計算が効率的
   - メモリアクセスパターンが最適化
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

# 📊 メモリ効率と計算効率の比較
"""
🚀 なぜ効率的なのか？

1. **メモリアクセスパターン**:
   - 連続したメモリアクセス
   - キャッシュヒット率向上

2. **並列処理**:
   - 8つのヘッドを同時計算
   - GPUの複数コアを活用

3. **行列積の最適化**:
   - BLAS (Basic Linear Algebra Subprograms) を活用
   - ハードウェア最適化の恩恵

❌ 非効率な実装例:
for head_idx in range(num_heads):
    # 各ヘッドを順次処理（遅い！）
    q_head = query[:, :, head_idx*head_dim:(head_idx+1)*head_dim]
    # ... 計算 ...

✅ 効率的な実装（現在のコード）:
# 全ヘッドを並列処理
scores = torch.matmul(query_all_heads, key_all_heads.transpose(-2, -1))
"""

# 🤔 FeedForward の活性化関数について詳細解説
"""
質問: 「FeedForwardは最後は活性化しないの？」

💡 答え: **最後は活性化しません！**

📊 理由の詳細:

1️⃣ **Residual Connection のため**
   TransformerBlockでは: output = x + FFN(x)
   → FFN(x)が制限されると、Residual学習が阻害される

2️⃣ **表現の柔軟性**
   - 正の値だけでなく負の値も重要な情報
   - ReLUやGELUは負の値を制限してしまう

3️⃣ **Layer Normalization が後処理**
   - FFN → LayerNorm の順序で正規化される
   - LayerNormが適切に値を調整

4️⃣ **標準的なTransformer設計**
   - 原論文 "Attention Is All You Need" でも最後は線形
   - BERT, GPT等も同様の設計

🔬 実験的な比較:
"""

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
