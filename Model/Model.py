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
            # mask形状: (batch, 1, 1, seq_len) → scores形状: (batch, num_heads, seq_len, seq_len)
            scores = scores + mask  # ブロードキャストで加算
        
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
        self.d_model = d_model
        self.num_layers = num_layers
        self.heads = num_heads
        # paper noted 4*d_model size for ff
        self.feed_forward_hidden = d_model * 4
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BertEmbeddings(vocab_size, d_model, max_seq_len, dropout)

        self.encoder_blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_model * 4, dropout) for _ in range(num_layers)])
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # TODO: BERT全体のforward passを実装
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        # (batch, seq_len) → (batch, 1, 1, seq_len)
        batch_size, seq_len = input_ids.shape
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # 0を-1e9に変換（Softmaxで0になるように）
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(input_ids, token_type_ids)
        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, extended_attention_mask)
        return x
