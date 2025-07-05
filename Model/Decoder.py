import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Encoder import MultiHeadAttention, PositionwiseFeedForward

class CrossAttention(nn.Module):
    """
    this module is implemented with modifying MultiHeadAttention.
    Query: English
    Key, Value: Italian
    You can see the difference in forward input
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__() # initialization
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # dimention of each head
        self.dropout = torch.nn.Dropout(dropout)
        
        # Q, K, V の線形変換（修正：torch.nn.linear → torch.nn.Linear）
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        
        # 最終的な出力変換
        self.out_proj = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query_input, key_value_input, mask=None): # here is the difference
        batch_size, q_len, _ = query_input.shape
        _, kv_len, _ = key_value_input.shape
        # ステップ1: Q, K, V を線形変換で生成
        query = self.query(query_input)  # (batch, seq_len, d_model)
        key = self.key(key_value_input)      # (batch, seq_len, d_model)
        value = self.value(key_value_input)  # (batch, seq_len, d_model)
        
        # ステップ2: Multi-Head用に次元を変形
        query = query.view(batch_size, q_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, kv_len, self.num_heads, self.head_dim)  # 修正: query.shape → batch_size
        value = value.view(batch_size, kv_len, self.num_heads, self.head_dim)
        
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
        context = context.contiguous().view(batch_size, q_len, self.num_heads * self.head_dim)
    
        # ステップ9: 最終的な線形変換
        return self.out_proj(context)  # 修正: output_linear → out_proj
    
class DecoderBlock(nn.Module):
    """
    Basically similar to EncoderBlock, but refer to the infomation of Input(English context)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        #First, implement Self Attention
        self.self_attention = MultiHeadAttention(d_model,num_heads)
        #Second, implement Cross Attention
        self.cross_attention = CrossAttention(d_model, num_heads)        
        #Third, FFNN
        self.ffn = PositionwiseFeedForward(d_model,d_ff)

        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        #Self Attention
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attention(x,mask=self_mask)
        x = self.dropout(x) + residual

        #Cross Attention
        residual = x
        x = self.layer_norm1(x)
        x = self.cross_attention(
            query_input=x,
            key_value_input=encoder_output,
            mask=cross_mask
        )
        x = self.dropout(x) + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual
        return x