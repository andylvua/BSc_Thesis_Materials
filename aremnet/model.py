import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(attn_output + x) 
        return x.permute(0, 2, 1)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pos_embed[:, :x.size(1)]
        return self.dropout(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        batch_size, channels, _ = x.size()
        out = self.squeeze(x).view(batch_size, channels)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return x * out.unsqueeze(-1)

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualUnit, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.se_block = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.se_block(out)

        out += residual
        return F.relu(out)

class ReductionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReductionBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        return out1 + out2

class ResidualReductionModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, attention_unit, max_len=133):
        super(ResidualReductionModule, self).__init__()
        self.attention_unit = attention_unit

        if attention_unit:
            self.pos_encoding = LearnablePositionalEncoding(max_len=max_len, embed_dim=in_channels)
            self.self_attention = SelfAttentionBlock(embed_dim=out_channels, num_heads=num_heads)
        
        self.residual_unit = ResidualUnit(in_channels, out_channels)

        self.reduction_block = ReductionBlock(out_channels, out_channels)
    
    def forward(self, x):
    
        if self.attention_unit:
            x = x.permute(0, 2, 1)
            x = self.pos_encoding(x)
            x = x.permute(0, 2, 1)
            x = self.self_attention(x)

        residual = self.residual_unit(x)

        reduced = self.reduction_block(residual)
        return reduced

class ARemNet(pl.LightningModule):
    def __init__(self, k, input_channels, low_level_features, num_rrms, num_heads, attention_unit=True, num_classes=1, lr=0.001):
        super(ARemNet, self).__init__()
        self.save_hyperparameters()
        n_attention = 2
        
        self.conv1d = nn.Conv1d(input_channels, low_level_features, kernel_size=9, padding=4)

        self.rrms = nn.ModuleList([
            ResidualReductionModule(
                low_level_features, 
                low_level_features, 
                num_heads, 
                attention_unit = (attention_unit and i < n_attention),
                max_len=k // 2 ** i
            ) for i in range(num_rrms)
        ])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear((k // 2 ** num_rrms) * low_level_features, num_classes)
        self.lr = lr
        self.criterion = nn.SmoothL1Loss(beta=1.0)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        out = F.relu(self.conv1d(x))
 
        for rrm in self.rrms:
            out = rrm(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return self.fc(out)

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs).squeeze()
        loss = self.criterion(outputs, labels.float())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs).squeeze()
        loss = self.criterion(outputs, labels.float())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs).squeeze()
        loss = self.criterion(outputs, labels.float())
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)