import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    """
    Pure PyTorch implementation of the Mamba Block (Selective State Space Model).
    Designed to avoid custom CUDA kernels for flawless Windows compatibility and ONNX/TFLite export.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # Delta, B, C projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # State space parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: shape (Batch, SequenceLength, d_model)
        Returns: (Batch, SequenceLength, d_model)
        """
        B, L, _ = x.shape
        # 1. Input projection and split
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        # 2. 1D Convolution
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 3. State Space Model (Selective Scan)
        x_proj_out = self.x_proj(x_conv)
        delta, B_mat, C_mat = torch.split(x_proj_out, [1, self.d_state, self.d_state], dim=-1)
        
        delta = self.dt_proj(delta) # Project first (B, L, d_inner)
        delta = F.softplus(delta)   # Softplus AFTER projection to guarantee strictly positive delta!
        
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # Pure PyTorch Selective Scan Loop
        # Excellent for ONNX export and small sequence lengths (e.g. ECG 187-300 length).
        ys = []
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        
        for t in range(L):
            delta_t = delta[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            B_t = B_mat[:, t, :].unsqueeze(1)      # (B, 1, d_state)
            C_t = C_mat[:, t, :].unsqueeze(1)      # (B, 1, d_state)
            x_t = x_conv[:, t, :].unsqueeze(-1)    # (B, d_inner, 1)
            
            A_bar = torch.exp(delta_t * A)         # (B, d_inner, d_state)
            B_bar = delta_t * B_t                  # (B, d_inner, d_state)
            
            h = A_bar * h + B_bar * x_t            # (B, d_inner, d_state)
            y = (h * C_t).sum(dim=-1)              # (B, d_inner)
            ys.append(y)
            
        y = torch.stack(ys, dim=1) # (B, L, d_inner)
        
        # 4. Skip connection and output projection
        y = y + x_conv * self.D
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        return out


class MambaECGClassifier(nn.Module):
    """
    Main Model for ECG Classification. 
    Combines embedding layer, multiple Mamba Blocks, and classification head.
    """
    def __init__(self, input_dim=1, num_classes=2, d_model=64, num_layers=2, d_state=16):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        x: (Batch, SequenceLength, Channels). For ECG, Channels=1.
        """
        # Embed single channel signal into d_model dimension
        x = self.embedding(x)
        
        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        # Global Average Pooling across the sequence length
        x = x.mean(dim=1) 
        
        # Fully Connected Classifier
        out = self.fc(x)
        return out

# For testing shapes locally
if __name__ == "__main__":
    dummy_input = torch.randn(8, 187, 1) # Batch=8, Length=187, Channels=1
    model = MambaECGClassifier(input_dim=1, num_classes=2, d_model=64, num_layers=2)
    output = model(dummy_input)
    print("Dummy Output Shape:", output.shape) # Expected: (8, 5)
