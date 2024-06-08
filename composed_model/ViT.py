import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class encoder_block(nn.Module):
    def __init__(self,input_dim,hidden_dim,embed_dim,num_heads,dropout=0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.attention=nn.MultiheadAttention(embed_dim=num_heads*embed_dim,num_heads=num_heads)
        self.FeedForward=FeedForward(input_dim=input_dim,hidden_dim=hidden_dim,dropout=dropout)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        i=x
        x=self.norm(x)
        x=self.attention(x)
        x=self.dropout(x)
        x=x+i

        i=x
        x=self.norm(x)
        x=self.FeedForward(x)
        x=self.dropout(x)
        x=x+i
        return x

class encoder(nn.Module):
    def __init__(self,input_dim,hidden_mlp_dim,embed_dim,num_heads,num_block,dropout=0) -> None:
        super().__init__()
        self.block=encoder_block(input_dim,hidden_mlp_dim,embed_dim,num_heads,dropout=dropout)
        self.blocks=nn.Sequential()
        for _ in range(num_block):
            self.blocks.append(self.block)    
    def forward(self,x):
        x=self.blocks(x)


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, num_block, num_heads, hidden_mlp_dim,embed_dim, in_channels = 3,dropout = 0.):
        super().__init__()
        self.num_patches=(image_size//patch_size)**2
        self.patch_size=patch_size
        self.norm=nn.LayerNorm(dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.Channel_Compression=nn.Conv2d(in_channels=in_channels,out_channels=dim, kernel_size=1, stride=1)
        self.conv_proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=self.patch_size, stride=self.patch_size)


        self.encoder=encoder(input_dim=dim,hidden_mlp_dim=hidden_mlp_dim,embed_dim=embed_dim,num_heads=num_heads,num_block=num_block,dropout=dropout)

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        batch_size=img.shape[0]

        temp=self.Channel_Compression(img)
        patch=self.conv_proj(temp)

        patch=patch.view(batch_size,self.num_patches,-1)
        
        batch_class_token = self.class_token.expand(batch_size, -1, -1)

        patch = torch.cat([batch_class_token,patch], dim=1)

        embedding = self.pos_embedding+patch

        feature=self.encoder(embedding )
       
        feature=  feature[:, 0]

        output = self.mlp_head(feature)

        return output
if __name__=="__main__":
    model=ViT(image_size=20, patch_size=10, num_classes=5, dim=10, num_block=3, num_heads=2, hidden_mlp_dim=10,embed_dim=4, in_channels = 3,dropout = 0.)
    print(model)


        


