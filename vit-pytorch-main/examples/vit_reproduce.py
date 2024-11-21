import torch
from torch import nn

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, transformer, pool='cls', channels=3):
        super().__init__()
        # Assertions
        self.image_height, self.image_width = pair(image_size)
        self.patch_size = patch_size
        self.num_patches = (self.image_height // patch_size) * (self.image_width // patch_size)
        self.patch_dim = channels * patch_size ** 2
        assert self.image_height % patch_size == 0 and self.image_width % patch_size == 0, \
            "Image dimensions must be divisible by the patch size."
        assert pool in {'cls', 'mean'}, "Pool type must be 'cls' or 'mean'."

        # Components
        self.channels = channels
        self.dim = dim
        self.patch_embedding = self.create_patch_embedding()
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer
        self.pool_type = pool
        self.mlp_head = self.create_mlp_head(dim, num_classes)

    def create_patch_embedding(self):
        """Creates the patch embedding module."""
        return nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim),
            nn.LayerNorm(self.dim)
        )

    def create_mlp_head(self, dim, num_classes):
        """Creates the MLP head for classification."""
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def extract_patches(self, img):
        """
        Splits an image into patches and flattens each patch.

        Input: 
            img (batch_size, channels, height, width)
        Output:
            patches (batch_size, num_patches, patch_dim)
        """
        batch_size, channels, height, width = img.shape
        patch_height, patch_width = self.patch_size, self.patch_size
        assert height == self.image_height and width == self.image_width, \
            "Input image size does not match model's expected dimensions."
        
        # Reshape to batch_size, num_patches_row, patch_height, num_patches_col, patch_width, channels
        img = img.view(batch_size, channels, height // patch_height, patch_height, width // patch_width, patch_width)
        
        # Permute to bring patch dimensions together: batch_size, num_patches_row, num_patches_col, patch_height, patch_width, channels
        img = img.permute(0, 2, 4, 3, 5, 1)
        
        # Reshape to flatten patches: batch_size, num_patches, patch_dim
        patches = img.reshape(batch_size, -1, patch_height * patch_width * channels)
        return patches

    def prepare_tokens(self, patches):
        """Prepends the class token and adds positional embeddings."""
        batch_size, num_patches, _ = patches.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand cls_token for each batch
        tokens = torch.cat((cls_tokens, patches), dim=1)  # Concatenate class token with patches
        return tokens + self.positional_embedding[:, :tokens.size(1)]

    def forward(self, img):
        # Step 1: Extract patches
        patches = self.extract_patches(img)

        # Step 2: Patch Embedding
        patches = self.patch_embedding(patches)

        # Step 3: Prepare Tokens
        tokens = self.prepare_tokens(patches)

        # Step 4: Apply Transformer
        transformed_tokens = self.transformer(tokens)

        # Step 5: Pooling
        if self.pool_type == 'mean':
            features = transformed_tokens.mean(dim=1)
        else:
            features = transformed_tokens[:, 0]  # CLS token

        # Step 6: Classification Head
        return self.mlp_head(features)