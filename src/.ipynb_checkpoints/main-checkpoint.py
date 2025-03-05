from sentence_transformers import SentenceTransformer, util
import torch

# Initialize model with explicit architectural choices
model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Sample sentences for demonstration
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]

# Generate embeddings with configurable parameters
embeddings = model.encode(
    sentences,
    convert_to_tensor=True,
    show_progress_bar=False,
    normalize_embeddings=True,
    batch_size=32
)

print(f"Embedding shape: {embeddings.shape}")  # Should be (3, 384)

# Calculate cosine similarity between embeddings
cos_sim = util.cos_sim(embeddings, embeddings)
print(f"\nCosine similarity matrix:\n{cos_sim}")
