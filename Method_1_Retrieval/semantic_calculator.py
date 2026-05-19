
import torch
from sentence_transformers import SentenceTransformer, util

# Step 1: Initialize the Multilingual Model (Optimized for English/Hindi)
# This model converts a WHOLE sentence into a 384-dimensional vector.
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
print(f"Loading model: {model_name}...")
model = SentenceTransformer(model_name)
print("Model loaded.\n")

def get_sentence_vector(sentence):
    """Encodes a whole sentence into a fixed-length semantic vector."""
    with torch.no_grad():
        embedding = model.encode(sentence, convert_to_tensor=True)
    return embedding

def compare_sentences():
    print("=== Semantic Vector Calculator ===")
    print("Type two sentences to see their semantic vectors and similarity score.\n")
    
    s1 = input("Enter Sentence 1: ")
    s2 = input("Enter Sentence 2: ")
    
    # Generate Vectors
    v1 = get_sentence_vector(s1)
    v2 = get_sentence_vector(s2)
    
    # Calculate Cosine Similarity
    similarity = util.pytorch_cos_sim(v1, v2).item()
    
    print("\n" + "="*60)
    print(f"SENTENCE 1: {s1}")
    print(f"Vector Snippet (First 5 dims): {v1[:5].tolist()}")
    print(f"Vector Dimension: {v1.shape[0]}")
    
    print("-" * 60)
    
    print(f"SENTENCE 2: {s2}")
    print(f"Vector Snippet (First 5 dims): {v2[:5].tolist()}")
    print(f"Vector Dimension: {v2.shape[0]}")
    
    print("="*60)
    print(f"COSINE SIMILARITY (Knob 1 & 2): {similarity:.4f}")
    print("="*60)

if __name__ == "__main__":
    while True:
        compare_sentences()
        cont = input("\nCompare another pair? (y/n): ").lower()
        if cont != 'y':
            break
