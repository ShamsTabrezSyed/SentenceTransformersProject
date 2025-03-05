# Sentence Transformers - Cosine Similarity Example

## Overview
Welcome to the **Sentence Transformers - Cosine Similarity** example! In this project, you'll learn how to use the **`sentence-transformers`** library to generate sentence embeddings and compute the cosine similarity between them. The core idea of the script is simple: it takes a list of sentences, encodes them into numerical vectors (embeddings) using a pre-trained model, and then calculates how similar those sentences are to each other by measuring the cosine similarity between their embeddings.

The script uses a very efficient and lightweight transformer model, specifically `all-MiniLM-L6-v2`, to generate these embeddings. You will also see how we use **GPU acceleration** (if available) for faster computations, making it easy to work with a larger number of sentences.

## Why This Approach?
Here are some reasons why this approach is a great choice:

- **Compact and Efficient Model:** 
  - The `all-MiniLM-L6-v2` model strikes a balance between performance and speed. It is a lightweight model that delivers high-quality sentence embeddings, while being computationally efficient.
  - This model is perfect when you need to process a good amount of data without straining your system’s resources.

- **GPU Utilization:**
  - By default, the script checks if a **GPU** is available (via **CUDA**) and automatically runs on it for **faster processing**. If no GPU is available, it runs on the CPU without any issues.
  
- **Configurable Embeddings:**
  - The model’s parameters are customizable. For example, you can adjust batch sizes, normalize embeddings, and control the progress bar display to better suit your needs. This allows for optimal similarity comparisons, even with large datasets.

## Alternatives Considered
While **sentence-transformers** is a great choice for this task, there are a few other approaches and models worth considering. Let’s take a quick look at some of the alternatives:

1. **BERT-Based Models:**
   - Models like `bert-base-uncased` are known for their accuracy and are widely used in NLP tasks. However, they are much slower to compute, require more memory, and are generally overkill for similarity tasks where a lightweight model like `all-MiniLM-L6-v2` would suffice.
   - **Pros:** Better accuracy, more powerful representations.
   - **Cons:** Slower, higher memory usage.

2. **Traditional Models (TF-IDF, Word2Vec, FastText):**
   - Older methods like **TF-IDF** or **Word2Vec** are popular, but they fall short when it comes to capturing deep semantic meanings of sentences. They are useful for simpler tasks but may not perform as well for nuanced comparisons.
   - **Pros:** Simple, fast, and works well for basic tasks.
   - **Cons:** Lacks deep semantic understanding, struggles with contextual meaning.

3. **Other Sentence Transformers (e.g., `all-mpnet-base-v2`):**
   - Larger models like `all-mpnet-base-v2` can provide even better sentence embeddings, but they are computationally heavier and may be unnecessary for smaller-scale tasks or real-time applications.
   - **Pros:** Higher quality embeddings.
   - **Cons:** Slower, higher resource requirements.

## Installation

Before running the script, make sure you have **Python** installed on your system. You'll also need to install the dependencies. Here’s how you can set it all up:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-repository/sentence-transformer-example.git
   cd sentence-transformer-example

