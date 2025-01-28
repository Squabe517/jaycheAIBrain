from createVectorDB import create_vector_db
from generateEmbeddings import *
import json
import numpy as np



def cache_embeddings(embeddings):
    with open("embeddings.json", "w") as f:
        json.dump(embeddings, f)
def load_embeddings(filename):
    with open(filename, "r") as f:
        return json.load(f)

# Create the vector database
create_vector_db("BookChunks")

# Load the book text
bookText = extract_text_from_pdf("beyond.pdf")

# Split the book text into chunks
chunks = chunk_text_with_langchain(bookText)

embeddings = []

# # Generate embeddings for each chunk
# for chunk in chunks:
#     embeddings.append(embed_text_with_openai(chunk))

# # Cache the embeddings
# cache_embeddings(embeddings)

# Load the embeddings
embeddings = load_embeddings("embeddings.json")

# Generate the vector database
clusteredData = cluster_embeddings(embeddings, 10, visualize=True)
# print(f"Chunks: {len(chunks)}\n Embeddings: {len(embeddings)}\nLabels: {len(clusteredData[0])}\nCentroids: {len(clusteredData[1])}")
# # Store in Weaviate
store_in_weaviate(chunks, embeddings, clusteredData[0], "BookChunks")

