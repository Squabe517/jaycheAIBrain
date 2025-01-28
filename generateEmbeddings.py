import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import weaviate


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file and returns it as a string.

    Parameters:
    file_path (str): The path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """

    # 1. Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # 2. Check if the file is a PDF by extension (basic check)
    if not file_path.lower().endswith('.pdf'):
        raise ValueError(f"Unsupported file format: '{file_path}' is not a PDF file.")

    # 3. Initialize an empty string to accumulate text
    extracted_text = ""

    try:
        # 4. Open the PDF file using PdfReader
        reader = PdfReader(file_path)

        # 5. Iterate through each page in the PDF
        for page in reader.pages:
            # page.extract_text() can sometimes return None, so we handle that gracefully
            text = page.extract_text() or ""
            extracted_text += text

    except Exception as e:
        # 6. Catch any other exceptions during PDF reading/parsing
        raise RuntimeError(f"An error occurred while reading the PDF: {str(e)}")

    # 7. Return the combined text from all pages
    return extracted_text


def chunk_text_with_langchain(text, chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ". "]):
    """
    Splits a given string of text into manageable chunks using LangChain's CharacterTextSplitter.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size for each chunk.
        chunk_overlap (int): The number of overlapping characters between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """

    # 1. Handle empty or invalid text input
    if not text:
        return []

    # 2. Validate chunk_size and chunk_overlap
    #    - chunk_size should be greater than zero to avoid an infinite loop.
    #    - chunk_overlap should not exceed chunk_size, otherwise you won't make progress.
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

    # 3. Create a CharacterTextSplitter instance with specified chunk_size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    # 4. Use the splitter to split the text into chunks
    #    This will return a list of strings, each being a chunk of the original text.
    chunks = splitter.split_text(text)

    # 5. Return the list of chunks
    return chunks



def embed_text_with_openai(text, model_name="text-embedding-ada-002"):
    """
    Generates an embedding for the given text using OpenAI's Embedding API.

    Args:
        text (str): The input string to embed.
        model_name (str): The name of the OpenAI model to use for embeddings.

    Returns:
        list: The embedding vector as a list of floats.

    Raises:
        ValueError: If the input text is empty.
        AuthenticationError: If there's an issue with the API key.
        OpenAIError: If any other API error occurs.
    """
    
    # 1. Check if the input text is empty
    if not text.strip():
        raise ValueError("Input text is empty. Please provide a valid string.")

    # 2. Call the OpenAI Embedding API
    response = client.embeddings.create(
        input=text,
        model=model_name
    )
    # 3. Extract the embedding from the API response
    embedding = response.data[0].embedding

    # 4. Return the embedding as a list of floats
    return embedding

def elbow_method(embeddings, max_clusters=20, random_state=42):
    """
    Applies the Elbow Method to find the optimal number of clusters for K-Means.

    Parameters
    ----------
    embeddings : list of lists or numpy.ndarray
        The data points (embeddings) to be clustered. Each sub-list or row in the array 
        represents a single data point's features/coordinates.
    max_clusters : int, optional
        The maximum number of clusters to consider. The search starts from 2 and goes 
        up to this value (default is 20).
    random_state : int, optional
        A seed for reproducibility of results within K-Means (default is 42).

    Returns
    -------
    None
        Displays a plot of the inertia values against the number of clusters. 
        It does not return anything.

    Notes
    -----
    - The "elbow" in the plot is typically the point where the reduction in inertia
      becomes less pronounced, indicating a suitable balance between cluster count
      and separation quality.
    - Adjust `max_clusters` based on your data size and intuition.
    """
    # Convert embeddings to NumPy array if not already
    data = np.array(embeddings, dtype=float)

    # List to store the inertia values (sum of squared distances)
    inertia_values = []

    # Range of cluster counts to try (start from 2 to avoid trivial single-cluster case)
    cluster_range = range(2, max_clusters + 1)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)

    # Plot the inertia for each k
    plt.figure(figsize=(8, 6))
    plt.plot(cluster_range, inertia_values, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.xticks(cluster_range)
    plt.grid(True)
    plt.show()
    


def cluster_embeddings(embeddings, n_clusters=5, visualize=False):
    """
    Clusters a set of numerical embeddings using the KMeans algorithm.

    Parameters
    ----------
    embeddings : list of lists or numpy.ndarray
        A list (or array) of numerical embeddings. Each element should be
        a list or array representing the embedding (e.g., [dim1, dim2, ..., dimN]).
    n_clusters : int, optional
        The number of clusters for the KMeans algorithm (default is 5).
    visualize : bool, optional
        If True, generates a 2D scatter plot of the clustered embeddings
        using PCA for dimensionality reduction. By default, this is False.

    Returns
    -------
    labels : numpy.ndarray
        Cluster labels assigned to each embedding.
    centroids : numpy.ndarray
        Coordinates of the cluster centroids in the original embedding space.

    Raises
    ------
    ValueError
        If `embeddings` is not in a valid format (e.g., empty or not a list/array).
    """

    # --- Input Validation ---
    if not isinstance(embeddings, (list, np.ndarray)):
        raise ValueError("Embeddings must be a list or numpy array.")
    if len(embeddings) == 0:
        raise ValueError("Embeddings list/array is empty.")
    
    # Convert the embeddings to a NumPy array if not already
    embeddings_array = np.array(embeddings, dtype=float)

    # Check dimensions to ensure it can be clustered
    if embeddings_array.ndim != 2:
        raise ValueError("Embeddings must be a 2D array-like structure.")
    
    # --- KMeans Clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings_array)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # --- Optional Visualization ---
    if visualize:
        # If embeddings have more than 2 dimensions, use PCA to reduce to 2
        if embeddings_array.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(embeddings_array)
        else:
            # If the data is already 2D, no need for PCA
            reduced_data = embeddings_array
        
        # Create a scatter plot for each cluster
        plt.figure(figsize=(8, 6))
        for cluster_id in range(n_clusters):
            # Select only the points belonging to the current cluster
            cluster_points = reduced_data[labels == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")
        
        plt.title("KMeans Clustering Visualization (2D)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.show()

    return labels, centroids

def store_in_weaviate(chunks, embeddings, labels, class_name):
    """
    Stores given chunks, their embeddings, and labels into Weaviate for semantic querying.
    
    Parameters
    ----------
    chunks : list of str
        The textual chunks to be stored.
    embeddings : list of list of float
        The vector embeddings corresponding to each chunk.
    labels : list
        Any label or additional categorical information associated with each chunk.
    weaviate_url : str
        The URL endpoint of your Weaviate instance (e.g., "http://localhost:8080").
    class_name : str
        The Weaviate class name under which to store the objects.
        
    Returns
    -------
    None
        The function doesn't return anything. It writes data to your Weaviate instance.
    
    Notes
    -----
    - This function assumes you have an existing schema in Weaviate with a class
      matching `class_name` and properties `text` (string) and `label` (string).
      If the class doesn't exist, you'll need to create it beforehand or modify
      the function to handle schema creation.
    - The embeddings are passed directly as vectors, allowing Weaviate to index
      and perform semantic searches based on these vectors.
    - Ensure you have the `weaviate-client` library installed (pip install weaviate-client).
    """
    
    # Validate input lengths
    if not (len(chunks) == len(embeddings) == len(labels)):
        raise ValueError("chunks, embeddings, and labels must all have the same length.")
    
    # Prepare the data for batching
    
    data_objects = []
    for i, chunk in enumerate(chunks):
        data_object = {
            "properties": {
                "text": chunk,              # The text chunk
                "label": str(labels[i]),
            },
            "vector": embeddings[i],  # Custom embedding
        }
        data_objects.append(data_object)
        
    
    # Initialize the Weaviate client
    client = weaviate.connect_to_local()
    collection = client.collections.get(class_name)
    
    ObjectCount = 0;
    with collection.batch.dynamic() as batch:
        for i, data_object in enumerate(data_objects):
            print("Added Object: " + str(batch.add_object(
                properties = data_object["properties"],
                vector = data_object["vector"]
            )))
        if batch.number_errors:
            print(f"Failed to add {len(batch.failed_objects)} objects.")
            for failed in batch.failed_objects:
                print(f"Failed object UUID: {failed['uuid']}, Error: {failed['result']['errors']}")

