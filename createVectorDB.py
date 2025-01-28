from weaviate import WeaviateClient
import weaviate
from weaviate.connect import ConnectionParams, ProtocolParams
from weaviate.classes.init import AdditionalConfig, Timeout
import weaviate.classes as wvc

def create_vector_db(class_name):
    """
    Creates (or updates) a Weaviate schema class for storing text and labels with external embeddings.
    
    Parameters
    ----------
    weaviate_url : str
        The URL of your Weaviate instance (e.g., "http://localhost:8080").
    class_name : str
        The name of the class to create within the Weaviate schema.
    
    Returns
    -------
    None
        The function doesn't return anything. It creates/updates the Weaviate schema.
    
    Notes
    -----
    - This function assumes you have Weaviate running.
    - It sets the class's `vectorizer` to "none" to use your own external embeddings.
    - The properties added are "text" and "label", both of type "text".
      Adjust or add more properties as needed.
    """
    
    
    client = weaviate.connect_to_local()
    
    # Define the schema for the class
    class_obj = {
        "name": class_name,
        "description": "A class to store textual chunks and labels with custom embeddings.",
        "vectorizer_config": wvc.config.Configure.Vectorizer.none(),  # Specify no vectorizer
        "properties": [
            wvc.config.Property(
                name="conversation",
                data_type=wvc.config.DataType.TEXT,
                description="The text chunk to be stored and queried semantically.",
            ),
            wvc.config.Property(
                name="label",
                data_type=wvc.config.DataType.TEXT,
                description="Any label or category associated with the chunk.",
            ),
            wvc.config.Property(
                name="text",
                data_type=wvc.config.DataType.TEXT,
                description="The text chunk to be stored and queried semantically.",
            ),
            wvc.config.Property(
                name="label",
                data_type=wvc.config.DataType.TEXT,
                description="Any label or category associated with the chunk.",
            ),
        ],
    }
    
    # Check if the class already exists
    existing_classes = []
    for collection in client.collections.list_all():
        print(collection)
        existing_classes.append(collection)

    
    try: 
        if class_name in existing_classes:
            print(f"Class '{class_name}' already exists in the schema. Skipping creation.")
        else:
            # Create the class in Weaviate
            client.collections.create(**class_obj)
            print(f"Class '{class_name}' created successfully in Weaviate.")
    finally:
        client.close()

