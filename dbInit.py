from weaviate import WeaviateClient
import weaviate
from weaviate.connect import ConnectionParams, ProtocolParams
from weaviate.classes.init import AdditionalConfig, Timeout
import weaviate.classes as wvc

def init_episodic_memory():
    """
    Creates (or updates) a Weaviate schema class for storing information regarding episodic memory
    """
    
    try:
        client = weaviate.connect_to_local()
        
        # Define the schema for the class
        class_obj = {
            "name": "EpisodicMemory",
            "description": "A class to store information regarding episodic memory.",
            "vectorizer_config": wvc.config.Configure.Vectorizer.none(),  # Specify no vectorizer
            "properties": [
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                    description="The text chunk to be stored and queried semantically.",
                ),
                wvc.config.Property(
                    name="keywords",
                    data_type=wvc.config.DataType.TEXT,
                    description="Any keyword associated with the text.",
                ),  
            ],
        }
        
        client.collections.create(**class_obj)
    finally:
        client.close()
    
def init_semantic_memory():
    """
    Creates (or updates) a Weaviate schema class for storing information regarding semantic memory
    """
    client = weaviate.connect_to_local()
    
    # Define the schema for the class
    class_obj = {
        "name": "SemanticMemory",
        "description": "A class to store information regarding semantic memory.",
        "vectorizer_config": wvc.config.Configure.Vectorizer.none(),  # Specify no vectorizer
        "properties": [
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
    
init_episodic_memory()
    
