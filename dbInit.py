from weaviate import WeaviateClient
import weaviate
from weaviate.connect import ConnectionParams, ProtocolParams
from weaviate.classes.init import AdditionalConfig, Timeout
import weaviate.classes as wvc
import os
from dotenv import load_dotenv



def init_episodic_memory():
    """
    Creates (or updates) a Weaviate schema class for storing information regarding episodic memory
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = weaviate.connect_to_local(
            headers={
                "X-OpenAI-Api-Key": api_key
            }
        )
    
    try:
           
        # Define the schema for the class
        EpisodicMemory = client.collections.create(
            name = "EpisodicMemory",
            description = "A class to store information regarding episodic memory.",
            properties = [
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
        )

        print(EpisodicMemory.config.get(simple=False))
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
    
