import weaviate

client = weaviate.Client("http://localhost:8080")

# Define schema
schema = {
    "classes": [
        {
            "class": "Document",
            "description": "A class to store documents and their vector embeddings",
            "vectorizer": "none",  # Use external embeddings
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                },
                {
                    "name": "source",
                    "dataType": ["string"],
                },
            ],
        }
    ]
}

# Add schema to Weaviate
client.schema.create(schema)
