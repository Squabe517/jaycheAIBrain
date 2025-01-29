import weaviate
from generateEmbeddings import embed_text_with_openai
from weaviate.classes.query import MetadataQuery, Filter


def readAllDB(class_name):       
    
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get(class_name)

        for item in collection.iterator():
            print(item.uuid, item.properties, item.vector)
    finally:
        client.close()
        
        
def readUUID(class_name, uuid):       
    
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get(class_name)
        
        data_object = collection.query.fetch_object_by_id(uuid, include_vector=True)
        print(data_object)
        
    finally:
        client.close()
# readDB("BookChunks")

# readUUID("BookChunks", "890f2f64-f8e2-4159-a31f-8aecb1219326")

def querySimilarity(query, class_name):
    queryVector = embed_text_with_openai(query);
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get(class_name)

        response = collection.query.near_vector(
            near_vector=queryVector, # your query vector goes here
            limit=5,
            certainty=0,
            return_metadata=MetadataQuery(distance=True)
        )
        return response
            
    finally:
        client.close()
        
def deleteCollection(class_name):
    try:
        client = weaviate.connect_to_local()
        client.collections.delete(class_name)
    finally:
        client.close()
        

# Query for similar results
# queryResult = querySimilarity("location", "EpisodicMemory")
# for i in queryResult.objects:
#     print(i.properties)
#     print(i.metadata.distance)

# readAllDB("EpisodicMemory")
    
querySimilarity("Gaige", "EpisodicMemory")


# deleteCollection("EpisodicMemory")