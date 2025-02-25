import motor.motor_asyncio


# database = client["tranquara_ai_service"]

async def init_mongo(db_name, db_url, collection):
    """
    Args:
        db_name:
        db_url:
        collection:

    Returns:

    """

    mongo_client = motor.motor_asyncio.AsyncIOMotorClient(db_url)
    mongo_database = mongo_client[db_name]
    mongo_collections = {
        "collections": mongo_database.get_collection(collection)
    }

    return mongo_client, mongo_database, mongo_collections