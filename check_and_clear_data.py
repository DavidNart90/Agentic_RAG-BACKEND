"""
Script to check and clear data from the database before ingestion.
"""

import asyncio
import asyncpg
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def check_and_clear_data():
    """Check if there's data in the database and clear it if needed."""
    # Get database URL from environment variable
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        return False
    
    try:
        # Connect to the database
        logger.info(f"Connecting to database...")
        conn = await asyncpg.connect(database_url)
        
        # Check if tables exist
        logger.info("Checking if tables exist...")
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND 
              table_name IN ('market_data', 'property_listings', 'market_chunks', 'property_chunks','market_chunks_enhanced','property_chunks_enhanced','conversation_messages', 'user_sessions');
        """
        tables = await conn.fetch(tables_query)
        
        if not tables:
            logger.info("No relevant tables found in the database")
            await conn.close()
            return False
        
        # Check if there's data in the tables
        data_exists = False
        for record in tables:
            table_name = record['table_name']
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            count = await conn.fetchval(count_query)
            logger.info(f"Table {table_name} has {count} records")
            if count > 0:
                data_exists = True
        
        if not data_exists:
            logger.info("No data found in the tables")
            await conn.close()
            return False
        
        # Ask for confirmation before deleting
        response = input("Data found in the database. Do you want to delete it? (y/n): ")
        if response.lower() != 'y':
            logger.info("Data deletion cancelled")
            await conn.close()
            return False
        
        # Delete data from the tables
        logger.info("Deleting data from tables...")
        for record in tables:
            table_name = record['table_name']
            delete_query = f"DELETE FROM {table_name}"
            await conn.execute(delete_query)
            logger.info(f"Deleted all records from {table_name}")
        
        logger.info("All data deleted successfully")
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error checking/clearing data: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(check_and_clear_data())