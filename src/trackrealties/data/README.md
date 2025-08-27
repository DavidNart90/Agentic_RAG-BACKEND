# Data Module

The `data` module provides utilities for data processing, transformation, and storage in the TrackRealties AI Platform.

## Components

### Transformation (`transformation.py`)

The transformation module provides utilities for transforming data between different formats and models, including field mapping, type conversion, and data normalization.

#### Key Classes

- **`DataTransformer`**: Transforms raw data between different formats and structures.
  - `transform_property_listing()`: Transforms property listing data to a standardized format.
  - `transform_market_data()`: Transforms market data to a standardized format.

- **`ModelTransformer`**: Transforms raw data into Pydantic models with validation.
  - `transform_to_property_listing()`: Transforms raw property data into a PropertyListing model.
  - `transform_to_market_data()`: Transforms raw market data into a MarketDataPoint model.

### Chunking

The chunking module provides utilities for breaking down data into semantically meaningful chunks for embedding and retrieval.

- **`JSONChunker`**: Chunks JSON data semantically based on its structure.
- **`PropertyListingChunker`**: Specialized chunker for property listings.
- **`MarketDataChunker`**: Specialized chunker for market data.
- **`GenericChunker`**: Fallback chunker for unknown data types.

### Embedding

The embedding module provides utilities for generating vector embeddings for chunks.

- **`Embedder`**: Abstract base class for embedding generators.
- **`OpenAIEmbedder`**: Generates embeddings using OpenAI's API.

### Database Integration

The database integration module provides utilities for saving data to PostgreSQL and Neo4j.

- **`DatabaseIntegration`**: Handles saving data to PostgreSQL.
- **`GraphBuilder`**: Handles saving data to Neo4j.

### Error Handling

The error handling module provides utilities for handling and reporting errors.

- **`DatabaseErrorHandler`**: Handles database errors.
- **`GraphErrorHandler`**: Handles graph database errors.

## Usage Examples

### Transforming Property Listing Data

```python
from trackrealties.data.transformation import DataTransformer, ModelTransformer

# Create transformers
data_transformer = DataTransformer()
model_transformer = ModelTransformer()

# Transform raw property data to standardized format
transformed_data = await data_transformer.transform_property_listing(raw_property_data)

# Transform raw property data to PropertyListing model
property_model, errors = await model_transformer.transform_to_property_listing(raw_property_data)
if not errors:
    # Use the property model
    print(property_model.get_summary())
else:
    # Handle validation errors
    print(f"Validation errors: {errors}")
```

### Transforming Market Data

```python
from trackrealties.data.transformation import DataTransformer, ModelTransformer

# Create transformers
data_transformer = DataTransformer()
model_transformer = ModelTransformer()

# Transform raw market data to standardized format
transformed_data = await data_transformer.transform_market_data(raw_market_data)

# Transform raw market data to MarketDataPoint model
market_model, errors = await model_transformer.transform_to_market_data(raw_market_data)
if not errors:
    # Use the market model
    print(f"Market health score: {market_model.calculate_market_health_score()}")
else:
    # Handle validation errors
    print(f"Validation errors: {errors}")
```