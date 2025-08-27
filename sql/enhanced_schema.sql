-- Enhanced NeonDB Schema for TrackRealties with Optimal Vector Search
-- This schema builds upon your existing design with performance optimizations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text similarity
CREATE EXTENSION IF NOT EXISTS btree_gin; -- For composite indexes
CREATE EXTENSION IF NOT EXISTS pg_stat_statements; -- For query performance monitoring

-- Enhanced property_chunks table with additional optimizations
DROP TABLE IF EXISTS property_chunks_enhanced CASCADE;
CREATE TABLE property_chunks_enhanced (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_listing_id UUID NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) GENERATED ALWAYS AS (encode(digest(content, 'sha256'), 'hex')) STORED,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_type VARCHAR(50) NOT NULL DEFAULT 'general',
    
    -- Enhanced metadata
    token_count INTEGER DEFAULT 0,
    semantic_score DECIMAL(4,3) DEFAULT 0.0 CHECK (semantic_score >= 0 AND semantic_score <= 1),
    content_density DECIMAL(4,3) DEFAULT 0.0 CHECK (content_density >= 0 AND content_density <= 1),
    
    -- Vector embedding (1536 dimensions for OpenAI text-embedding-3-small)
    embedding vector(1536),
    
    -- JSON metadata with validation
    metadata JSONB DEFAULT '{}',
    
    -- Entity extraction results
    extracted_entities JSONB DEFAULT '{}',
    entity_types TEXT[] DEFAULT '{}',
    
    -- Relationship tracking
    parent_chunk_id UUID NULL REFERENCES property_chunks_enhanced(id),
    related_chunk_ids UUID[] DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT unique_property_chunk_index UNIQUE (property_listing_id, chunk_index),
    CONSTRAINT valid_chunk_type CHECK (chunk_type IN ('property_core', 'location_context', 'features_amenities', 'financial_analysis', 'agent_info', 'general')),
    
    -- Foreign key (assuming you have a property_listings table)
    FOREIGN KEY (property_listing_id) REFERENCES property_listings(id) ON DELETE CASCADE
);

-- Enhanced market_chunks table
DROP TABLE IF EXISTS market_chunks_enhanced CASCADE;
CREATE TABLE market_chunks_enhanced (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_data_id UUID NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) GENERATED ALWAYS AS (encode(digest(content, 'sha256'), 'hex')) STORED,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_type VARCHAR(50) NOT NULL DEFAULT 'general',
    
    -- Enhanced metadata
    token_count INTEGER DEFAULT 0,
    semantic_score DECIMAL(4,3) DEFAULT 0.0,
    content_density DECIMAL(4,3) DEFAULT 0.0,
    
    -- Vector embedding
    embedding vector(1536),
    
    -- JSON metadata
    metadata JSONB DEFAULT '{}',
    
    -- Market-specific fields
    market_region VARCHAR(100),
    data_source VARCHAR(50),
    report_date DATE,
    
    -- Entity extraction
    extracted_entities JSONB DEFAULT '{}',
    entity_types TEXT[] DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT unique_market_chunk_index UNIQUE (market_data_id, chunk_index),
    CONSTRAINT valid_market_chunk_type CHECK (chunk_type IN ('market_overview', 'price_trends', 'inventory_analysis', 'demographic_data', 'economic_indicators', 'general')),
    
    -- Foreign key (assuming you have a market_data table)
    FOREIGN KEY (market_data_id) REFERENCES market_data(id) ON DELETE CASCADE
);

-- Create optimized indexes for vector search performance
-- HNSW indexes for approximate nearest neighbor search (most important for performance)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_embedding_hnsw 
ON property_chunks_enhanced USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_embedding_hnsw 
ON market_chunks_enhanced USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat indexes for different workload patterns
-- CREATE INDEX CONCURRENTLY idx_property_chunks_embedding_ivfflat 
-- ON property_chunks_enhanced USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Composite indexes for filtered vector searches
CREATE INDEX CONCURRENTLY idx_property_chunks_type_score_embedding 
ON property_chunks_enhanced (chunk_type, semantic_score DESC) 
WHERE embedding IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_market_chunks_type_score_embedding 
ON market_chunks_enhanced (chunk_type, semantic_score DESC) 
WHERE embedding IS NOT NULL;

-- Hash indexes for exact content matching (deduplication)
CREATE INDEX CONCURRENTLY idx_property_chunks_content_hash 
ON property_chunks_enhanced USING hash (content_hash);

CREATE INDEX CONCURRENTLY idx_market_chunks_content_hash 
ON market_chunks_enhanced USING hash (content_hash);

-- GIN indexes for metadata and entity searches
CREATE INDEX CONCURRENTLY idx_property_chunks_metadata_gin 
ON property_chunks_enhanced USING gin (metadata);

CREATE INDEX CONCURRENTLY idx_property_chunks_entities_gin 
ON property_chunks_enhanced USING gin (extracted_entities);

CREATE INDEX CONCURRENTLY idx_property_chunks_entity_types_gin 
ON property_chunks_enhanced USING gin (entity_types);

CREATE INDEX CONCURRENTLY idx_market_chunks_metadata_gin 
ON market_chunks_enhanced USING gin (metadata);

-- B-tree indexes for common filtering operations
CREATE INDEX CONCURRENTLY idx_property_chunks_listing_id_type 
ON property_chunks_enhanced (property_listing_id, chunk_type);

CREATE INDEX CONCURRENTLY idx_market_chunks_region_date 
ON market_chunks_enhanced (market_region, report_date DESC);

CREATE INDEX CONCURRENTLY idx_market_chunks_source_date 
ON market_chunks_enhanced (data_source, report_date DESC);

-- Partial indexes for high-quality chunks only
CREATE INDEX CONCURRENTLY idx_property_chunks_high_quality 
ON property_chunks_enhanced (semantic_score DESC, content_density DESC) 
WHERE semantic_score > 0.7 AND embedding IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_market_chunks_high_quality 
ON market_chunks_enhanced (semantic_score DESC, content_density DESC) 
WHERE semantic_score > 0.7 AND embedding IS NOT NULL;

-- Text search indexes using pg_trgm
CREATE INDEX CONCURRENTLY idx_property_chunks_content_trgm 
ON property_chunks_enhanced USING gin (content gin_trgm_ops);

CREATE INDEX CONCURRENTLY idx_market_chunks_content_trgm 
ON market_chunks_enhanced USING gin (content gin_trgm_ops);

-- Create updated_at trigger for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_property_chunks_updated_at 
    BEFORE UPDATE ON property_chunks_enhanced 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_market_chunks_updated_at 
    BEFORE UPDATE ON market_chunks_enhanced 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enhanced similarity search functions with better performance
CREATE OR REPLACE FUNCTION enhanced_property_search(
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.7,
    max_results integer DEFAULT 10,
    chunk_types text[] DEFAULT NULL,
    min_semantic_score float DEFAULT 0.0,
    boost_semantic_score boolean DEFAULT true,
    include_metadata boolean DEFAULT true
)
RETURNS TABLE (
    id UUID,
    property_listing_id UUID,
    content TEXT,
    chunk_type VARCHAR(50),
    similarity_score float,
    semantic_score DECIMAL(4,3),
    combined_score float,
    metadata JSONB,
    entity_types TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pc.id,
        pc.property_listing_id,
        pc.content,
        pc.chunk_type,
        1 - (pc.embedding <=> query_embedding) as similarity_score,
        pc.semantic_score,
        CASE 
            WHEN boost_semantic_score THEN 
                (0.7 * (1 - (pc.embedding <=> query_embedding)) + 0.3 * pc.semantic_score::float)
            ELSE 
                (1 - (pc.embedding <=> query_embedding))
        END as combined_score,
        CASE WHEN include_metadata THEN pc.metadata ELSE '{}'::jsonb END as metadata,
        pc.entity_types
    FROM property_chunks_enhanced pc
    WHERE 
        pc.embedding IS NOT NULL
        AND (1 - (pc.embedding <=> query_embedding)) >= similarity_threshold
        AND pc.semantic_score >= min_semantic_score
        AND (chunk_types IS NULL OR pc.chunk_type = ANY(chunk_types))
    ORDER BY 
        CASE 
            WHEN boost_semantic_score THEN 
                (0.7 * (1 - (pc.embedding <=> query_embedding)) + 0.3 * pc.semantic_score::float)
            ELSE 
                (pc.embedding <=> query_embedding)
        END DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Hybrid search function combining vector and text search
CREATE OR REPLACE FUNCTION hybrid_property_search(
    query_text text,
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.6,
    text_similarity_threshold float DEFAULT 0.3,
    max_results integer DEFAULT 10,
    vector_weight float DEFAULT 0.7,
    text_weight float DEFAULT 0.3
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    chunk_type VARCHAR(50),
    vector_similarity float,
    text_similarity float,
    combined_score float,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pc.id,
        pc.content,
        pc.chunk_type,
        (1 - (pc.embedding <=> query_embedding)) as vector_similarity,
        similarity(pc.content, query_text) as text_similarity,
        (vector_weight * (1 - (pc.embedding <=> query_embedding)) + 
         text_weight * similarity(pc.content, query_text)) as combined_score,
        pc.metadata
    FROM property_chunks_enhanced pc
    WHERE 
        pc.embedding IS NOT NULL
        AND (
            (1 - (pc.embedding <=> query_embedding)) >= similarity_threshold
            OR similarity(pc.content, query_text) >= text_similarity_threshold
        )
    ORDER BY combined_score DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Analytics function to monitor search performance
CREATE OR REPLACE FUNCTION get_search_analytics(
    hours_back integer DEFAULT 24
)
RETURNS TABLE (
    total_searches bigint,
    avg_similarity float,
    most_common_chunk_types text[],
    performance_stats jsonb
) AS $$
BEGIN
    -- This would be implemented based on your logging/analytics needs
    -- For now, return basic stats from the chunks tables
    RETURN QUERY
    SELECT 
        (SELECT count(*) FROM property_chunks_enhanced WHERE created_at > CURRENT_TIMESTAMP - (hours_back || ' hours')::interval)::bigint as total_searches,
        (SELECT avg(semantic_score)::float FROM property_chunks_enhanced WHERE embedding IS NOT NULL) as avg_similarity,
        (SELECT array_agg(DISTINCT chunk_type) FROM property_chunks_enhanced LIMIT 10) as most_common_chunk_types,
        json_build_object(
            'total_property_chunks', (SELECT count(*) FROM property_chunks_enhanced),
            'total_market_chunks', (SELECT count(*) FROM market_chunks_enhanced),
            'avg_tokens_per_chunk', (SELECT avg(token_count) FROM property_chunks_enhanced),
            'chunks_with_embeddings', (SELECT count(*) FROM property_chunks_enhanced WHERE embedding IS NOT NULL)
        )::jsonb as performance_stats;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for frequently accessed search statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS search_performance_stats AS
SELECT 
    chunk_type,
    count(*) as chunk_count,
    avg(semantic_score) as avg_semantic_score,
    avg(content_density) as avg_content_density,
    avg(token_count) as avg_token_count,
    count(*) FILTER (WHERE embedding IS NOT NULL) as chunks_with_embeddings
FROM property_chunks_enhanced
GROUP BY chunk_type
UNION ALL
SELECT 
    chunk_type,
    count(*) as chunk_count,
    avg(semantic_score) as avg_semantic_score,
    avg(content_density) as avg_content_density,
    avg(token_count) as avg_token_count,
    count(*) FILTER (WHERE embedding IS NOT NULL) as chunks_with_embeddings
FROM market_chunks_enhanced
GROUP BY chunk_type;

-- Create unique index on the materialized view
CREATE UNIQUE INDEX ON search_performance_stats (chunk_type);

-- Function to refresh performance stats
CREATE OR REPLACE FUNCTION refresh_search_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY search_performance_stats;
END;
$$ LANGUAGE plpgsql;

-- Set up automatic statistics refresh (run every hour)
-- Note: This requires pg_cron extension or external scheduler
-- SELECT cron.schedule('refresh-search-stats', '0 * * * *', 'SELECT refresh_search_stats();');

-- Optimize PostgreSQL settings for vector search (add to postgresql.conf)
/*
# Vector search optimizations for NeonDB
shared_preload_libraries = 'vector'
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 64MB
maintenance_work_mem = 256MB

# Vector-specific settings
vector.hnsw_ef_search = 40  # Higher values = better recall, slower search
vector.ivfflat_probes = 1   # For IVFFlat indexes

# General performance settings
random_page_cost = 1.1      # SSD optimization
seq_page_cost = 1.0
cpu_tuple_cost = 0.01
cpu_index_tuple_cost = 0.005
cpu_operator_cost = 0.0025

# Query planning
default_statistics_target = 100
effective_io_concurrency = 200

# WAL settings for better write performance
wal_buffers = 16MB
checkpoint_completion_target = 0.7
*/

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON property_chunks_enhanced TO neondb_owner;
GRANT SELECT, INSERT, UPDATE, DELETE ON market_chunks_enhanced TO neondb_owner;
GRANT EXECUTE ON FUNCTION enhanced_property_search TO neondb_owner;
GRANT EXECUTE ON FUNCTION hybrid_property_search TO neondb_owner;
GRANT EXECUTE ON FUNCTION get_search_analytics TO neondb_owner;
GRANT SELECT ON search_performance_stats TO neondb_owner;

-- Sample data insertion with proper formatting
INSERT INTO property_chunks_enhanced (
    property_listing_id, 
    content, 
    chunk_type, 
    token_count, 
    semantic_score, 
    content_density,
    embedding,
    metadata,
    extracted_entities,
    entity_types
) VALUES 
(
    gen_random_uuid(),
    'PROPERTY OVERVIEW: Beautiful 3-bedroom, 2.5-bathroom single-family home in prestigious Maplewood neighborhood. Price: $475,000. Square Footage: 2,100 sq ft. Built in 2015, this home features an open floor plan with hardwood floors throughout.',
    'property_core',
    45,
    0.92,
    0.78,
    '[0.1, 0.2, 0.3]'::vector,  -- Placeholder - replace with actual embedding
    '{"address": "123 Maple St", "city": "Maplewood", "state": "CA", "zip": "90210", "listing_agent": "John Smith"}',
    '{"price": 475000, "bedrooms": 3, "bathrooms": 2.5, "sqft": 2100, "year_built": 2015}',
    ARRAY['property', 'price', 'location', 'specifications']
),
(
    gen_random_uuid(),
    'LOCATION & NEIGHBORHOOD: Located in the heart of Maplewood, this property is within walking distance of top-rated schools including Maplewood Elementary (9/10 rating) and Jefferson High School (8/10 rating). Close to shopping centers, parks, and public transportation.',
    'location_context',
    38,
    0.85,
    0.72,
    '[0.4, 0.5, 0.6]'::vector,  -- Placeholder - replace with actual embedding
    '{"school_district": "Maplewood USD", "walkability_score": 85, "transit_score": 78}',
    '{"schools": ["Maplewood Elementary", "Jefferson High School"], "amenities": ["shopping", "parks", "transit"]}',
    ARRAY['location', 'schools', 'amenities', 'transportation']
);

-- Performance monitoring queries
-- Query to check index usage
CREATE OR REPLACE VIEW index_usage_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan
FROM pg_stat_user_indexes
WHERE tablename IN ('property_chunks_enhanced', 'market_chunks_enhanced')
ORDER BY idx_scan DESC;

-- Query to monitor vector search performance
CREATE OR REPLACE FUNCTION monitor_vector_search_performance()
RETURNS TABLE (
    table_name text,
    total_chunks bigint,
    chunks_with_embeddings bigint,
    avg_embedding_dimension int,
    most_common_chunk_type text,
    avg_semantic_score numeric
) AS $
BEGIN
    RETURN QUERY
    SELECT 
        'property_chunks_enhanced'::text,
        count(*)::bigint,
        count(*) FILTER (WHERE embedding IS NOT NULL)::bigint,
        (SELECT array_length(embedding, 1) FROM property_chunks_enhanced WHERE embedding IS NOT NULL LIMIT 1) as avg_embedding_dimension,
        (SELECT chunk_type FROM property_chunks_enhanced GROUP BY chunk_type ORDER BY count(*) DESC LIMIT 1) as most_common_chunk_type,
        avg(semantic_score) as avg_semantic_score
    FROM property_chunks_enhanced
    
    UNION ALL
    
    SELECT 
        'market_chunks_enhanced'::text,
        count(*)::bigint,
        count(*) FILTER (WHERE embedding IS NOT NULL)::bigint,
        (SELECT array_length(embedding, 1) FROM market_chunks_enhanced WHERE embedding IS NOT NULL LIMIT 1) as avg_embedding_dimension,
        (SELECT chunk_type FROM market_chunks_enhanced GROUP BY chunk_type ORDER BY count(*) DESC LIMIT 1) as most_common_chunk_type,
        avg(semantic_score) as avg_semantic_score
    FROM market_chunks_enhanced;
END;
$ LANGUAGE plpgsql;

-- Maintenance procedures
CREATE OR REPLACE FUNCTION cleanup_orphaned_chunks()
RETURNS integer AS $
DECLARE
    deleted_count integer;
BEGIN
    -- Clean up property chunks without valid embeddings or parent records
    DELETE FROM property_chunks_enhanced 
    WHERE embedding IS NULL 
    AND created_at < CURRENT_TIMESTAMP - INTERVAL '7 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log the cleanup
    INSERT INTO maintenance_log (operation, affected_rows, performed_at)
    VALUES ('cleanup_orphaned_chunks', deleted_count, CURRENT_TIMESTAMP);
    
    RETURN deleted_count;
END;
$ LANGUAGE plpgsql;

-- Create maintenance log table
CREATE TABLE IF NOT EXISTS maintenance_log (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(100) NOT NULL,
    affected_rows INTEGER DEFAULT 0,
    performed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    details JSONB DEFAULT '{}'
);

-- Function to analyze search query patterns
CREATE OR REPLACE FUNCTION analyze_search_patterns(
    days_back integer DEFAULT 7
)
RETURNS TABLE (
    pattern_type text,
    frequency bigint,
    avg_similarity float,
    sample_query text
) AS $
BEGIN
    -- This would analyze actual search logs
    -- For now, return sample analysis based on chunk data
    RETURN QUERY
    SELECT 
        pc.chunk_type::text as pattern_type,
        count(*)::bigint as frequency,
        avg(pc.semantic_score)::float as avg_similarity,
        (array_agg(pc.content))[1] as sample_query
    FROM property_chunks_enhanced pc
    WHERE pc.created_at > CURRENT_TIMESTAMP - (days_back || ' days')::interval
    GROUP BY pc.chunk_type
    ORDER BY frequency DESC;
END;
$ LANGUAGE plpgsql;

-- Advanced search function with query expansion
CREATE OR REPLACE FUNCTION search_with_query_expansion(
    original_query text,
    query_embedding vector(1536),
    expand_synonyms boolean DEFAULT true,
    max_results integer DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    chunk_type VARCHAR(50),
    similarity_score float,
    match_type text,
    metadata JSONB
) AS $
DECLARE
    expanded_terms text[];
    similarity_threshold float := 0.6;
BEGIN
    -- Simple query expansion (in production, use more sophisticated NLP)
    IF expand_synonyms THEN
        expanded_terms := ARRAY[
            original_query,
            replace(original_query, 'house', 'home'),
            replace(original_query, 'home', 'house'),
            replace(original_query, 'bedroom', 'bed'),
            replace(original_query, 'bathroom', 'bath')
        ];
    ELSE
        expanded_terms := ARRAY[original_query];
    END IF;
    
    RETURN QUERY
    SELECT 
        pc.id,
        pc.content,
        pc.chunk_type,
        (1 - (pc.embedding <=> query_embedding)) as similarity_score,
        CASE 
            WHEN (1 - (pc.embedding <=> query_embedding)) > 0.8 THEN 'high_similarity'
            WHEN pc.content ~* ANY(expanded_terms) THEN 'text_match'
            ELSE 'semantic_match'
        END as match_type,
        pc.metadata
    FROM property_chunks_enhanced pc
    WHERE 
        pc.embedding IS NOT NULL
        AND (
            (1 - (pc.embedding <=> query_embedding)) >= similarity_threshold
            OR pc.content ~* ANY(expanded_terms)
        )
    ORDER BY 
        (1 - (pc.embedding <=> query_embedding)) DESC,
        pc.semantic_score DESC
    LIMIT max_results;
END;
$ LANGUAGE plpgsql;

-- Real-time search analytics view
CREATE OR REPLACE VIEW real_time_search_analytics AS
SELECT 
    date_trunc('hour', created_at) as time_bucket,
    chunk_type,
    count(*) as chunks_created,
    avg(semantic_score) as avg_quality,
    count(*) FILTER (WHERE embedding IS NOT NULL) as embedded_chunks,
    avg(token_count) as avg_tokens
FROM property_chunks_enhanced
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY date_trunc('hour', created_at), chunk_type
ORDER BY time_bucket DESC, avg_quality DESC;

-- Backup and restore procedures
CREATE OR REPLACE FUNCTION backup_embeddings_metadata()
RETURNS text AS $
DECLARE
    backup_file text;
    record_count integer;
BEGIN
    backup_file := 'embeddings_backup_' || to_char(CURRENT_TIMESTAMP, 'YYYY_MM_DD_HH24_MI_SS');
    
    -- Create backup of metadata (embeddings are too large for JSON export)
    COPY (
        SELECT 
            id,
            property_listing_id,
            content_hash,
            chunk_type,
            semantic_score,
            metadata,
            entity_types,
            created_at
        FROM property_chunks_enhanced
    ) TO PROGRAM 'gzip > /tmp/' || backup_file || '_property.csv.gz' WITH CSV HEADER;
    
    GET DIAGNOSTICS record_count = ROW_COUNT;
    
    INSERT INTO maintenance_log (operation, affected_rows, details)
    VALUES ('backup_embeddings', record_count, 
            json_build_object('backup_file', backup_file, 'timestamp', CURRENT_TIMESTAMP));
    
    RETURN backup_file;
END;
$ LANGUAGE plpgsql;

-- Performance tuning recommendations function
CREATE OR REPLACE FUNCTION get_performance_recommendations()
RETURNS TABLE (
    recommendation_type text,
    description text,
    priority text,
    impact text
) AS $
BEGIN
    RETURN QUERY
    SELECT 
        'Index Usage'::text,
        'HNSW index is ' || 
        CASE 
            WHEN (SELECT idx_scan FROM pg_stat_user_indexes WHERE indexname LIKE '%embedding_hnsw%') > 1000 
            THEN 'performing well with ' || (SELECT idx_scan FROM pg_stat_user_indexes WHERE indexname LIKE '%embedding_hnsw%')::text || ' scans'
            ELSE 'underutilized - consider query optimization'
        END,
        'High'::text,
        'Search Performance'::text
    
    UNION ALL
    
    SELECT 
        'Data Quality'::text,
        'Average semantic score: ' || round(avg(semantic_score)::numeric, 3)::text || 
        ' (target: >0.7)',
        CASE WHEN avg(semantic_score) > 0.7 THEN 'Low' ELSE 'High' END,
        'Search Relevance'::text
    FROM property_chunks_enhanced
    
    UNION ALL
    
    SELECT 
        'Embedding Coverage'::text,
        round((count(*) FILTER (WHERE embedding IS NOT NULL) * 100.0 / count(*))::numeric, 1)::text || 
        '% of chunks have embeddings',
        CASE 
            WHEN (count(*) FILTER (WHERE embedding IS NOT NULL) * 100.0 / count(*)) > 95 THEN 'Low'
            ELSE 'Medium'
        END,
        'Search Coverage'::text
    FROM property_chunks_enhanced;
END;
$ LANGUAGE plpgsql;

-- Final setup verification
DO $
BEGIN
    -- Verify extensions are loaded
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'Vector extension not found. Please install pgvector.';
    END IF;
    
    -- Verify indexes are created
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname LIKE '%embedding_hnsw%') THEN
        RAISE WARNING 'HNSW indexes may not be created yet. Check index creation status.';
    END IF;
    
    RAISE NOTICE 'Enhanced NeonDB schema setup completed successfully!';
    RAISE NOTICE 'Use SELECT * FROM get_performance_recommendations(); to check optimization status.';
END $;