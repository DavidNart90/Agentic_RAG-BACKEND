
 CREATE TABLE property_chunks_enhanced (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     property_listing_id UUID NOT NULL,
     content TEXT NOT NULL,

     content_hash VARCHAR(64) GENERATED ALWAYS AS (encode(digest(content, 'sha256'), 'hex')) STORED,
     chunk_index INTEGER NOT NULL DEFAULT 0,
     chunk_type VARCHAR(50) NOT NULL DEFAULT 'general',
    

     token_count INTEGER DEFAULT 0,
     semantic_score DECIMAL(4,3) DEFAULT 0.0 CHECK (semantic_score >= 0 AND semantic_score <= 1),
     content_density DECIMAL(4,3) DEFAULT 0.0 CHECK (content_density >= 0 AND content_density <= 1),
     
  
     embedding vector(1536),
     
 
     metadata JSONB DEFAULT '{}',
     
  
     extracted_entities JSONB DEFAULT '{}',
     entity_types TEXT[] DEFAULT '{}',
     
 
     parent_chunk_id UUID NULL REFERENCES property_chunks_enhanced(id),
     related_chunk_ids UUID[] DEFAULT '{}',
     
 
     created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
     updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    

     CONSTRAINT unique_property_chunk_index UNIQUE (property_listing_id, chunk_index),
     CONSTRAINT valid_chunk_type CHECK (chunk_type IN ('property_core', 'location_context', 'features_amenities', 'financial_analysis', 'agent_info', 'general'))
     

 );
 
 

 CREATE TABLE market_chunks_enhanced (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     market_data_id UUID NOT NULL,
     content TEXT NOT NULL,
 
     content_hash VARCHAR(64) GENERATED ALWAYS AS (encode(digest(content, 'sha256'), 'hex')) STORED,
     chunk_index INTEGER NOT NULL DEFAULT 0,
     chunk_type VARCHAR(50) NOT NULL DEFAULT 'general',
     
 
     token_count INTEGER DEFAULT 0,
     semantic_score DECIMAL(4,3) DEFAULT 0.0,
     content_density DECIMAL(4,3) DEFAULT 0.0,
     
 
     embedding vector(1536),
     
   
     metadata JSONB DEFAULT '{}',
     
 
     market_region VARCHAR(100),
     data_source VARCHAR(50),
     report_date DATE,
     
 
     extracted_entities JSONB DEFAULT '{}',
     entity_types TEXT[] DEFAULT '{}',
     
 
     created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
     updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
 
     CONSTRAINT unique_market_chunk_index UNIQUE (market_data_id, chunk_index),
     CONSTRAINT valid_market_chunk_type CHECK (chunk_type IN ('market_overview', 'price_trends', 'inventory_analysis', 'demographic_data', 'economic_indicators', 'general'))
    
 
 );
 
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_embedding_hnsw 
 ON property_chunks_enhanced USING hnsw (embedding vector_cosine_ops) 
 WITH (m = 16, ef_construction = 64);
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_embedding_hnsw 
 ON market_chunks_enhanced USING hnsw (embedding vector_cosine_ops) 
 WITH (m = 16, ef_construction = 64);

 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_type_score_embedding 
 ON property_chunks_enhanced (chunk_type, semantic_score DESC) 
 WHERE embedding IS NOT NULL;
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_type_score_embedding 
 ON market_chunks_enhanced (chunk_type, semantic_score DESC) 
 WHERE embedding IS NOT NULL;
 
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_content_hash 
 ON property_chunks_enhanced USING hash (content_hash);
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_content_hash 
 ON market_chunks_enhanced USING hash (content_hash);
 
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_metadata_gin 
 ON property_chunks_enhanced USING gin (metadata);
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_entities_gin 
 ON property_chunks_enhanced USING gin (extracted_entities);
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_entity_types_gin 
 ON property_chunks_enhanced USING gin (entity_types);
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_metadata_gin 
 ON market_chunks_enhanced USING gin (metadata);
 
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_listing_id_type 
 ON property_chunks_enhanced (property_listing_id, chunk_type);
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_region_date 
 ON market_chunks_enhanced (market_region, report_date DESC);
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_source_date 
 ON market_chunks_enhanced (data_source, report_date DESC);
 

 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_high_quality 
 ON property_chunks_enhanced (semantic_score DESC, content_density DESC) 
 WHERE semantic_score > 0.7 AND embedding IS NOT NULL;
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_high_quality 
 ON market_chunks_enhanced (semantic_score DESC, content_density DESC) 
 WHERE semantic_score > 0.7 AND embedding IS NOT NULL;
 
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_property_chunks_content_trgm 
 ON property_chunks_enhanced USING gin (content gin_trgm_ops);
 
 CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_chunks_content_trgm 
 ON market_chunks_enhanced USING gin (content gin_trgm_ops);
 
 
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
 
 
 CREATE OR REPLACE FUNCTION monitor_vector_search_performance()
 RETURNS TABLE (
     table_name text,
     total_chunks bigint,
     chunks_with_embeddings bigint,
     avg_embedding_dimension int,
     most_common_chunk_type text,
     avg_semantic_score numeric
 ) AS $$
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
 $$ LANGUAGE plpgsql;
 
 
 CREATE OR REPLACE FUNCTION test_enhanced_schema()
 RETURNS text AS $$
 DECLARE
     result text := '';
     property_count bigint;
     market_count bigint;
 BEGIN
 
     SELECT count(*) INTO property_count FROM property_chunks_enhanced;
     SELECT count(*) INTO market_count FROM market_chunks_enhanced;
     
     result := format('✅ Enhanced schema working! Property chunks: %s, Market chunks: %s', 
                     property_count, market_count);
     
 
     PERFORM '[0.1,0.2,0.3]'::vector(3);
     result := result || E'\n✅ Vector extension working!';
     
 
     PERFORM digest('test', 'sha256');
     result := result || E'\n✅ pgcrypto extension working!';
     
     RETURN result;
 EXCEPTION
     WHEN OTHERS THEN
         RETURN '❌ Error: ' || SQLERRM;
 END;
 $$ LANGUAGE plpgsql;

 
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
     '[0.1, 0.2, 0.3]'::vector, 
     '{"address": "123 Maple St", "city": "Maplewood", "state": "CA", "zip": "90210", "listing_agent": "John Smith"}',
     '{"price": 475000, "bedrooms": 3, "bathrooms": 2.5, "sqft": 2100, "year_built": 2015}',
     ARRAY['property', 'price', 'location', 'specifications']
 );
 
 
 SELECT test_enhanced_schema();


 SELECT 
     chunk_type,
     content_hash,
     semantic_score,
     left(content, 100) || '...' as content_preview
 FROM property_chunks_enhanced 
 LIMIT 3;
 
 
 DO $$
 BEGIN
     RAISE NOTICE '🎉 Enhanced NeonDB schema fixed and deployed successfully!';
     RAISE NOTICE '📊 Run: SELECT test_enhanced_schema(); to verify everything works';
     RAISE NOTICE '🔍 Run: SELECT * FROM monitor_vector_search_performance(); for analytics';
 END $$;

