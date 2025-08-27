-- TrackRealties AI Platform Database Schema
-- PostgreSQL with pgvector extension

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing tables if they exist (for development)
DROP TABLE IF EXISTS conversation_messages CASCADE;
DROP TABLE IF EXISTS user_sessions CASCADE;
DROP TABLE IF EXISTS property_chunks CASCADE;
DROP TABLE IF EXISTS market_chunks CASCADE;
DROP TABLE IF EXISTS property_listings CASCADE;
DROP TABLE IF EXISTS market_data CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- Documents table for storing source documents
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    source VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    document_type VARCHAR(50) DEFAULT 'general',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market data table with embeddings
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_id VARCHAR(100) NOT NULL,
    region_name VARCHAR(200) NOT NULL,
    region_type VARCHAR(20) NOT NULL CHECK (region_type IN ('metro', 'county', 'city', 'zip', 'neighborhood')),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    duration VARCHAR(50) NOT NULL,
    
    -- Geographic coordinates
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    
    -- Market metrics (stored as JSONB for flexibility)
    metrics JSONB NOT NULL DEFAULT '{}',
    
    -- Data quality and source tracking
    data_quality_score DECIMAL(3, 2) CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    sample_size INTEGER,
    source VARCHAR(200) NOT NULL,
    source_url TEXT,
    source_metadata JSONB DEFAULT '{}',
    
    -- Validation tracking
    is_validated BOOLEAN DEFAULT FALSE,
    validation_errors TEXT[],
    validation_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Vector embedding for semantic search
    embedding vector(1536), -- OpenAI text-embedding-3-small dimensions
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_period CHECK (period_end > period_start)
);

-- Property listings table with embeddings
CREATE TABLE property_listings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id VARCHAR(200) UNIQUE NOT NULL,
    formatted_address TEXT NOT NULL,
    
    -- Address components
    address_line1 VARCHAR(200),
    address_line2 VARCHAR(200),
    city VARCHAR(100) NOT NULL,
    state VARCHAR(10) NOT NULL,
    zip_code VARCHAR(20),
    county VARCHAR(100),
    
    -- Geographic coordinates
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    
    -- Property details
    property_type VARCHAR(50) NOT NULL,
    bedrooms INTEGER,
    bathrooms DECIMAL(3, 1),
    square_footage INTEGER,
    lot_size INTEGER,
    year_built INTEGER,
    
    -- Listing information
    status VARCHAR(50) NOT NULL,
    price DECIMAL(12, 2) NOT NULL,
    listing_type VARCHAR(50),
    listed_date TIMESTAMP WITH TIME ZONE,
    removed_date TIMESTAMP WITH TIME ZONE,
    days_on_market INTEGER,
    
    -- MLS information
    mls_name VARCHAR(100),
    mls_number VARCHAR(100),
    
    -- Agent and office information (stored as JSONB)
    listing_agent JSONB,
    listing_office JSONB,
    
    -- Property history
    history JSONB DEFAULT '{}',
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}',
    
    -- Data source tracking
    source VARCHAR(200) NOT NULL,
    source_url TEXT,
    source_metadata JSONB DEFAULT '{}',
    
    -- Validation tracking
    is_validated BOOLEAN DEFAULT FALSE,
    validation_errors TEXT[],
    validation_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Vector embedding for semantic search
    embedding vector(1536), -- OpenAI text-embedding-3-small dimensions
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT positive_price CHECK (price > 0),
    CONSTRAINT valid_coordinates CHECK (
        (latitude IS NULL AND longitude IS NULL) OR 
        (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180)
    )
);

-- Market data chunks for RAG
CREATE TABLE market_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_data_id UUID NOT NULL REFERENCES market_data(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(market_data_id, chunk_index)
);

-- Property data chunks for RAG
CREATE TABLE property_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_listing_id UUID NOT NULL REFERENCES property_listings(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(property_listing_id, chunk_index)
);

-- User sessions for conversation management
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(200),
    user_role VARCHAR(50) NOT NULL CHECK (user_role IN ('investor', 'developer', 'buyer', 'agent', 'general')),
    session_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '1 hour'),
    is_active BOOLEAN DEFAULT TRUE
);

-- Conversation messages
CREATE TABLE conversation_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES user_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    
    -- Tool usage tracking
    tools_used JSONB DEFAULT '[]',
    
    -- Validation and quality metrics
    validation_result JSONB,
    confidence_score DECIMAL(3, 2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Performance metrics
    processing_time_ms INTEGER,
    token_count INTEGER,
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance optimization

-- Market data indexes
CREATE INDEX idx_market_data_region ON market_data(region_id, region_type);
CREATE INDEX idx_market_data_period ON market_data(period_start, period_end);
CREATE INDEX idx_market_data_location ON market_data(latitude, longitude) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
CREATE INDEX idx_market_data_embedding ON market_data USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_market_data_source ON market_data(source);
CREATE INDEX idx_market_data_validated ON market_data(is_validated);

-- Property listings indexes
CREATE INDEX idx_property_listings_location ON property_listings(city, state, zip_code);
CREATE INDEX idx_property_listings_price ON property_listings(price);
CREATE INDEX idx_property_listings_type ON property_listings(property_type, status);
CREATE INDEX idx_property_listings_coordinates ON property_listings(latitude, longitude) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
CREATE INDEX idx_property_listings_embedding ON property_listings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_property_listings_listed_date ON property_listings(listed_date);
CREATE INDEX idx_property_listings_source ON property_listings(source);

-- Chunk indexes for RAG
CREATE INDEX idx_market_chunks_embedding ON market_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_property_chunks_embedding ON property_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Session and message indexes
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_role ON user_sessions(user_role);
CREATE INDEX idx_user_sessions_active ON user_sessions(is_active, expires_at);
CREATE INDEX idx_conversation_messages_session ON conversation_messages(session_id, created_at);
CREATE INDEX idx_conversation_messages_role ON conversation_messages(role);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_market_data_updated_at BEFORE UPDATE ON market_data FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_property_listings_updated_at BEFORE UPDATE ON property_listings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_sessions_updated_at BEFORE UPDATE ON user_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions WHERE expires_at < NOW() AND is_active = FALSE;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function for similarity search with market data
CREATE OR REPLACE FUNCTION search_market_data_by_embedding(
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.7,
    max_results integer DEFAULT 10,
    region_filter text DEFAULT NULL,
    region_type_filter text DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    region_id VARCHAR(100),
    region_name VARCHAR(200),
    region_type VARCHAR(20),
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    metrics JSONB,
    similarity_score float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        md.id,
        md.region_id,
        md.region_name,
        md.region_type,
        md.period_start,
        md.period_end,
        md.metrics,
        1 - (md.embedding <=> query_embedding) as similarity_score
    FROM market_data md
    WHERE 
        md.embedding IS NOT NULL
        AND (region_filter IS NULL OR md.region_id = region_filter)
        AND (region_type_filter IS NULL OR md.region_type = region_type_filter)
        AND (1 - (md.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY md.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function for similarity search with property listings
CREATE OR REPLACE FUNCTION search_properties_by_embedding(
    query_embedding vector(1536),
    similarity_threshold float DEFAULT 0.7,
    max_results integer DEFAULT 10,
    city_filter text DEFAULT NULL,
    state_filter text DEFAULT NULL,
    min_price decimal DEFAULT NULL,
    max_price decimal DEFAULT NULL,
    property_type_filter text DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    property_id VARCHAR(200),
    formatted_address TEXT,
    city VARCHAR(100),
    state VARCHAR(10),
    property_type VARCHAR(50),
    price DECIMAL(12, 2),
    bedrooms INTEGER,
    bathrooms DECIMAL(3, 1),
    square_footage INTEGER,
    similarity_score float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pl.id,
        pl.property_id,
        pl.formatted_address,
        pl.city,
        pl.state,
        pl.property_type,
        pl.price,
        pl.bedrooms,
        pl.bathrooms,
        pl.square_footage,
        1 - (pl.embedding <=> query_embedding) as similarity_score
    FROM property_listings pl
    WHERE 
        pl.embedding IS NOT NULL
        AND (city_filter IS NULL OR pl.city = city_filter)
        AND (state_filter IS NULL OR pl.state = state_filter)
        AND (min_price IS NULL OR pl.price >= min_price)
        AND (max_price IS NULL OR pl.price <= max_price)
        AND (property_type_filter IS NULL OR pl.property_type = property_type_filter)
        AND (1 - (pl.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY pl.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Insert some sample data for testing
INSERT INTO documents (title, source, content, document_type) VALUES
('TrackRealties Platform Overview', 'internal', 'TrackRealties is an AI-powered real estate intelligence platform that provides role-specific insights for investors, developers, buyers, and agents.', 'documentation'),
('Real Estate Market Analysis Guide', 'internal', 'This guide covers key metrics for analyzing real estate markets including median prices, inventory levels, days on market, and market trends.', 'documentation');

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO neondb_owner;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO neondb_owner;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO neondb_owner;
