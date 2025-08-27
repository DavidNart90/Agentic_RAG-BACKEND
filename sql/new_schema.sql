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
    location VARCHAR(200) NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    median_price DECIMAL(12, 2),
    inventory_count INTEGER,
    sales_volume INTEGER,
    new_listings INTEGER,
    days_on_market INTEGER,
    months_supply DECIMAL(10, 2),
    price_per_sqft DECIMAL(10, 2),
    source VARCHAR(200),
    region_type VARCHAR(50),
    region_id VARCHAR(100),
    duration VARCHAR(50),
    last_updated TIMESTAMP WITH TIME ZONE,
    city VARCHAR(100),
    state VARCHAR(50),
    
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Property listings table with embeddings
CREATE TABLE property_listings (
    id VARCHAR(200) PRIMARY KEY,
    formatted_address TEXT NOT NULL,
    address_line1 VARCHAR(200),
    address_line2 VARCHAR(200),
    city VARCHAR(100) NOT NULL,
    state VARCHAR(10) NOT NULL,
    zip_code VARCHAR(20),
    county VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    property_type VARCHAR(50) NOT NULL,
    bedrooms INTEGER,
    bathrooms DECIMAL(3, 1),
    square_footage INTEGER,
    lot_size INTEGER,
    year_built INTEGER,
    status VARCHAR(50) NOT NULL,
    price DECIMAL(12, 2) NOT NULL,
    listing_type VARCHAR(50),
    listed_date TIMESTAMP WITH TIME ZONE,
    removed_date TIMESTAMP WITH TIME ZONE,
    created_date TIMESTAMP WITH TIME ZONE,
    last_seen_date TIMESTAMP WITH TIME ZONE,
    days_on_market INTEGER,
    mls_name VARCHAR(100),
    mls_number VARCHAR(100),
    listing_agent JSONB,
    listing_office JSONB,
    history JSONB DEFAULT '{}',
    
    
    -- Timestamps
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market data chunks for RAG
CREATE TABLE market_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_data_id UUID NOT NULL REFERENCES market_data(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    embedding vector(1536), -- OpenAI text-embedding-3-small dimensions
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(market_data_id, chunk_index)
);

-- Property data chunks for RAG
CREATE TABLE property_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_listing_id VARCHAR(200) NOT NULL REFERENCES property_listings(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    embedding vector(1536), -- OpenAI text-embedding-3-small dimensions
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
CREATE INDEX idx_market_data_location ON market_data(location);
CREATE INDEX idx_market_data_date ON market_data(date);

-- Property listings indexes
CREATE INDEX idx_property_listings_location ON property_listings(city, state, zip_code);
CREATE INDEX idx_property_listings_price ON property_listings(price);
CREATE INDEX idx_property_listings_type ON property_listings(property_type, status);

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
