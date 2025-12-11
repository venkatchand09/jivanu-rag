#!/bin/bash
set -e

echo "🚀 Jivanu RAG - AWS Deployment Script"
echo "======================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Creating .env file..."
    read -p "Enter OpenAI API Key: " api_key
    cat > .env << ENVEOF
OPENAI_API_KEY=$api_key
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL_NAME=gpt-4o
LLM_TEMPERATURE=0.3
ENVEOF
    echo "✅ .env file created"
fi

# Create data directories with proper permissions
echo "📁 Creating data directories..."
mkdir -p data/{uploads,vector_db,conversations,extracted_assets}
chmod -R 755 data/

# Build and start
echo "🐳 Building Docker image..."
docker-compose down 2>/dev/null || true
docker-compose build --no-cache

echo "🚀 Starting application..."
docker-compose up -d

echo "⏳ Waiting for application to start..."
sleep 10

# Check status
docker-compose ps

echo ""
echo "✅ Deployment complete!"
echo "======================================"
echo "📍 Access your application at:"
echo "   http://$(curl -s ifconfig.me)"
echo ""
echo "📊 Useful commands:"
echo "   docker-compose logs -f      # View logs"
echo "   docker-compose ps           # Check status"
echo "   docker-compose restart      # Restart app"
echo "   docker-compose down         # Stop app"
echo ""
echo "💾 Data is stored in:"
echo "   $(pwd)/data/"
