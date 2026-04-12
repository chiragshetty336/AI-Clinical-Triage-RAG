#!/bin/bash

###############################################################################
# MEDICAL TRIAGE RAG SYSTEM - QUICK START SCRIPT
# Initializes, configures, and tests your RAG comparison system
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="."
RAG_MODULE_PATH="${PROJECT_ROOT}/improved_llm_compare.py"
TEST_MODULE_PATH="${PROJECT_ROOT}/test_rag_evaluation.py"
DATASET_PATH="${PROJECT_ROOT}/data/medical_triage_dataset.json"
API_ROUTES_PATH="${PROJECT_ROOT}/api/compare_routes.py"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     MEDICAL TRIAGE RAG SYSTEM - QUICK START SETUP             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

###############################################################################
# STEP 1: Verify Prerequisites
###############################################################################

echo -e "\n${YELLOW}[1/6] Checking prerequisites...${NC}"

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $2 not found. Please install it.${NC}"
        return 1
    else
        echo -e "${GREEN}✓ $2 found${NC}"
        return 0
    fi
}

check_command "python" "Python 3"
check_command "pip" "pip"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

###############################################################################
# STEP 2: Verify Files
###############################################################################

echo -e "\n${YELLOW}[2/6] Verifying RAG files...${NC}"

if [ ! -f "$RAG_MODULE_PATH" ]; then
    echo -e "${RED}❌ improved_llm_compare.py not found${NC}"
    echo -e "   Expected at: $RAG_MODULE_PATH"
    exit 1
else
    echo -e "${GREEN}✓ improved_llm_compare.py found${NC}"
fi

if [ ! -f "$TEST_MODULE_PATH" ]; then
    echo -e "${RED}❌ test_rag_evaluation.py not found${NC}"
    exit 1
else
    echo -e "${GREEN}✓ test_rag_evaluation.py found${NC}"
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}❌ medical_triage_dataset.json not found${NC}"
    exit 1
else
    echo -e "${GREEN}✓ medical_triage_dataset.json found${NC}"
fi

###############################################################################
# STEP 3: Check Environment Variables
###############################################################################

echo -e "\n${YELLOW}[3/6] Checking environment configuration...${NC}"

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file not found. Creating template...${NC}"
    cat > .env << 'EOF'
# Ollama Configuration (Mistral)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here

# RAG Configuration
INDEX_PATH=./data/faiss_index
CACHE_PATH=./data/embeddings_cache
MODEL_NAME=all-MiniLM-L6-v2
EOF
    echo -e "${GREEN}✓ Created .env template${NC}"
    echo -e "${YELLOW}  Please update with your API keys${NC}"
else
    echo -e "${GREEN}✓ .env file found${NC}"
    
    # Check for key values
    if grep -q "GROQ_API_KEY=your_groq_api_key_here" .env; then
        echo -e "${YELLOW}⚠ GROQ_API_KEY not configured${NC}"
    else
        echo -e "${GREEN}✓ GROQ_API_KEY configured${NC}"
    fi
fi

###############################################################################
# STEP 4: Check Services
###############################################################################

echo -e "\n${YELLOW}[4/6] Checking required services...${NC}"

# Check Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
    
    # Check if Mistral is pulled
    if curl -s http://localhost:11434/api/tags | grep -q "mistral"; then
        echo -e "${GREEN}  ✓ Mistral model available${NC}"
    else
        echo -e "${YELLOW}⚠ Mistral not found. Run: ollama pull mistral:7b${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Ollama not running. Start with: ollama serve${NC}"
    echo -e "  Mistral queries will fail until Ollama is started"
fi

###############################################################################
# STEP 5: Check RAG Index
###############################################################################

echo -e "\n${YELLOW}[5/6] Checking RAG index...${NC}"

if [ -d "data/faiss_index" ] && [ -f "data/embeddings_cache/metadata.pkl" ]; then
    echo -e "${GREEN}✓ RAG index found${NC}"
    
    # Count documents in metadata
    if python -c "import pickle; m=pickle.load(open('data/embeddings_cache/metadata.pkl','rb')); print(f'{len(m.get(\"chunks\", []))} chunks')" 2>/dev/null; then
        echo -e "${GREEN}✓ RAG documents loaded${NC}"
    fi
else
    echo -e "${YELLOW}⚠ RAG index not found${NC}"
    echo -e "  You must run PDF ingestion first"
    echo -e "  Check your main.py or ingestion scripts"
fi

###############################################################################
# STEP 6: Test System
###############################################################################

echo -e "\n${YELLOW}[6/6] Testing system components...${NC}"

# Test imports
echo -e "\n  Testing Python imports..."

python << 'PYEOF'
import sys
try:
    from sentence_transformers import SentenceTransformer
    print("  ✓ sentence-transformers available")
except:
    print("  ❌ sentence-transformers not found (pip install sentence-transformers)")
    sys.exit(1)

try:
    import faiss
    print("  ✓ faiss available")
except:
    print("  ❌ faiss not found (pip install faiss-cpu)")
    sys.exit(1)

try:
    from groq import Groq
    print("  ✓ groq available")
except:
    print("  ⚠ groq not found (pip install groq)")

try:
    from fastapi import FastAPI
    print("  ✓ fastapi available")
except:
    print("  ⚠ fastapi not found (pip install fastapi)")

print("\n✅ Core dependencies available")
PYEOF

###############################################################################
# SUMMARY & NEXT STEPS
###############################################################################

echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    SETUP SUMMARY                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${GREEN}✅ Setup Complete!${NC}\n"

echo -e "${YELLOW}📋 NEXT STEPS:${NC}\n"

echo "1. ${YELLOW}Configure Environment${NC}"
echo "   Edit .env and set your GROQ_API_KEY"
echo "   Update OLLAMA_BASE_URL if running on different host"

echo -e "\n2. ${YELLOW}Start Services${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   Terminal 1: ollama serve"
    echo "   Terminal 2: (continue with step 3)"
else
    echo "   ✓ Ollama already running"
fi

echo -e "\n3. ${YELLOW}Test Single Query${NC}"
echo "   python improved_llm_compare.py"

echo -e "\n4. ${YELLOW}Run Batch Evaluation${NC}"
echo "   python test_rag_evaluation.py"
echo "   Select option 1 for full evaluation"

echo -e "\n5. ${YELLOW}Start API Server${NC}"
echo "   uvicorn api.api_main:app --reload"

echo -e "\n6. ${YELLOW}Test API Endpoint${NC}"
echo "   curl -X POST http://localhost:8000/api/compare/models \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"query\": \"Patient with chest pain...\", \"use_rag\": true}'"

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}📚 DOCUMENTATION:${NC}"
echo "   • PROJECT_SUMMARY.md - System overview"
echo "   • IMPLEMENTATION_GUIDE.md - Detailed guide"
echo "   • improved_llm_compare.py - Main module documentation"

echo -e "\n${YELLOW}🔗 USEFUL COMMANDS:${NC}"
echo "   # Start Ollama"
echo "   ollama serve"
echo ""
echo "   # Pull Mistral"
echo "   ollama pull mistral:7b"
echo ""
echo "   # Test single query"
echo "   python improved_llm_compare.py"
echo ""
echo "   # Batch evaluation"
echo "   python test_rag_evaluation.py"
echo ""
echo "   # Start API"
echo "   uvicorn api.api_main:app --reload"

echo -e "\n${YELLOW}⚠️  IMPORTANT:${NC}"
echo "   • This system is for research/evaluation only"
echo "   • Always validate against official clinical guidelines"
echo "   • Never use for actual clinical decisions without MD oversight"
echo "   • Follow HIPAA/privacy regulations"

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "\n${GREEN}🎉 Your Medical Triage RAG System is ready!${NC}\n"

# Offer to run tests
echo -e "${YELLOW}Would you like to run a quick test now? (y/n)${NC}"
read -p "> " response

if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
    echo -e "\n${BLUE}Starting single query test...${NC}\n"
    python improved_llm_compare.py
fi

echo -e "\n${GREEN}✅ Setup complete. Happy testing!${NC}\n"
