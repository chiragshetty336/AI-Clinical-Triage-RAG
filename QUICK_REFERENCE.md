# 🏥 MEDICAL TRIAGE RAG - QUICK REFERENCE CARD

## ⚡ Common Commands

### Start Services
```bash
# Terminal 1: Ollama (Mistral)
ollama serve

# Terminal 2: API Server
uvicorn api.api_main:app --reload

# Terminal 3: Tests
python improved_llm_compare.py
```

### Test Single Query
```bash
python improved_llm_compare.py
# Then follow interactive prompts
```

### Batch Evaluation
```bash
python test_rag_evaluation.py
# Select: 1 (all), 2 (sample), or 3 (single)
```

### Check Health
```bash
curl http://localhost:8000/api/compare/health
```

### API Test
```bash
curl -X POST http://localhost:8000/api/compare/models \
  -H "Content-Type: application/json" \
  -d '{"query":"chest pain","use_rag":true}'
```

---

## 🔧 Configuration Quick Setup

### 1. Environment Variables (.env)
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
GROQ_API_KEY=<your_key>
INDEX_PATH=./data/faiss_index
CACHE_PATH=./data/embeddings_cache
```

### 2. Pull Mistral
```bash
ollama pull mistral:7b
```

### 3. Verify FAISS Index
```bash
ls -la data/faiss_index
ls -la data/embeddings_cache/metadata.pkl
```

---

## 📊 Metrics at a Glance

| Metric | Range | Good | Fair | Poor |
|--------|-------|------|------|------|
| **Triage Accuracy** | % | >80% | 60-80% | <60% |
| **RAG Adherence** | 0-1 | >0.75 | 0.50-0.75 | <0.50 |
| **Model Similarity** | 0-1 | 0.65-0.85 | 0.50-0.65 | <0.50 |
| **Response Latency** | sec | <2s | 2-5s | >5s |

---

## 🎯 Expected Output Patterns

### ✅ Good RAG Integration
```
Query: [medical scenario]
✅ Retrieved 5 relevant sources (relevance: 0.85+)
✅ Triage levels match (RED/YELLOW/GREEN)
✅ RAG adherence: 0.80+
✅ Models cite specific guidelines
✅ Answer grounded in sources
```

### ❌ Poor RAG Integration
```
Query: [medical scenario]
⚠ Retrieved only 2 sources (low relevance)
❌ Triage levels don't match
❌ RAG adherence: 0.50 (not following guidelines)
❌ Models ignore retrieved context
❌ Generic/hallucinated answers
```

---

## 🚨 Error Messages & Fixes

### "FAISS index not found"
```bash
# Check if exists
ls data/faiss_index

# If not, run ingestion
# Check your main.py or ingestion script
python main.py
```

### "Ollama service not running"
```bash
# Start Ollama
ollama serve

# In new terminal, pull model
ollama pull mistral:7b
```

### "Connection refused" to Groq
```bash
# Check API key in .env
grep GROQ_API_KEY .env

# If empty, get from https://console.groq.com
# Then update .env
```

### "Timeout" on Mistral
```bash
# Increase timeout in improved_llm_compare.py
# Change: timeout=120 to timeout=300

# Or use Groq instead (faster)
use_rag=False  # Skip RAG for faster response
```

### "Low RAG adherence scores"
```bash
# 1. Check PDF quality
# 2. Verify chunks are meaningful
# 3. Try increasing top_k from 5 to 10
# 4. Use different embedding model
```

---

## 📋 Evaluation Checklist

### Before Running Tests
- [ ] Ollama running (`ollama serve`)
- [ ] FAISS index exists
- [ ] metadata.pkl loaded
- [ ] .env configured
- [ ] Dataset JSON valid
- [ ] Python dependencies installed

### After Running Tests
- [ ] Batch results exported (JSON)
- [ ] Accuracy > 70%
- [ ] RAG adherence > 0.7
- [ ] Model similarity 0.6-0.9
- [ ] No errors in logs

---

## 🔍 Debug Mode

### Inspect RAG Retrieval
```python
from improved_llm_compare import retrieve_rag_context

query = "45-year-old with chest pain, HR 110, BP 145/90"
context, sources = retrieve_rag_context(query, top_k=10)

print("Retrieved sources:")
for src in sources:
    print(f"  {src['source']} - Relevance: {src['relevance_score']}")
    print(f"    {src['content'][:200]}...")
```

### Test Single Model
```python
from improved_llm_compare import query_mistral, query_groq

# Test Mistral
result = query_mistral("Your query here", context="")
print(f"Mistral: {result['answer']}")
print(f"Error: {result['error']}")

# Test Groq
result = query_groq("Your query here", context="")
print(f"Groq: {result['answer']}")
print(f"Error: {result['error']}")
```

### Manual Evaluation
```python
from test_rag_evaluation import load_test_dataset
from improved_llm_compare import compare_models_with_rag

dataset = load_test_dataset()
test_case = dataset[0]  # First test case

result = compare_models_with_rag(
    query=test_case['query'],
    base_answer=test_case['base_answer'],
    use_rag=True
)

print(f"Expected: {test_case['expected_triage']}")
print(f"Mistral: {result['mistral']['triage_level']}")
print(f"Groq: {result['groq']['triage_level']}")
```

---

## 📊 Performance Optimization

### Speed Up Mistral
```bash
# Use quantized version
ollama pull mistral:7b-q4

# Or try Llama instead
ollama pull llama2:7b-q4
```

### Speed Up Groq
```bash
# Groq is already fast (cloud-based)
# Model selection in query_groq():
# Fastest: gemma-7b-it
# Balanced: llama2-70b-4096
# Most capable: mixtral-8x7b-32768
```

### Improve RAG Speed
```python
# Reduce top_k in retrieve_rag_context()
top_k = 3  # Was 5 (faster but less complete)

# Or cache embeddings
# Already implemented in your system
```

---

## 🧪 Testing Strategies

### Quick Test (5 min)
```bash
python improved_llm_compare.py
# Enter 1-2 queries
# Skip base answers
# Check if both models respond
```

### Medium Test (15 min)
```bash
python test_rag_evaluation.py
# Select option 2 (sample)
# Test 3-4 cases
# Review accuracy metrics
```

### Full Test (30-45 min)
```bash
python test_rag_evaluation.py
# Select option 1 (all)
# Test all 7 cases
# Export full results
# Review detailed metrics
```

---

## 🎯 Feature Mapping

| What You Want | Command | File |
|-------|---------|------|
| Compare 2 models | `python improved_llm_compare.py` | improved_llm_compare.py |
| Batch test | `python test_rag_evaluation.py` | test_rag_evaluation.py |
| Use API | `uvicorn api.api_main:app` | api/compare_routes.py |
| Add test case | Edit `medical_triage_dataset.json` | medical_triage_dataset.json |
| Change RAG params | Edit top_k, MODEL_NAME | improved_llm_compare.py:80-130 |
| Use different model | Modify `query_mistral()` | improved_llm_compare.py:150-190 |

---

## 📈 Interpreting Results

### Accuracy Scores
```
86-100% = Excellent (both models correct)
71-85%  = Good (mostly correct)
56-70%  = Fair (some errors)
<56%    = Poor (needs improvement)
```

### RAG Adherence
```
0.80-1.0 = Excellent (follows guidelines closely)
0.65-0.79 = Good (mostly follows guidelines)
0.50-0.64 = Fair (some guideline adherence)
<0.50    = Poor (ignores guidelines)
```

### Model Similarity
```
0.80-1.0 = Very similar answers (possibly hallucinating together)
0.65-0.79 = Similar with differences (good)
0.50-0.64 = Somewhat different (check for disagreement)
<0.50    = Very different (one likely wrong)
```

---

## ⏱️ Typical Execution Times

| Operation | Time | Notes |
|-----------|------|-------|
| Single query (Ollama) | 2-3s | Network + inference |
| Single query (Groq) | 1-2s | Cloud-based, faster |
| RAG retrieval | 0.5-1s | FAISS semantic search |
| Batch test (7 cases) | 30-45s | Total with both models |
| Model comparison | 0.2-0.3s | Similarity computation |

---

## 🔐 Security Notes

### Never Commit These
```bash
.env              # Contains API keys
*.pkl            # Cached data
faiss_index/     # Large index files
logs/            # May contain sensitive info
```

### Secure Your API
```bash
# Add authentication
# Use HTTPS in production
# Rate limit endpoints
# Validate inputs
# Sanitize outputs
```

### Data Privacy
```bash
# De-identify patient data
# Follow HIPAA if applicable
# Don't log medical details
# Secure database connections
```

---

## 📞 Quick Troubleshooting Flowchart

```
System not working?
  ↓
Are services running?
  ├─ No → Start Ollama + API
  └─ Yes → Continue
  ↓
Is FAISS index loaded?
  ├─ No → Run ingestion
  └─ Yes → Continue
  ↓
Do models respond?
  ├─ No → Check .env config
  └─ Yes → Continue
  ↓
Are results good quality?
  ├─ No → Improve RAG params
  └─ Yes → ✅ System working!
```

---

## 🎓 Learning Resources in Code

### RAG Logic
```python
# improved_llm_compare.py, lines 70-140
# Shows FAISS retrieval, embedding, context formatting
```

### Evaluation Metrics
```python
# test_rag_evaluation.py, lines 15-80
# Shows how to compute accuracy, adherence, similarity
```

### API Integration
```python
# compare_routes_updated.py, lines 75-150
# Shows proper request/response handling
```

### Dataset Format
```python
# medical_triage_dataset.json
# Shows expected structure for test cases
```

---

## 🚀 Next Level: Customization

### Add Your Medical PDFs
```bash
1. Place PDFs in data/guidelines/
2. Run your ingestion script
3. Restart RAG components
4. Test retrieval
```

### Add More Test Cases
```json
{
  "query_id": "CUSTOM_001",
  "query": "Your patient case...",
  "base_answer": "Expected clinical reasoning...",
  "expected_triage": "RED"
}
```

### Use Different LLMs
```python
# Edit improved_llm_compare.py
# Modify query_mistral() and query_groq() functions
# Add new function for different model
```

---

## ✅ Success Indicators

✅ System is working well if:
- All services start without errors
- Both models respond to queries
- RAG retrieves relevant sources
- Accuracy > 75%
- RAG adherence > 0.70
- Models mostly agree (similarity 0.6-0.9)

---

**Last Updated**: April 10, 2026  
**Version**: 1.0  
**Status**: Ready for Production  

For detailed info, see README.md or IMPLEMENTATION_GUIDE.md
