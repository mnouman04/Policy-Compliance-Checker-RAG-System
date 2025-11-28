# Policy Compliance Checker RAG System

A LangChain-based RAG system that evaluates company policies, security protocols, and HR manuals against predefined compliance rules.

## 📋 Overview

This system uses Retrieval-Augmented Generation (RAG) to automatically check policy documents for compliance with legal and organizational requirements. It provides detailed evidence, identifies non-compliant sections, and suggests remediation steps.

## 🎯 Features

- **18 Compliance Rules** covering 8 categories
- **PDF Document Processing** with intelligent chunking
- **Local Vector Embeddings** (no API quota limits)
- **AI-Powered Compliance Analysis** using Gemini 2.0 Flash
- **Multi-Step Agent Workflow** for thorough checking
- **Detailed Reporting** with evidence and recommendations
- **Comparison Tables** showing compliant vs non-compliant sections

## 📁 Project Structure

```
Task-2/
├── compliance_checker.py      # Main compliance checking system
├── compliance_rules.json      # 18 predefined compliance rules
├── task2.pdf                  # Policy document to check
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── policy_vector_db/          # Vector database (auto-created)
└── compliance_results/        # Analysis results (auto-created)
    ├── detailed_results_*.json
    ├── compliance_report_*.json
    └── comparison_table_*.txt
```

## 🔧 Installation

1. **Create virtual environment:**
```bash
python -m venv venv
```

2. **Activate virtual environment:**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API key:**
Open `compliance_checker.py` and set your Gemini API key:
```python
GEMINI_API_KEY = "your-api-key-here"
```

Get a free key from: https://aistudio.google.com/app/apikey

## 🚀 Usage

### First Run (Create Vector Database)

```python
# In compliance_checker.py, set:
CREATE_NEW_DB = True
```

Then run:
```bash
python compliance_checker.py
```

This will:
1. Load the PDF document
2. Split it into chunks
3. Create vector embeddings
4. Store in local database

### Subsequent Runs (Use Existing Database)

```python
# In compliance_checker.py, set:
CREATE_NEW_DB = False
```

Then run:
```bash
python compliance_checker.py
```

Much faster - loads existing vector database.

## 📊 Compliance Rules

The system checks 18 rules across 8 categories:

### Categories:
1. **Data Protection** (3 rules)
   - Personal data collection consent
   - Data retention periods
   - Third-party data sharing

2. **Intellectual Property** (2 rules)
   - IP ownership clauses
   - Work product assignment

3. **Liability & Indemnification** (2 rules)
   - Limitation of liability
   - Indemnification obligations

4. **Termination** (2 rules)
   - Termination notice periods
   - Post-termination obligations

5. **Payment Terms** (2 rules)
   - Payment schedules
   - Late payment penalties

6. **Confidentiality** (2 rules)
   - Confidentiality definitions
   - Confidentiality duration

7. **Dispute Resolution** (2 rules)
   - Governing law
   - Arbitration clauses

8. **Compliance & Other** (3 rules)
   - Regulatory compliance
   - Force majeure provisions
   - Amendment procedures

### Severity Levels:
- 🔴 **Critical**: Must be addressed immediately
- 🟠 **High**: Important issues requiring attention
- 🟡 **Medium**: Should be addressed soon
- 🟢 **Low**: Nice to have improvements

## 📈 Output Files

### 1. Detailed Results (`detailed_results_*.json`)
Complete analysis for each rule including:
- Compliance status (COMPLIANT / NON-COMPLIANT / PARTIALLY COMPLIANT)
- Confidence level
- Evidence from document
- AI explanation
- Recommendations
- Relevant document sections

### 2. Summary Report (`compliance_report_*.json`)
High-level overview with:
- Overall compliance rate
- Statistics by severity
- Statistics by category
- List of critical issues
- Timestamp

### 3. Comparison Table (`comparison_table_*.txt`)
Easy-to-read table showing:
- ✅ Compliant rules
- ❌ Non-compliant rules
- ⚠️ Partially compliant rules
- Organized by category and severity

## 🔍 How It Works

### 1. Document Processing
```
PDF → Pages → Chunks (1500 chars) → Vector Embeddings
```

### 2. Compliance Checking Workflow
```
For each rule:
  1. Search vector DB for relevant sections
  2. Retrieve top 5 most relevant chunks
  3. Send to Gemini AI for analysis
  4. Extract compliance status and evidence
  5. Generate recommendations
  6. Store results
```

### 3. AI Prompt Template
The system uses a specialized prompt that:
- Provides rule details and keywords
- Shows relevant document sections
- Requests structured analysis
- Asks for specific evidence
- Requires actionable recommendations

## 💡 Example Output

```
================================================================================
COMPLIANCE REPORT SUMMARY
================================================================================

Overall Statistics:
  Total Rules Checked: 18
  ✅ Compliant: 12
  ❌ Non-Compliant: 4
  ⚠️  Partially Compliant: 2
  📊 Compliance Rate: 66.7%

By Severity:
  CRITICAL: 4 rules, 2 issues
  HIGH: 7 rules, 1 issues
  MEDIUM: 5 rules, 2 issues
  LOW: 2 rules, 1 issues

⚠️  CRITICAL ISSUES FOUND (2):
  - Personal Data Collection Consent [NON-COMPLIANT]
  - Third-Party Data Sharing [PARTIALLY COMPLIANT]
```

## 🎯 Customization

### Add New Rules
Edit `compliance_rules.json`:

```json
{
  "rule_id": "R019",
  "category": "Your Category",
  "rule_name": "Your Rule Name",
  "description": "What this rule checks for",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "severity": "high"
}
```

### Adjust Chunk Size
In `compliance_checker.py`, modify:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increase for longer context
    chunk_overlap=300,  # Increase for more overlap
)
```

### Change Retrieval Count
In `create_compliance_checker()`:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Retrieve more/fewer chunks
)
```

## ⚙️ Technical Details

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (local)
- **LLM**: Google Gemini 2.0 Flash
- **Vector DB**: ChromaDB (persistent)
- **PDF Processing**: PyPDFLoader from LangChain
- **Temperature**: 0.1 (for consistent analysis)

## 🐛 Troubleshooting

### "No module named 'pypdf'"
```bash
pip install pypdf
```

### "API key not found"
Make sure you set `GEMINI_API_KEY` in `compliance_checker.py`

### "Vector database not found"
Set `CREATE_NEW_DB = True` to create it first

### Slow processing
- First run creates embeddings (takes time)
- Subsequent runs are fast
- Reduce number of rules to test faster

## 📝 Notes

- **First run** with vector creation takes 5-10 minutes
- **Subsequent runs** take 2-3 minutes for 18 rules
- Uses **local embeddings** - no API quota issues
- Only **LLM calls** use Gemini API (minimal usage)
- Results are **timestamped** - won't overwrite

## 🔐 Security

- Don't commit API keys to version control
- `.gitignore` excludes sensitive files
- Results contain document excerpts - review before sharing

## 📄 License

Educational project for learning RAG systems and compliance automation.

## 🤝 Contributing

This is an educational project. Feel free to:
- Add more compliance rules
- Improve the analysis prompt
- Add new output formats
- Enhance error handling

## 📧 Support

For questions about:
- **Gemini API**: https://ai.google.dev/docs
- **LangChain**: https://python.langchain.com/docs
- **ChromaDB**: https://docs.trychroma.com/

---

**Built with LangChain 🦜 × Gemini AI 🤖 × ChromaDB 💾**
