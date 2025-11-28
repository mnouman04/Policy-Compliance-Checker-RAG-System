import streamlit as st
import os
import json
from datetime import datetime
from typing import List, Dict
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(
    page_title="Policy Compliance Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .block-container {
        padding: 2rem 3rem;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .upload-box {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .result-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateX(5px);
    }
    .compliant {
        border-left-color: #51cf66;
    }
    .non-compliant {
        border-left-color: #ff6b6b;
    }
    .partial {
        border-left-color: #ffd43b;
    }
    .stat-box {
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .info-box {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #4c6ef5;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.2);
        border-radius: 10px 10px 0 0;
        color: white;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.9);
        color: #667eea;
    }
    .category-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .severity-critical {
        background-color: #ff6b6b;
        color: white;
    }
    .severity-high {
        background-color: #ff922b;
        color: white;
    }
    .severity-medium {
        background-color: #ffd43b;
        color: #333;
    }
    .severity-low {
        background-color: #51cf66;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def load_compliance_rules(rules_file: str = "compliance_rules.json") -> List[Dict]:
    with open(rules_file, 'r') as file:
        rules_data = json.load(file)
    return rules_data['compliance_rules']


def process_pdf(pdf_file, progress_bar, status_text):
    status_text.text("Saving PDF file...")
    pdf_path = "temp_policy.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    progress_bar.progress(10)
    
    status_text.text("Loading PDF document...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    progress_bar.progress(30)
    
    status_text.text(f"Splitting {len(pages)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_documents(pages)
    progress_bar.progress(50)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 0
    
    status_text.text("Creating vector database...")
    embeddings = load_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./temp_policy_vector_db"
    )
    progress_bar.progress(100)
    status_text.text("Vector database created successfully!")
    
    return vectorstore, len(pages), len(chunks)


def create_compliance_checker(vectorstore, api_key):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    compliance_prompt = """You are a compliance expert analyzing company policy documents.

Your task: Determine if the following compliance rule is satisfied in the provided document sections.

COMPLIANCE RULE:
Category: {category}
Rule: {rule_name}
Description: {rule_description}
Keywords to look for: {keywords}

RELEVANT DOCUMENT SECTIONS:
{context}

INSTRUCTIONS:
1. Carefully read the document sections
2. Determine if the rule is COMPLIANT, NON-COMPLIANT, or PARTIALLY COMPLIANT
3. Provide specific evidence from the document (quote relevant text)
4. If non-compliant, suggest what needs to be added or changed
5. Be precise and cite specific sections

Your response should follow this format:
STATUS: [COMPLIANT / NON-COMPLIANT / PARTIALLY COMPLIANT]
CONFIDENCE: [HIGH / MEDIUM / LOW]
EVIDENCE: [Quote specific text from the document that supports your finding]
EXPLANATION: [Explain your reasoning]
RECOMMENDATIONS: [If not compliant, what changes are needed?]

Begin your analysis:"""
    
    prompt = PromptTemplate(
        template=compliance_prompt,
        input_variables=["category", "rule_name", "rule_description", "keywords", "context"]
    )
    
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    return retriever, prompt, llm


def check_rule_compliance(rule: Dict, retriever, prompt: PromptTemplate, llm) -> Dict:
    search_query = f"{rule['rule_name']} {rule['description']} {' '.join(rule['keywords'])}"
    relevant_docs = retriever.invoke(search_query)
    
    context = "\n\n---\n\n".join([
        f"[Page {doc.metadata.get('page', 'N/A')}, Chunk {doc.metadata.get('chunk_id', 'N/A')}]\n{doc.page_content}"
        for doc in relevant_docs
    ])
    
    formatted_prompt = prompt.format(
        category=rule['category'],
        rule_name=rule['rule_name'],
        rule_description=rule['description'],
        keywords=', '.join(rule['keywords']),
        context=context
    )
    
    analysis = llm.invoke(formatted_prompt).content
    
    status = "UNKNOWN"
    if "STATUS: COMPLIANT" in analysis and "NON-COMPLIANT" not in analysis:
        status = "COMPLIANT"
    elif "STATUS: NON-COMPLIANT" in analysis:
        status = "NON-COMPLIANT"
    elif "STATUS: PARTIALLY COMPLIANT" in analysis:
        status = "PARTIALLY COMPLIANT"
    
    return {
        'rule_id': rule['rule_id'],
        'category': rule['category'],
        'rule_name': rule['rule_name'],
        'severity': rule['severity'],
        'status': status,
        'analysis': analysis,
        'timestamp': datetime.now().isoformat()
    }


def main():
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🔍 Policy Compliance Checker</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem; margin-top: 0;'>AI-Powered Document Compliance Analysis</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your API key")
        
        st.markdown("### 📋 About")
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; color: white;'>
        <p>This system analyzes policy documents against compliance rules using AI.</p>
        <p><strong>Features:</strong></p>
        <ul>
        <li>PDF document processing</li>
        <li>AI-powered analysis</li>
        <li>Multi-category compliance checks</li>
        <li>Detailed reporting</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Rules Available")
        try:
            rules = load_compliance_rules()
            st.metric("Total Rules", len(rules))
            categories = {}
            for rule in rules:
                cat = rule['category']
                categories[cat] = categories.get(cat, 0) + 1
            st.markdown("**By Category:**")
            for cat, count in categories.items():
                st.text(f"• {cat}: {count}")
        except:
            st.warning("Rules file not found")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results Dashboard", "📝 Detailed Report"])
    
    with tab1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.markdown("### Upload Policy Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload your policy document in PDF format"
        )
        
        if uploaded_file:
            st.success(f"✓ File uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                analyze_button = st.button("🚀 Start Analysis", use_container_width=True)
            
            if analyze_button:
                if not api_key:
                    st.error("⚠️ Please enter your Gemini API key in the sidebar")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        vectorstore, num_pages, num_chunks = process_pdf(
                            uploaded_file, progress_bar, status_text
                        )
                        
                        st.success(f"✓ Processed {num_pages} pages into {num_chunks} chunks")
                        
                        rules = load_compliance_rules()
                        retriever, prompt, llm = create_compliance_checker(vectorstore, api_key)
                        
                        st.markdown("### 🔄 Running Compliance Checks...")
                        results = []
                        
                        analysis_progress = st.progress(0)
                        analysis_status = st.empty()
                        
                        for idx, rule in enumerate(rules):
                            analysis_status.text(f"Checking rule {idx+1}/{len(rules)}: {rule['rule_name']}")
                            result = check_rule_compliance(rule, retriever, prompt, llm)
                            results.append(result)
                            analysis_progress.progress((idx + 1) / len(rules))
                        
                        st.session_state['results'] = results
                        st.session_state['analysis_complete'] = True
                        
                        analysis_status.text("✓ Analysis complete!")
                        time.sleep(1)
                        st.balloons()
                        st.success("🎉 Compliance analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        if 'results' in st.session_state and st.session_state.get('analysis_complete'):
            results = st.session_state['results']
            
            compliant = sum(1 for r in results if r['status'] == 'COMPLIANT')
            non_compliant = sum(1 for r in results if r['status'] == 'NON-COMPLIANT')
            partial = sum(1 for r in results if r['status'] == 'PARTIALLY COMPLIANT')
            total = len(results)
            compliance_rate = (compliant / total * 100) if total > 0 else 0
            
            st.markdown("### 📈 Compliance Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-number' style='color: #51cf66;'>{compliant}</div>
                    <div class='stat-label'>COMPLIANT</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-number' style='color: #ff6b6b;'>{non_compliant}</div>
                    <div class='stat-label'>NON-COMPLIANT</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-number' style='color: #ffd43b;'>{partial}</div>
                    <div class='stat-label'>PARTIAL</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-number'>{compliance_rate:.1f}%</div>
                    <div class='stat-label'>COMPLIANCE RATE</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 By Severity")
                severity_data = {}
                for severity in ['critical', 'high', 'medium', 'low']:
                    severity_results = [r for r in results if r['severity'] == severity]
                    issues = sum(1 for r in severity_results if r['status'] != 'COMPLIANT')
                    if severity_results:
                        severity_data[severity.upper()] = {
                            'total': len(severity_results),
                            'issues': issues
                        }
                
                for sev, data in severity_data.items():
                    st.markdown(f"""
                    <div class='info-box'>
                        <strong>{sev}:</strong> {data['total']} rules, {data['issues']} issues
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📋 By Category")
                categories = {}
                for result in results:
                    cat = result['category']
                    if cat not in categories:
                        categories[cat] = {'total': 0, 'compliant': 0}
                    categories[cat]['total'] += 1
                    if result['status'] == 'COMPLIANT':
                        categories[cat]['compliant'] += 1
                
                for category, counts in sorted(categories.items()):
                    rate = (counts['compliant'] / counts['total'] * 100) if counts['total'] > 0 else 0
                    st.markdown(f"""
                    <div class='info-box'>
                        <strong>{category}:</strong> {counts['compliant']}/{counts['total']} ({rate:.0f}%)
                    </div>
                    """, unsafe_allow_html=True)
            
            critical_issues = [r for r in results if r['severity'] == 'critical' and r['status'] != 'COMPLIANT']
            if critical_issues:
                st.markdown("### ⚠️ Critical Issues")
                for issue in critical_issues:
                    st.markdown(f"""
                    <div class='result-card non-compliant'>
                        <h4>🔴 {issue['rule_name']}</h4>
                        <p><strong>Category:</strong> {issue['category']}</p>
                        <p><strong>Status:</strong> {issue['status']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
        else:
            st.info("📤 Upload and analyze a document to see results here")
    
    with tab3:
        if 'results' in st.session_state and st.session_state.get('analysis_complete'):
            results = st.session_state['results']
            
            st.markdown("### 📄 Detailed Compliance Report")
            
            filter_status = st.multiselect(
                "Filter by Status",
                ["COMPLIANT", "NON-COMPLIANT", "PARTIALLY COMPLIANT"],
                default=["COMPLIANT", "NON-COMPLIANT", "PARTIALLY COMPLIANT"]
            )
            
            filter_severity = st.multiselect(
                "Filter by Severity",
                ["critical", "high", "medium", "low"],
                default=["critical", "high", "medium", "low"]
            )
            
            filtered_results = [
                r for r in results 
                if r['status'] in filter_status and r['severity'] in filter_severity
            ]
            
            st.markdown(f"**Showing {len(filtered_results)} of {len(results)} results**")
            
            for result in filtered_results:
                status_class = "compliant" if result['status'] == "COMPLIANT" else "non-compliant" if result['status'] == "NON-COMPLIANT" else "partial"
                
                severity_class = f"severity-{result['severity']}"
                
                with st.expander(f"{result['rule_name']} - {result['status']}", expanded=False):
                    st.markdown(f"""
                    <div class='result-card {status_class}'>
                        <p><strong>Rule ID:</strong> {result['rule_id']}</p>
                        <p><strong>Category:</strong> {result['category']}</p>
                        <p><strong>Severity:</strong> <span class='category-badge {severity_class}'>{result['severity'].upper()}</span></p>
                        <p><strong>Status:</strong> {result['status']}</p>
                        <hr>
                        <h4>Analysis:</h4>
                        <p style='white-space: pre-wrap;'>{result['analysis']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.button("💾 Download Full Report (JSON)", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_data = {
                    'timestamp': timestamp,
                    'total_rules': len(results),
                    'results': results
                }
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"compliance_report_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("📤 Upload and analyze a document to see detailed report here")


if __name__ == "__main__":
    main()
