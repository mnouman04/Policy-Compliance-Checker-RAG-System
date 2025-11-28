import os
import json
from typing import List, Dict, Tuple
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader


GEMINI_API_KEY = "YOUR_API_KEY_HERE"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


def load_compliance_rules(rules_file: str = "compliance_rules.json") -> List[Dict]:
    print(f"Loading compliance rules from {rules_file}...")
    
    with open(rules_file, 'r') as file:
        rules_data = json.load(file)
    
    rules = rules_data['compliance_rules']
    print(f"Loaded {len(rules)} compliance rules")
    
    categories = {}
    for rule in rules:
        category = rule['category']
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    print("\nRules by category:")
    for category, count in categories.items():
        print(f"  - {category}: {count} rules")
    
    return rules


def load_policy_document(pdf_path: str) -> List[Document]:
    print(f"\nLoading policy document from {pdf_path}...")
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages from PDF")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_documents(pages)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 0
    
    print(f"Split document into {len(chunks)} chunks for analysis")
    
    return chunks


def create_vector_database(chunks: List[Document], db_path: str = "./policy_vector_db"):
    print("\nCreating vector database...")
    print("Using local sentence-transformers (no API limits)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = None
    batch_size = 500
    
    print(f"Processing {len(chunks)} chunks in batches of {batch_size}...")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(chunks)-1)//batch_size + 1
        
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=db_path
            )
        else:
            vectorstore.add_documents(batch)
        
        print("[DONE]")
    
    print("Vector database created successfully!")
    return vectorstore


def load_existing_vector_database(db_path: str = "./policy_vector_db"):
    print(f"\nLoading existing vector database from {db_path}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    print("Vector database loaded successfully!")
    return vectorstore


def create_compliance_checker(vectorstore):
    print("\nSetting up compliance checker agent...")
    
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
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    print("Compliance checker ready!")
    return retriever, prompt, llm


def check_rule_compliance(
    rule: Dict, 
    retriever, 
    prompt: PromptTemplate, 
    llm
) -> Dict:
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
    
    result = {
        'rule_id': rule['rule_id'],
        'category': rule['category'],
        'rule_name': rule['rule_name'],
        'severity': rule['severity'],
        'status': status,
        'analysis': analysis,
        'relevant_sections': [
            {
                'page': doc.metadata.get('page', 'N/A'),
                'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                'text': doc.page_content[:200] + "..."
            }
            for doc in relevant_docs
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    return result


def run_full_compliance_check(
    rules: List[Dict],
    retriever,
    prompt: PromptTemplate,
    llm
) -> List[Dict]:
    print("\n" + "="*80)
    print("STARTING FULL COMPLIANCE ANALYSIS")
    print("="*80)
    
    results = []
    
    for idx, rule in enumerate(rules, 1):
        print(f"\n[{idx}/{len(rules)}] Checking: {rule['rule_name']}")
        print(f"  Category: {rule['category']} | Severity: {rule['severity']}")
        
        result = check_rule_compliance(rule, retriever, prompt, llm)
        results.append(result)
        
        status_label = {
            'COMPLIANT': '[PASS]',
            'NON-COMPLIANT': '[FAIL]',
            'PARTIALLY COMPLIANT': '[WARN]',
            'UNKNOWN': '[UNKNOWN]'
        }
        
        label = status_label.get(result['status'], '[UNKNOWN]')
        print(f"  Result: {label} {result['status']}")
    
    return results


def generate_compliance_report(results: List[Dict]) -> Dict:
    print("\n" + "="*80)
    print("COMPLIANCE REPORT SUMMARY")
    print("="*80)
    
    compliant_count = sum(1 for r in results if r['status'] == 'COMPLIANT')
    non_compliant_count = sum(1 for r in results if r['status'] == 'NON-COMPLIANT')
    partial_count = sum(1 for r in results if r['status'] == 'PARTIALLY COMPLIANT')
    
    total = len(results)
    compliance_rate = (compliant_count / total * 100) if total > 0 else 0
    
    print(f"\nOverall Statistics:")
    print(f"  Total Rules Checked: {total}")
    print(f"  [PASS] Compliant: {compliant_count}")
    print(f"  [FAIL] Non-Compliant: {non_compliant_count}")
    print(f"  [WARN] Partially Compliant: {partial_count}")
    print(f"  [RATE] Compliance Rate: {compliance_rate:.1f}%")
    
    print(f"\nBy Severity:")
    for severity in ['critical', 'high', 'medium', 'low']:
        severity_results = [r for r in results if r['severity'] == severity]
        non_compliant = sum(1 for r in severity_results if r['status'] != 'COMPLIANT')
        if severity_results:
            print(f"  {severity.upper()}: {len(severity_results)} rules, {non_compliant} issues")
    
    print(f"\nBy Category:")
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
        print(f"  {category}: {counts['compliant']}/{counts['total']} ({rate:.0f}%)")
    
    critical_issues = [
        r for r in results 
        if r['severity'] == 'critical' and r['status'] != 'COMPLIANT'
    ]
    
    if critical_issues:
        print(f"\n[!] CRITICAL ISSUES FOUND ({len(critical_issues)}):")
        for issue in critical_issues:
            print(f"  - {issue['rule_name']} [{issue['status']}]")
    
    report = {
        'summary': {
            'total_rules': total,
            'compliant': compliant_count,
            'non_compliant': non_compliant_count,
            'partially_compliant': partial_count,
            'compliance_rate': round(compliance_rate, 2)
        },
        'by_severity': {},
        'by_category': categories,
        'critical_issues': critical_issues,
        'timestamp': datetime.now().isoformat()
    }
    
    return report


def save_results(results: List[Dict], report: Dict, output_dir: str = "compliance_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Detailed results saved to: {results_file}")
    
    report_file = os.path.join(output_dir, f"compliance_report_{timestamp}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[SAVED] Summary report saved to: {report_file}")
    
    comparison_file = os.path.join(output_dir, f"comparison_table_{timestamp}.txt")
    with open(comparison_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPLIANCE COMPARISON TABLE\n")
        f.write("="*100 + "\n\n")
        
        f.write("[PASS] COMPLIANT RULES:\n")
        f.write("-"*100 + "\n")
        compliant_rules = [r for r in results if r['status'] == 'COMPLIANT']
        for r in compliant_rules:
            f.write(f"  [{r['rule_id']}] {r['rule_name']} ({r['category']})\n")
        
        f.write("\n\n")
        
        f.write("[FAIL] NON-COMPLIANT RULES:\n")
        f.write("-"*100 + "\n")
        non_compliant_rules = [r for r in results if r['status'] == 'NON-COMPLIANT']
        for r in non_compliant_rules:
            f.write(f"  [{r['rule_id']}] {r['rule_name']} ({r['category']}) - SEVERITY: {r['severity'].upper()}\n")
        
        f.write("\n\n")
        
        f.write("[WARN] PARTIALLY COMPLIANT RULES:\n")
        f.write("-"*100 + "\n")
        partial_rules = [r for r in results if r['status'] == 'PARTIALLY COMPLIANT']
        for r in partial_rules:
            f.write(f"  [{r['rule_id']}] {r['rule_name']} ({r['category']})\n")
    
    print(f"[SAVED] Comparison table saved to: {comparison_file}")
    print(f"\n[FOLDER] All results saved in: {output_dir}/")


def main():
    print("="*80)
    print("POLICY COMPLIANCE CHECKER SYSTEM")
    print("="*80)
    
    PDF_PATH = "task2.pdf"
    RULES_FILE = "compliance_rules.json"
    CREATE_NEW_DB = False
    
    rules = load_compliance_rules(RULES_FILE)
    
    if CREATE_NEW_DB:
        chunks = load_policy_document(PDF_PATH)
        vectorstore = create_vector_database(chunks)
    else:
        vectorstore = load_existing_vector_database()
    
    retriever, prompt, llm = create_compliance_checker(vectorstore)
    
    results = run_full_compliance_check(rules, retriever, prompt, llm)
    
    report = generate_compliance_report(results)
    
    save_results(results, report)
    
    print("\n" + "="*80)
    print("COMPLIANCE CHECK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
