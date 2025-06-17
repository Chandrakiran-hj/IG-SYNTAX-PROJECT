# NOTE: For deployment, ensure you include the 'nltk_data' folder (with required NLTK resources) and the 'en_core_web_sm' spaCy model folder in your project root.
# See deployment instructions in the README.

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import io
import base64
from typing import List, Dict, Tuple, Optional, Any
import traceback
from datetime import datetime
import time
import os
from pathlib import Path

# Core libraries
import openai
from openai import OpenAI
import PyPDF2
import docx
from docx import Document
import xlsxwriter
from io import BytesIO

# Advanced NLP libraries
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
import textstat
import nltk

# Set up local nltk_data path
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

def ensure_nltk_resource(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        st.error(f"Required NLTK resource '{resource}' not found in local 'nltk_data' directory. Please download it and place it in the 'nltk_data' folder.")
        raise

ensure_nltk_resource('tokenizers/punkt')
ensure_nltk_resource('taggers/averaged_perceptron_tagger')
ensure_nltk_resource('chunkers/maxent_ne_chunker')
ensure_nltk_resource('corpora/words')
ensure_nltk_resource('corpora/stopwords')

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        # Try to load local model first (for deployment)
        return spacy.load("./en_core_web_sm")
    except OSError:
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            st.error("SpaCy English model not found. Please include 'en_core_web_sm' in your project or install it with: python -m spacy download en_core_web_sm")
            return None

# Configuration
st.set_page_config(
    page_title="IG 2.0 Syntax Coding Automator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# IG 2.0 Syntax Components and Rules
IG_COMPONENTS = {
    "regulative": {
        "A": "Attribute (Actor)",
        "I": "Aim (Action/Goal)",
        "C": "Context (Conditions/Constraints)",
        "B": "Object (Target/Recipient)",
        "D": "Deontic (Obligation/Permission)",
        "O": "Or else (Consequences)"
    },
    "constitutive": {
        "E": "Constituted Entity",
        "F": "Constitutive Function",
        "P": "Constituting Properties",
        "C": "Context",
        "D": "Deontic",
        "O": "Or else"
    }
}

DEONTIC_OPERATORS = {
    "must": "obligation",
    "shall": "obligation",
    "should": "obligation",
    "required": "obligation",
    "mandatory": "obligation",
    "must not": "prohibition",
    "shall not": "prohibition",
    "prohibited": "prohibition",
    "forbidden": "prohibition",
    "may": "permission",
    "can": "permission",
    "allowed": "permission",
    "permitted": "permission",
    "will": "prediction",
    "would": "conditional"
}

CONTEXT_TYPES = {
    "temporal": ["when", "during", "before", "after", "while", "annually", "daily", "monthly"],
    "spatial": ["where", "at", "in", "on", "near", "within", "outside"],
    "conditional": ["if", "unless", "provided that", "in case", "when"],
    "procedural": ["according to", "pursuant to", "following", "in accordance with"],
    "domain": ["for", "regarding", "concerning", "with respect to"]
}

class IGSyntaxCoder:
    """Advanced IG 2.0 Syntax Coding Engine"""
    
    def __init__(self, openai_client, nlp_model=None):
        self.client = openai_client
        self.nlp = nlp_model
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text and extract sentences"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def identify_statement_type(self, sentence: str) -> str:
        """Identify if statement is regulative or constitutive"""
        # Use OpenAI to classify statement type
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in Institutional Grammar 2.0. Classify the user's statement as 'regulative' or 'constitutive'.

- **Regulative statements** prescribe behavior. They specify an actor (Attribute) who is expected to perform an action (Aim). They regulate what actors must, may, or must not do.
- **Constitutive statements** create, define, or modify entities, roles, or realities. They define what something *is* (e.g., 'A certified organic farm is defined as...').

Respond with only 'regulative' or 'constitutive'."""
                    },
                    {
                        "role": "user",
                        "content": f"Classify this statement: {sentence}"
                    }
                ],
                temperature=0.1,
                max_tokens=10
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            st.error(f"Error classifying statement type: {str(e)}")
            return "regulative"  # Default fallback
    
    def extract_components_with_openai(self, sentence: str, statement_type: str) -> Dict:
        """Extract IG components using OpenAI with detailed prompting"""
        
        if statement_type == "regulative":
            component_prompt = """
            Extract IG 2.0 REGULATIVE components. Return a JSON object with these keys. If a component is not present, use an empty string "".

            - "A" (Attribute): An actor (individual or corporate) that carries out, or is expected to/to not carry out, the action. Include descriptors of the actor.
            - "D" (Deontic): Extract ONLY the core prescriptive operator (e.g., must, shall, may, must not, shall not). For example, from "shall have the right to", the correct output is "shall".
            - "I" (Aim): The goal or action of the statement assigned to the Attribute.
            - "B" (Object): The inanimate or animate receiver of the action (Aim). Can be direct or indirect, real-world or abstract.
            - "Cac" (Activation Condition): The context for when the rule applies (e.g., 'when', 'if'). Default if not specified: 'under all conditions'.
            - "Cex" (Execution Constraint): The context that qualifies how the action is performed (e.g., 'according to', 'in such manner as'). Default if not specified: 'no constraints'.
            - "O" (Or else): The consequence or incentive associated with compliance or non-compliance.
            - "nesting": Contains nested clauses (true/false).
            - "logical_operators": Any AND, OR, NOT relationships.
            
            Example format:
            {
                "A": "certified farmer",
                "D": "must",
                "I": "submit",
                "B": "organic systems plan",
                "Cac": "annually",
                "Cex": "",
                "O": "",
                "nesting": false,
                "logical_operators": ""
            }
            """
        else:
            component_prompt = """
            Extract IG 2.0 CONSTITUTIVE components. Return a JSON object with these keys. If a component is not present, use an empty string "".

            - "E" (Constituted Entity): The entity being constituted, reconstituted, or modified.
            - "F" (Constitutive Function): The action that constitutes the entity or reflects its functional relationship to properties (e.g., constitutes, is defined as, serves as).
            - "P" (Constituting Properties): Properties or characteristics linked to the entity.
            - "Cac" (Activation Condition): The context for when the rule applies (e.g., 'when', 'if'). Default if not specified: 'under all conditions'.
            - "Cex" (Execution Constraint): The context that qualifies how the action is performed. Default if not specified: 'no constraints'.
            - "D" (Deontic): If present, extract ONLY the core prescriptive operator (e.g., must, shall, may, must not). For example, from "shall be considered", the correct output is "shall".
            - "O" (Or else): The consequence or incentive.
            - "nesting": Contains nested clauses (true/false).
            - "logical_operators": Any AND, OR, NOT relationships.
            
            Example format:
            {
                "E": "organic certification",
                "F": "constitutes",
                "P": "valid agricultural status",
                "Cac": "when issued by accredited body",
                "Cex": "",
                "D": "",
                "O": "",
                "nesting": false,
                "logical_operators": ""
            }
            """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert in Institutional Grammar 2.0 coding. {component_prompt}"
                    },
                    {
                        "role": "user",
                        "content": f"Extract components from: {sentence}"
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            content = re.sub(r'```json\n|```', '', content)
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback to basic parsing if JSON fails
                return self.fallback_component_extraction(sentence, statement_type)
                
        except Exception as e:
            st.error(f"Error extracting components: {str(e)}")
            return self.fallback_component_extraction(sentence, statement_type)
    
    def fallback_component_extraction(self, sentence: str, statement_type: str) -> Dict:
        """Fallback method for component extraction using rule-based approach"""
        components = {}
        
        # Basic deontic detection
        deontic_found = ""
        for deontic, category in DEONTIC_OPERATORS.items():
            if deontic.lower() in sentence.lower():
                deontic_found = deontic
                break
        
        # Basic context detection
        context_found = ""
        for context_type, keywords in CONTEXT_TYPES.items():
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    context_found = keyword
                    break
        
        if statement_type == "regulative":
            components = {
                "A": "",
                "I": "",
                "Cac": "",
                "Cex": "",
                "B": "",
                "D": deontic_found,
                "O": "",
                "nesting": False,
                "logical_operators": ""
            }
        else:
            components = {
                "E": "",
                "F": "",
                "P": "",
                "Cac": context_found,
                "Cex": "",
                "D": deontic_found,
                "O": "",
                "nesting": False,
                "logical_operators": ""
            }
        
        return components
    
    def generate_ig_code(self, sentence: str, components: Dict, statement_type: str) -> str:
        """Generate formatted IG code string"""
        if statement_type == "regulative":
            code_parts = []
            if components.get("A"):
                code_parts.append(f"{components['A']} (A)")
            if components.get("D"):
                code_parts.append(f"{components['D']} (D)")
            if components.get("I"):
                code_parts.append(f"{components['I']} (I)")
            if components.get("B"):
                code_parts.append(f"{components['B']} (B)")
            if components.get("Cac"):
                code_parts.append(f"{components['Cac']} (Cac)")
            if components.get("Cex"):
                code_parts.append(f"{components['Cex']} (Cex)")
            if components.get("O"):
                code_parts.append(f"or else [{components['O']}] (O)")
        else:
            code_parts = []
            if components.get("E"):
                code_parts.append(f"{components['E']} (E)")
            if components.get("F"):
                code_parts.append(f"{components['F']} (F)")
            if components.get("P"):
                code_parts.append(f"{components['P']} (P)")
            if components.get("Cac"):
                code_parts.append(f"{components['Cac']} (Cac)")
            if components.get("Cex"):
                code_parts.append(f"{components['Cex']} (Cex)")
            if components.get("D"):
                code_parts.append(f"{components['D']} (D)")
            if components.get("O"):
                code_parts.append(f"or else [{components['O']}] (O)")
        
        return " ".join(code_parts) if code_parts else "Unable to parse"
    
    def process_statements(self, statements: List[str], progress_callback=None) -> List[Dict]:
        """Process multiple statements and return results"""
        results = []
        total = len(statements)
        
        for i, statement in enumerate(statements):
            if progress_callback:
                progress_callback(i + 1, total)
            
            # Clean statement
            statement = statement.strip()
            if not statement:
                continue
            
            try:
                # Identify statement type
                stmt_type = self.identify_statement_type(statement)
                
                # Extract components
                components = self.extract_components_with_openai(statement, stmt_type)
                
                # Generate IG code
                ig_code = self.generate_ig_code(statement, components, stmt_type)
                
                # Prepare result
                result = {
                    "Statement": statement,
                    "Type": stmt_type.title(),
                    "IG_Code": ig_code,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add individual components
                if stmt_type == "regulative":
                    result.update({
                        "Attribute_A": components.get("A", ""),
                        "Aim_I": components.get("I", ""),
                        "Activation_Cac": components.get("Cac", ""),
                        "Execution_Cex": components.get("Cex", ""),
                        "Object_B": components.get("B", ""),
                        "Deontic_D": components.get("D", ""),
                        "OrElse_O": components.get("O", "")
                    })
                else:
                    result.update({
                        "Entity_E": components.get("E", ""),
                        "Function_F": components.get("F", ""),
                        "Properties_P": components.get("P", ""),
                        "Activation_Cac": components.get("Cac", ""),
                        "Execution_Cex": components.get("Cex", ""),
                        "Deontic_D": components.get("D", ""),
                        "OrElse_O": components.get("O", "")
                    })
                
                result.update({
                    "Nesting": components.get("nesting", False),
                    "Logical_Operators": components.get("logical_operators", "")
                })
                
                results.append(result)
                
            except Exception as e:
                st.error(f"Error processing statement: {statement[:100]}... - {str(e)}")
                continue
        
        return results

# Document processing functions
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    try:
        return str(file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

def create_excel_download(df: pd.DataFrame) -> bytes:
    """Create Excel file for download"""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='IG_Coding_Results', index=False)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['IG_Coding_Results']
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write the column headers with formatting
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-adjust column widths
        for i, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).str.len().max(),
                len(str(col))
            )
            worksheet.set_column(i, i, min(max_length + 2, 50))
    
    buffer.seek(0)
    return buffer.getvalue()

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'ig_results' not in st.session_state:
        st.session_state.ig_results = []
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

# Main application
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ IG 2.0 Syntax Coding Automator</h1>
        <p>Advanced Institutional Grammar 2.0 automated coding tool using OpenAI API</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("🔑 Configuration")
        
        # OpenAI API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. It will be stored locally in cache for this session only."
        )
        
        if api_key:
            try:
                st.session_state.openai_client = OpenAI(api_key=api_key)
                st.success("✅ OpenAI API key configured successfully!")
            except Exception as e:
                st.error(f"❌ Invalid API key: {str(e)}")
                st.session_state.openai_client = None
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "OpenAI Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=0,
            help="Choose the OpenAI model for processing"
        )
        
        # Processing options
        enable_advanced_nlp = st.checkbox(
            "Enable Advanced NLP",
            value=True,
            help="Use spaCy and NLTK for enhanced text processing"
        )
        
        # Clear results button
        if st.button("🗑️ Clear All Results"):
            st.session_state.ig_results = []
            st.session_state.processing_complete = False
            st.rerun()
    
    # Check if API key is configured
    if not st.session_state.openai_client:
        st.warning("⚠️ Please configure your OpenAI API key in the sidebar to begin.")
        st.info("""
        **Getting Started:**
        1. Enter your OpenAI API key in the sidebar
        2. Choose your preferred model settings
        3. Input legal statements or upload documents
        4. Click 'Process Statements' to generate IG 2.0 coding
        """)
        return
    
    # Load NLP model
    nlp_model = None
    if enable_advanced_nlp:
        nlp_model = load_spacy_model()
    
    # Initialize IG Syntax Coder
    ig_coder = IGSyntaxCoder(st.session_state.openai_client, nlp_model)
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📄 Document Upload", "📊 Results & Export"])
    
    with tab1:
        st.header("📝 Legal Statement Input")
        
        # Text input area
        statement_text = st.text_area(
            "Enter Legal Statements",
            height=200,
            placeholder="Enter legal statements here. Use new lines to separate multiple statements.\n\nExample:\nCertified farmers must submit an organic systems plan annually.\nOrganic certifiers must send farmers notification of compliance within thirty days of inspection.",
            help="Enter one or more legal statements. Each statement should be on a separate line."
        )
        
        # Process button
        col1, col2 = st.columns([1, 3])
        with col1:
            process_button = st.button("🚀 Process Statements", type="primary")
        
        with col2:
            if st.button("➕ Add to Existing Results"):
                if statement_text.strip():
                    statements = [s.strip() for s in statement_text.split('\n') if s.strip()]
                    
                    # Processing with progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(current, total):
                        progress = current / total
                        progress_bar.progress(progress)
                        status_text.text(f"Processing statement {current} of {total}")
                    
                    # Process statements
                    new_results = ig_coder.process_statements(statements, update_progress)
                    
                    # Add to existing results
                    st.session_state.ig_results.extend(new_results)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Added {len(new_results)} new statements to results!")
                    st.rerun()
        
        # Process statements
        if process_button and statement_text.strip():
            statements = [s.strip() for s in statement_text.split('\n') if s.strip()]
            
            if statements:
                # Clear previous results
                st.session_state.ig_results = []
                
                # Processing with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing statement {current} of {total}")
                
                # Process statements
                with st.spinner("Processing statements with IG 2.0 syntax coding..."):
                    results = ig_coder.process_statements(statements, update_progress)
                
                # Store results
                st.session_state.ig_results = results
                st.session_state.processing_complete = True
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Successfully processed {len(results)} statements!")
                st.rerun()
            else:
                st.warning("⚠️ Please enter at least one legal statement.")
    
    with tab2:
        st.header("📄 Document Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Legal Document",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files containing legal statements"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"📁 **File:** {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Extract text based on file type
            extracted_text = ""
            
            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                extracted_text = extract_text_from_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                extracted_text = extract_text_from_txt(uploaded_file)
            
            if extracted_text:
                # Display preview
                with st.expander("📖 Document Preview", expanded=False):
                    st.text_area("Extracted Text", extracted_text, height=300)
                
                # Process document
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🚀 Process Document", type="primary"):
                        # Preprocess and extract statements
                        statements = ig_coder.preprocess_text(extracted_text)
                        
                        if statements:
                            # Clear previous results
                            st.session_state.ig_results = []
                            
                            # Processing with progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(current, total):
                                progress = current / total
                                progress_bar.progress(progress)
                                status_text.text(f"Processing statement {current} of {total}")
                            
                            # Process statements
                            with st.spinner(f"Processing {len(statements)} statements from document..."):
                                results = ig_coder.process_statements(statements, update_progress)
                            
                            # Store results
                            st.session_state.ig_results = results
                            st.session_state.processing_complete = True
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success(f"✅ Successfully processed {len(results)} statements from document!")
                            st.rerun()
                        else:
                            st.warning("⚠️ No valid statements found in the document.")
                
                with col2:
                    if st.button("➕ Add Document to Existing"):
                        statements = ig_coder.preprocess_text(extracted_text)
                        
                        if statements:
                            # Processing with progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(current, total):
                                progress = current / total
                                progress_bar.progress(progress)
                                status_text.text(f"Processing statement {current} of {total}")
                            
                            # Process statements
                            new_results = ig_coder.process_statements(statements, update_progress)
                            
                            # Add to existing results
                            st.session_state.ig_results.extend(new_results)
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success(f"✅ Added {len(new_results)} statements from document to results!")
                            st.rerun()
            else:
                st.error("❌ Failed to extract text from the uploaded file.")
    
    with tab3:
        st.header("📊 Results & Export")
        
        if st.session_state.ig_results:
            df = pd.DataFrame(st.session_state.ig_results)
            
            # Regulative table columns
            regulative_cols = [
                'Statement', 'Attribute_A', 'Deontic_D', 'Aim_I', 'Object_B', 'Activation_Cac', 'Execution_Cex', 'OrElse_O'
            ]
            # Constitutive table columns
            constitutive_cols = [
                'Statement', 'Entity_E', 'Function_F', 'Properties_P', 'Activation_Cac', 'Execution_Cex', 'Deontic_D', 'OrElse_O'
            ]

            # Filter dataframes
            df_reg = df[df['Type'] == 'Regulative'][regulative_cols].copy() if 'Attribute_A' in df.columns else pd.DataFrame(columns=regulative_cols)
            df_con = df[df['Type'] == 'Constitutive'][constitutive_cols].copy() if 'Entity_E' in df.columns else pd.DataFrame(columns=constitutive_cols)

            st.subheader('Regulative Statements Table')
            if not df_reg.empty:
                reg_rows = df_reg.astype(str).apply(lambda row: ','.join(row.values), axis=1)
                reg_content = '\n'.join(reg_rows)
                if st.button('Copy Regulative Table'):
                    show_copyable_text('Copy the content below:', reg_content, 'reg_copy')
                st.dataframe(df_reg, use_container_width=True, hide_index=True)
            else:
                st.info('No regulative statements found.')

            st.subheader('Constitutive Statements Table')
            if not df_con.empty:
                con_rows = df_con.astype(str).apply(lambda row: ','.join(row.values), axis=1)
                con_content = '\n'.join(con_rows)
                if st.button('Copy Constitutive Table'):
                    show_copyable_text('Copy the content below:', con_content, 'con_copy')
                st.dataframe(df_con, use_container_width=True, hide_index=True)
            else:
                st.info('No constitutive statements found.')
            
            st.divider()
            
            # Export options
            st.header("📥 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download as Excel
                excel_data = create_excel_download(df)
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"ig_coding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # Download as CSV
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"ig_coding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Download as JSON
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"ig_coding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            # No results yet
            st.info("📋 No results to display yet. Process some legal statements in the Text Input or Document Upload tabs.")
            
            # Show IG 2.0 reference guide
            with st.expander("📚 IG 2.0 Quick Reference Guide", expanded=False):
                st.markdown("""
                ### Regulative Statement Components:
                - **A (Attribute)**: The actor who performs the action
                - **I (Aim)**: The action or goal to be performed
                - **Cac (Activation Condition)**: The condition(s) that must be met for the rule to be activated
                - **Cex (Execution Constraint)**: The condition(s) that must be met during/for execution
                - **B (Object)**: The target or recipient of the action
                - **D (Deontic)**: Obligation/permission indicators (must, shall, may, etc.)
                - **O (Or else)**: Consequences of non-compliance
                
                ### Constitutive Statement Components:
                - **E (Constituted Entity)**: The entity being defined
                - **F (Constitutive Function)**: The function being assigned
                - **P (Constituting Properties)**: Properties being assigned
                - **Cac (Activation Condition)**: The context for when the rule applies.
                - **Cex (Execution Constraint)**: The context that qualifies how the action is performed.
                - **D (Deontic)**: Obligation/permission elements
                - **O (Or else)**: Consequences
                
                ### Example Codings:
                **Regulative**: "Certified farmer (A) must (D) submit (I) organic systems plan (B) annually (Cac)"
                
                **Constitutive**: "Organic certification (E) constitutes (F) valid agricultural status (P) when issued by accredited body (Cac)"
                """)

# Additional utility functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ig_examples():
    """Get cached IG 2.0 examples for reference"""
    return {
        "regulative_examples": [
            "Certified farmers must submit an organic systems plan annually.",
            "Organic certifiers must send farmers notification of compliance within thirty days of inspection.",
            "Certified organic farmers must not apply synthetic chemicals to crops at any time once organic certification is conferred, or else certifier will revoke certification from farmer."
        ],
        "constitutive_examples": [
            "Organic certification constitutes valid agricultural status when issued by an accredited body.",
            "A certified organic farm is defined as a farm that has met all organic standards and requirements.",
            "The organic systems plan serves as the foundation document for organic certification."
        ]
    }

# Error handling and logging
def log_error(error_message: str, context: str = ""):
    """Log errors for debugging"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_log = f"[{timestamp}] {context}: {error_message}"
    
    # In production, you might want to log to a file or external service
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    
    st.session_state.error_log.append(error_log)
    
    # Keep only last 100 errors
    if len(st.session_state.error_log) > 100:
        st.session_state.error_log = st.session_state.error_log[-100:]

# Help and documentation section
def show_help_section():
    """Display help and documentation"""
    with st.expander("🆘 Help & Documentation", expanded=False):
        st.markdown("""
        ## How to Use This Tool
        
        ### 1. Configuration
        - Enter your OpenAI API key in the sidebar
        - Choose your preferred model (GPT-4o recommended for best results)
        - Enable advanced NLP features for enhanced processing
        
        ### 2. Input Methods
        **Text Input Tab:**
        - Enter legal statements directly in the text area
        - Use new lines to separate multiple statements
        - Click "Process Statements" to analyze
        
        **Document Upload Tab:**
        - Upload PDF, DOCX, or TXT files
        - Preview extracted text before processing
        - Process entire documents automatically
        
        ### 3. Results & Analysis
        - View coded results in structured table format
        - Filter by statement type (Regulative/Constitutive)
        - Export results to Excel, CSV, or JSON
        - Analyze patterns and quality metrics
        
        ### 4. Advanced Features
        - **Progressive Processing**: Add new statements to existing results
        - **Component Analysis**: Frequency analysis of IG components
        - **Quality Metrics**: Completion rates and quality scores
        - **Pattern Recognition**: Complexity and structure analysis
        
        ### 5. Troubleshooting
        - Ensure API key is valid and has sufficient credits
        - Check internet connection for API calls
        - Large documents may take several minutes to process
        - Complex legal language may require manual review
        
        ### 6. Best Practices
        - Review automated coding results for accuracy
        - Use clear, well-structured legal statements
        - Break down complex nested statements when possible
        - Verify deontic operators and context clauses
        """)

# Footer information
def show_footer():
    """Display footer with additional information"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>IG 2.0 Syntax Coding Automator | Built with Streamlit & OpenAI API</p>
        <p>⚠️ Automated results should be reviewed and validated by domain experts</p>
        <p>📚 Based on Institutional Grammar 2.0 framework</p>
    </div>
    """, unsafe_allow_html=True)

def show_copyable_text(label, text, key):
    st.text_area(label, text, height=200, key=key)
    st.markdown(
        f"""
        <script>
        const textarea = document.querySelector('textarea[data-testid="stTextArea"]');
        if (textarea) {{
            textarea.focus();
            textarea.select();
        }}
        </script>
        """,
        unsafe_allow_html=True,
    )

# Run the application
if __name__ == "__main__":
    try:
        main()
        show_help_section()
        show_footer()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")
        log_error(str(e), "Main Application")
        
        # Show error details in debug mode
        if st.checkbox("Show Error Details (Debug Mode)"):
            st.code(traceback.format_exc())
