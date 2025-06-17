# IG 2.0 Syntax Coding Automator

An advanced Institutional Grammar 2.0 automated coding tool built with Streamlit and OpenAI API for analyzing legal and institutional statements.

## 🚀 Live Demo

[Deploy on Streamlit Cloud](https://share.streamlit.io/Chandrakiran-hj/IG-SYNTAX-PROJECT/main/streamlit_app.py)

## 📋 Features

- **Advanced IG 2.0 Analysis**: Automated extraction of regulative and constitutive statement components
- **Multiple Input Methods**: Text input, document upload (PDF, DOCX, TXT)
- **OpenAI Integration**: Powered by GPT-4o for accurate component extraction
- **Export Capabilities**: Download results as Excel, CSV, or JSON
- **Real-time Processing**: Progress tracking and status updates
- **Component Analysis**: Detailed breakdown of IG 2.0 components

## 🏗️ IG 2.0 Components

### Regulative Statements
- **A (Attribute)**: Actor performing the action
- **D (Deontic)**: Prescriptive operator (must, shall, may)
- **I (Aim)**: Goal or action to be performed
- **B (Object)**: Target or recipient of the action
- **Cac (Activation Condition)**: When the rule applies
- **Cex (Execution Constraint)**: How the action is performed
- **O (Or else)**: Consequences of non-compliance

### Constitutive Statements
- **E (Constituted Entity)**: Entity being defined
- **F (Constitutive Function)**: Function being assigned
- **P (Constituting Properties)**: Properties being assigned
- **Cac (Activation Condition)**: When the rule applies
- **Cex (Execution Constraint)**: How the action is performed
- **D (Deontic)**: Prescriptive operator (if present)
- **O (Or else)**: Consequences

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chandrakiran-hj/IG-SYNTAX-PROJECT.git
   cd IG-SYNTAX-PROJECT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account
2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your forked repository
   - Set the main file path to: `streamlit_app.py`
   - Click "Deploy"

## 🔑 Configuration

### OpenAI API Key

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Enter the key in the sidebar when running the application
3. The key is stored locally for the session only

### Model Selection

- **GPT-4o**: Best accuracy (recommended)
- **GPT-4o-mini**: Faster processing
- **GPT-4-turbo**: Alternative option

## 📖 Usage

### Text Input
1. Enter legal statements in the text area
2. Separate multiple statements with new lines
3. Click "Process Statements"

### Document Upload
1. Upload PDF, DOCX, or TXT files
2. Preview extracted text
3. Process entire documents automatically

### Results & Export
1. View coded results in structured tables
2. Filter by statement type
3. Export to Excel, CSV, or JSON

## 📊 Example Output

**Input**: "Certified farmers must submit an organic systems plan annually."

**Output**:
- **Type**: Regulative
- **IG Code**: "certified farmers (A) must (D) submit (I) organic systems plan (B) annually (Cac)"

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **OpenAI**: GPT-4o API for component extraction
- **spaCy**: Advanced NLP processing
- **NLTK**: Text processing and tokenization
- **Pandas**: Data manipulation and export
- **PyPDF2**: PDF text extraction
- **python-docx**: DOCX text extraction

### File Structure
```
IG-SYNTAX-PROJECT/
├── app.py                 # Main application
├── streamlit_app.py       # Streamlit deployment entry point
├── requirements.txt       # Python dependencies
├── .streamlit/           # Streamlit configuration
│   └── config.toml
├── nltk_data/            # NLTK resources
├── en_core_web_sm/       # spaCy model
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

Automated results should be reviewed and validated by domain experts. This tool is designed to assist with IG 2.0 coding but should not replace expert analysis.

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation in the application
- Review the IG 2.0 framework literature

---

**Built with ❤️ for Institutional Grammar 2.0 research and analysis** 