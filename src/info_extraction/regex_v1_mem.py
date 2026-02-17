import re
import json
import pandas as pd
from typing import Dict, List, Optional, Union

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}\b")
ADDRESS_BLOCK_REGEX = re.compile(
    r"(?:Address|Morada|Location)\s*[:\-]\s*(.*?)(?=E-mail|Email|Phone|Telefone|Objective|Experience|Personal|Education|$)",
    re.IGNORECASE | re.DOTALL
)

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[\r\n\t\uf0b7\u2022*]+', ' ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def get_regex_dataframe(data: Union[List[Dict], List[str], Dict[str, str]]) -> pd.DataFrame:
    """
    Extract contact information from text data in memory.
    
    Args:
        data: Can be:
            - List of dicts with 'text' or 'file' keys: [{"text": "...", "file": "name"}, ...]
            - List of strings: ["text1", "text2", ...]
            - Dict with key-value pairs: {"file1": "text1", "file2": "text2", ...}
    
    Returns:
        DataFrame with columns: file, email, phone, address
    """
    rows = []
    
    # Normalize input to list of tuples (file_identifier, text)
    data_tuples = []
    
    if isinstance(data, dict):
        data_tuples = [(name, text) for name, text in data.items()]
    elif isinstance(data, list):
        if data and isinstance(data[0], dict):
            data_tuples = [(d.get("file", f"item_{i}"), d.get("text", "")) for i, d in enumerate(data)]
        else:
            data_tuples = [(f"item_{i}", text) for i, text in enumerate(data)]
    
    # Process each text item
    for file_identifier, text in data_tuples:
        try:
            if isinstance(text, dict):
                text = text.get("text", "")
            
            emails = [clean_text(m.group(0)) for m in EMAIL_REGEX.finditer(text)]
            phones = [clean_text(m.group(0)) for m in PHONE_REGEX.finditer(text)]
            
            addr_match = ADDRESS_BLOCK_REGEX.search(text)
            address = clean_text(addr_match.group(1)) if addr_match else None

            rows.append({
                "file": file_identifier,
                "email": emails[0] if emails else None,
                "phone": phones[0] if phones else None,
                "address": address
            })
        except Exception as e:
            rows.append({"file": file_identifier, "error": str(e)})

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Example usage with in-memory sample data
    sample_data = [
        {
            "file": "cv_sample1.txt",
            "text": """John Doe
Email: john.doe@example.com
Phone: +351 912-345-678
Address: 123 Main Street, Lisbon, Portugal
Experience: Software Developer""",
        },
        {
            "file": "cv_sample2.txt",
            "text": """Jane Smith
Contact: jane.smith@company.org
Telefone: (21) 4567-8900
Morada: Avenida da Liberdade, Porto
Objective: Senior Engineer""",
        },
    ]
    
    # Process in-memory data
    df_regex = get_regex_dataframe(sample_data)
    
    # Display results
    print(df_regex)
    print(f"\nExtracted {len(df_regex)} records from {len(df_regex)} CVs in memory.")