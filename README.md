# Support Ticket Triage Agent

## Overview

The Support Ticket Triage Agent is a terminal-based system that processes customer support queries, classifies them into predefined categories, and generates structured responses.

It is designed to be deterministic, explainable, and reliable, avoiding generative outputs and ensuring consistent behavior.

---

## Key Features

- Terminal-based execution
- Automatic ticket ID generation (ROW_n format)
- Combines Issue and Subject fields for better context
- Rule-based classification (no hallucination)
- Template-driven responses
- Automatic escalation for fraud/security issues
- Confidence scoring for transparency

---

## Input Format

The system expects a CSV file with the following columns:

- Issue
- Subject
- Company

---

## Output Format

The system generates a CSV file with:

- ticket_id
- request_type
- category
- decision (reply / escalate)
- response
- confidence (0–1)

---

## Example

### Input

```csv
Issue,Subject,Company
"My account was hacked","Login issue","ABC Corp"
```

### Output

```csv
ticket_id,request_type,category,decision,response,confidence
ROW_1,complaint,security,escalate,"Your issue has been escalated to our security team.",0.95
```

---

## How It Works

1. Load input CSV
2. Generate ticket IDs if missing
3. Merge Issue and Subject fields
4. Apply rule-based classification
5. Select response template
6. Decide: reply or escalate
7. Assign confidence score
8. Save output CSV

---

## Installation

### Requirements

- Python 3.8+

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py input.csv output.csv
```

- `input.csv` → Input dataset
- `output.csv` → Generated results

---

## Project Structure

```
.
├── main.py
├── classifier.py
├── templates.py
├── utils.py
├── input.csv
└── output.csv
```

---

## Limitations

- May not handle complex or ambiguous language
- Rule-based system has limited flexibility
- Template responses may not cover all cases

---

## Future Improvements

- Add machine learning-based classification
- Improve semantic understanding
- Enhance support corpus usage
- Build a user interface

---

## Summary

A simple, reliable, and explainable support ticket triage system using rule-based logic and structured outputs.
