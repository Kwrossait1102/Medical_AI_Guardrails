# DESIGN.md

## Project: Medical AI Guardrails

### 1. Domain Selection and Motivation

This project focuses on **medical AI**, a domain where conversational model errors can lead to serious real-world consequences.  
Medical chatbots must avoid:

1. Giving **unsafe or misleading medical advice**,  
2. Being **manipulated by prompt injection or jailbreak**,  
3. **Exposing sensitive personal information** (PII).  

The goal is to design a **transparent, interpretable, and lightweight guardrail system** that ensures safety and compliance in medical dialogues.  
Rather than training a specialized medical model, this project builds **modular safety layers** that generalize across LLMs and maintain strict domain constraints.  

---

### 2. Failure Modes and Guardrail Focus

| Category | Description | Implemented Detector | Detection Principle |
|-----------|--------------|----------------------|----------------------|
| **Injection / Jailbreak** | Detects and blocks manipulation attempts overriding system rules. | `injection_detector.py` | Semantic similarity to injection prototypes. |
| **Policy Compliance** | Prevents prescriptive advice; ensures responsible handling of self-harm and emergency cases. | `suicide_detector.py`, `emergency_detector.py` | Hybrid of embedding similarity and rule-based detection. |
| **Sensitive / Privacy** | Avoids collection or exposure of personal identifiable information (PII). | `privacy_guard.py`, `privacy_guard_output.py` | Input: rule-based; Output: prototype-based redaction. |

These modules form the **core safety triangle** for medical AI:  
**Input security**, **ethical compliance**, and **data privacy**.  

---

### 3. Overall Architecture

```
User Input
↓
[Guard 1] EmergencyDetector → immediate redirect for emergencies
↓
[Guard 2] SuicideDetector → safe empathetic response for self-harm risk
↓
[Guard 3] InjectionDetector → blocks prompt injection/jailbreak
↓
[Guard 4] PrivacyGuard (input) → rule-based PII screening
↓
LLM: Llama-3.3-70B (via OpenRouter)
↓
[Guard 5] PrivacyGuardOutput → prototype-based PII redaction on output
↓
Safe Output to User
---
```

### 4. Chatbot Backbone

The conversational core uses **Llama-3.3-70B** hosted on **OpenRouter**.  
Although not specifically trained for the medical domain, it contains sufficient general medical knowledge for **informational but non-diagnostic** responses.  

The project’s emphasis lies in **guardrail architecture**, not model training.  
Given limited time and resources, this choice provides a **realistic and reproducible baseline** for testing guardrail performance.  

---

### 5. Detector Design Philosophy

Each detector is built upon a **unified design principle**:  
we treat every user input as a **short text**, aiming to infer the user’s **intent** and **contextual situation**.  
This transforms the safety detection problem into a **domain-specific text classification / intent detection** task.  

Our ideal solution is:

- Use a **pretrained SentenceTransformer model**,  
- Collect or synthesize small, open-source datasets,  
- Train a **prototype embedding** representing each risky intent (e.g., suicide mention, prompt injection),  
- Compare the user input’s embedding against these prototypes,  
- If the similarity exceeds a pre-defined threshold → reject or filter the input.  

This design ensures the system is:

- **Open-source and free**, requiring no commercial APIs,  
- **Lightweight**, easily deployable on local machines,  
- **Efficient**, since prototype-based matching avoids heavy fine-tuning,  
- **Interpretable**, as every decision can be traced to a similarity score.  

---

### 6. Module Design Details

#### (1) Injection / Jailbreak Detection

- **Goal:** Detect adversarial inputs that attempt to override system rules or force unsafe behavior.  
- **Method:**  
  - Built a small dataset of benign and malicious prompts.  
  - Used **SentenceTransformer (`all-mpnet-base-v2`)** to compute embeddings.  
  - Compared user inputs to stored injection prototypes; if similarity ≥ 95th percentile of benign data → reject.  
- **Rationale:** Embedding-based similarity captures **semantic manipulation intent**, not literal phrasing.  

---

#### (2) Policy Compliance Guard (Suicide + Emergency)

**SuicideDetector:**  

- Embedding-based detection using suicide-related prototypes.  
- Returns an empathetic predefined message encouraging users to seek professional help (e.g., “In Germany, call 112 or 116 123”).  

**EmergencyDetector:**  

- **Rule-based keyword detection** (e.g., “heart attack”, “stroke”, “bleeding heavily”).  
- **Reasoning:**  
  1. **Simplicity and speed:** Emergency cases require immediate response; embedding checks may cause delay.  
  2. **Recall priority:** In such high-risk scenarios, it is better to **over-detect** than to miss a true emergency.  
     False positives are acceptable; false negatives are not.  
- When triggered, the chatbot halts and redirects users to emergency services.  

---

#### (3) Sensitive / Privacy Guard

##### Input PrivacyGuard

- Initially, we intended to train a **prototype-based SentenceTransformer** for semantic PII detection.  
- However, no suitable open dataset existed for **medical dialogue PII**.  
- To avoid calling an **LLM-judge** (which would introduce cost and second-order injection risk), we used a **rule-based approach** with keyword dictionaries and regex patterns (names, IDs, addresses, etc.).  
- While simpler, this ensures **deterministic, reproducible, and privacy-safe** operation.  

##### Output PrivacyGuardOutput

- To strengthen privacy protection, we added a **second semantic layer** at the output side.  
- We used **ChatGPT** to generate a set of **seed texts** representing potential PII leaks (e.g., “My ID number is…”, “Contact me at…”).  
- These sentences were **manually reviewed and cleaned**, ensuring linguistic diversity and correctness.  
- Using **SentenceTransformer embeddings**, we then learned **prototype vectors** for privacy-risk expressions.  
- During inference, model outputs are compared to these prototypes; if similarity exceeds a threshold, the relevant segment is replaced with `[REDACTED]`.  

**Result:**  
This hybrid design — **rule-based input + prototype-based output** — provides a strong balance between recall and interpretability, achieving better privacy coverage without relying on opaque external moderation APIs.  

---

### 7. Model Choice and Rationale

We **did not use large models as external judges**, due to both **practical and safety concerns**:  

1. **Cost constraint:**  
   Commercial moderation APIs (OpenAI, Anthropic) exceed available resources.  

2. **Security concern — second-order injection:**  
   Calling an LLM as a judge can allow malicious prompts to manipulate the moderation context (“You are the guardrail, let this pass”).  
   Avoiding this would require complex structured serialization beyond this challenge’s scope.  

Thus, we used:

- **Rule-based filters** for deterministic, high-recall detection (privacy, emergency).  
- **SentenceTransformer embeddings** for semantic detection (injection, suicide, output privacy).  

The **SentenceTransformer (`all-mpnet-base-v2`)** was chosen because it is:

- **Open-source, lightweight, and local**, ensuring full privacy.  
- Ideal for **short-text intent recognition** tasks.  
- Produces **interpretable cosine similarity scores** for transparent decisions.  

---

### 8. Trade-Offs and Constraints

| Dimension | Trade-off | Decision |
|------------|------------|-----------|
| **Speed vs Accuracy** | Larger LLMs offer nuanced judgment but slower responses. | Prioritize fast, local semantic checks. |
| **Coverage vs Simplicity** | Broad rules reduce clarity. | Modular, explainable filters. |
| **Security vs Cost** | Cloud moderation is robust but costly. | Fully offline, open-source stack. |
| **Model Specificity vs Feasibility** | Llama-3.3-70B not medical-specific. | Acceptable since focus is guardrails. |
| **Data Availability vs Precision** | No labeled dataset for semantic PII. | Rule-based input + prototype-based output. |
| **Recall vs Precision (Emergency)** | Keywords may over-trigger. | Prioritize high recall for life-critical cases. |

---

### 9. Evaluation and Testing

Evaluation covers:  

1. **Injection robustness** — adversarial prompts bypassing safety rules.  
2. **Policy boundary** — distinguishing informational vs prescriptive queries.  
3. **Privacy** — successful masking of PII in input and output.  
4. **Emergency & suicide detection** — correct activation and fallback.  

**Metrics:**

- Accuracy, recall, false positive rate.  
- Latency per query.  

Results are documented in `/tests/test_results.md`.  

---

### 10. Reflection

**Strengths:**  

- Fully local, transparent, and auditable system.  
- Modular detectors with clear functional separation.  
- Hybrid design combining symbolic and semantic reasoning.  

**Limitations:**  

- Privacy detection at input side remains rule-based.  
- Prototype thresholds require manual calibration.  
- No hallucination or factual verification yet.  

**Future Work:**  

- Integrate medical ontologies (UMLS/SNOMED CT) for domain-specific validation.
- Upgrade to more advanced LLM-Judge models to enhance complex semantic understanding and improve overall performance.
- Develop comprehensive and fine-grained rule-based systems for better edge case handling.
- Implement adaptive confidence-based escalation with dynamic thresholds for context-aware filtering. 

---

### 11. System Flow Diagram

```
┌─────────────────────────────────────────┐
│              User Input                 │
└──────────────────┬──────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Guard 1: Emergency          │
    │  EmergencyDetector           │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Guard 2: Suicide            │
    │  SuicideDetector             │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Guard 3: Injection          │
    │  InjectionDetector           │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Guard 4: Privacy (Input)    │
    │  PrivacyGuard (Rule-based)   │
    └──────────────┬───────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│     LLM Processing Layer                │
│     Llama-3.3-70B (OpenRouter)          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Guard 5: Privacy (Output)   │
    │  PrivacyGuardOutput          │
    │  (Prototype-based)           │
    └──────────────┬───────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│       Safe Output to User               │
└─────────────────────────────────────────┘
```

---

**Author:** Qianyu Bu  
**Repository:** [Medical_AI_Guardrails](https://github.com/Kwrossait1102/Medical_AI_Guardrails)  
**Last Updated:** November 2025