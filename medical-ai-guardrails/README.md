# Medical AI Chatbot with Multi-Layer Safety Guards

A medical chatbot system with comprehensive safety guardrails designed to provide general health information while preventing harmful or inappropriate outputs.

## Disclaimer

**This chatbot is designed solely for general educational information.** It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for personal medical concerns.

## Safety Features

The chatbot implements a multi-layer safety pipeline with six detection systems:

### Input Guards

1. **Emergency Detector** - Detects medical emergencies and provides immediate help resources
2. **Suicide Risk Detector** - Identifies suicide ideation and provides crisis support resources
3. **Prompt Injection Detector** - Blocks attempts to manipulate or bypass the system
4. **Privacy Guard (Input)** - Prevents personalized medical advice requests

### Output Guards

1. **Privacy Guard (Output)** - Validates responses to ensure no personalized medical advice is provided

## Requirements

- Python 3.8+
- PyTorch
- Sentence Transformers
- Additional dependencies listed in `requirements.txt`

## Installation

### 1. Clone or download the repository

```bash
git clone <repository-url>
cd medical-ai-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
# OpenRouter API Key (required)
OPENROUTER_API_KEY=sk-or-your-api-key-here

# Model configuration (optional)
DEFAULT_MODEL=meta-llama/llama-3.3-70b-instruct:free

# Verbose mode (optional)
VERBOSE=False
```

### 4. Prepare data files

The project includes all necessary data files in the following structure:

```bash
project_root/
├── chatbot.py
├── guardrails/
│   ├── input_guards/
│   │   ├── emergency_detector.py
│   │   ├── suicide_detector.py
│   │   ├── injection_detector.py
│   │   ├── privacy_guard.py
│   │   ├── emergency_keywords.json
│   │   ├── Suicide_Detection.csv
│   │   └── injection_dataset.parquet
│   └── output_guards/
│       └── privacy_guard_output.py
```

#### Included Data Files

All required data files and pre-trained models are included in the repository:

- **emergency_keywords.json**: Keywords for emergency detection
  - Source: NIH MedlinePlus – Recognizing Medical Emergencies (https://medlineplus.gov/ency/article/001927.htm)
  - Source: Cleveland Clinic – Medical Emergencies: What to Do (https://my.clevelandclinic.org/health/articles/medical-emergency)

- **Suicide_Detection.csv**: Training data for suicide risk detection
  - Source: Kaggle - Suicide Watch Dataset (https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

- **injection_dataset.parquet**: Training data for prompt injection detection
  - Source: HuggingFace - Evaded Prompt Injection and Jailbreak Samples (https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples)

- **Pre-trained model artifacts**: All detection models are pre-trained and ready to use
  - suicide_proto_artifacts/
  - injection_proto_artifacts/
  - output_proto_artifacts/

## Usage

### Interactive Mode

Run the chatbot in interactive CLI mode:

```bash
python chatbot.py
```

Commands:

- Type your question and press Enter
- Type `verbose` to toggle detailed logging
- Type `quit`, `exit`, or `bye` to exit

## Project Structure

```bash
.
├── chatbot.py                          # Main chatbot application
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables
├── README.md                           # This file
└── guardrails/
    ├── input_guards/
    │   ├── emergency_detector.py       # Emergency detection
    │   ├── suicide_detector.py         # Suicide risk detection
    │   ├── injection_detector.py       # Prompt injection detection
    │   ├── privacy_guard.py            # Input privacy guard
    │   ├── emergency_keywords.json     # Emergency keywords
    │   ├── Suicide_Detection.csv       # Suicide training data
    │   ├── injection_dataset.parquet   # Injection training data
    │   ├── suicide_proto_artifacts/    # Pre-trained suicide detection model
    │   └── injection_proto_artifacts/  # Pre-trained injection detection model
    └── output_guards/
        ├── privacy_guard_output.py     # Output privacy guard
        └── output_proto_artifacts/     # Pre-trained output guard prototypes
```

## Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `DEFAULT_MODEL`: LLM model to use (default: meta-llama/llama-3.3-70b-instruct:free)
- `VERBOSE`: Enable detailed logging (default: False)

### Guard Parameters

Each guard can be configured with custom thresholds and parameters. See individual guard files for details.

## Ethical Considerations

This system is designed with safety as a top priority:

- Never provides personalized medical advice
- Detects and responds to crisis situations
- Prevents prompt injection and jailbreaking
- Validates all outputs for safety

**Always consult healthcare professionals for medical decisions.**