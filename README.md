# Prosody-Constrained Neural Lyric Translation (PC-NLT)

## Project Description

Prosody-Constrained Neural Lyric Translation (PC-NLT) is a natural language processing system designed to translate English song lyrics into Hindi while preserving semantic meaning and maintaining singability. Unlike conventional machine translation systems, which focus solely on semantic equivalence, PC-NLT incorporates prosodic constraints—specifically syllable count alignment—to ensure that the translated lyrics can be sung to the same melody as the original.

The system operates at the line level, processing each lyric line independently. For every input line, the model performs the following steps:

1. **Syllable Analysis of the English line**
2. **Neural Generation of Multiple Hindi Candidates**
3. **Syllable Counting and Constraint Verification**
4. **Semantic Similarity Scoring to ensure meaning preservation**
5. **Fluency and Naturalness Evaluation**
6. **Selection of the Best Candidate Based on a Weighted Ranking Mechanism**

Rather than producing literal translations, PC-NLT generates loosely adapted Hindi lyrics that reflect the original purport while resembling modern Hindi song style. The system prioritizes natural phrasing and conversational tone over formal or dictionary-like translation.

## Installation

To set up the project environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Syllable-Constrained-Neural-Machine-Translation-for-Songs
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv nlp_venv
    ```

3.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        nlp_venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source nlp_venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Technology Stack

*   **Core:** Python
*   **Deep Learning:** PyTorch, Transformers, Accelerate
*   **NLP Tools:** NLTK, Indic NLP Library, inltk, polyglot, textstat
*   **Phonetics & Syllables:** Pyphen, Epitran
*   **Evaluation:** BLEU, METEOR, BERTScore, SacreBLEU
*   **Data Handling:** Pandas, NumPy, Datasets
