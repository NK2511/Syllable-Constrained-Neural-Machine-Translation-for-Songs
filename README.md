# Syllable-Constrained Neural Machine Translation for Songs

This project explores the translation of English songs into Hindi with a focus on **emotional purport** and **rhythmic constraints**.

---

## 🏗️ Core Metrics
The project is built around finding the right balance between:
1. **Purport**: Capturing the high-level intent/vibe.
2. **Actual Meaning**: Semantic accuracy.
3. **Syllable Matching**: Rhythmic singability.
4. **Informality**: Ensuring a natural Bollywood lyrical style.

---

## 📂 Key Files (.py)

### 1. [syllable_counter.py](syllable_counter.py)
*   **Purpose**: Counts syllables in both English and Hindi. 
*   **Logic**: Uses phoneme-based counting for English (CMU Dict) and orthographic 'Akshar' counting for Hindi (based on Unicode analysis). This is used to ensure translation candidates fit the song's beat.

### 2. [semantic_calculator.py](semantic_calculator.py)
*   **Purpose**: A tool to compare the "Vibe" (Purport) of any two sentences. 
*   **Logic**: Uses a local Multilingual Transformer model to turn sentences into numbers (vectors). You can input any English and Hindi sentence to see a similarity score (0.0 to 1.0) showing how closely their meanings align in a conceptual space.

### 3. [semantic_translator.py](semantic_translator.py)
*   **Purpose**: A search engine that finds the best Hindi "Dub" for an English line.
*   **Logic**: It indexes all 43,000+ lines from the Bollywood database. When you enter an English line, it instantly scans the entire database to find the Hindi lyrics that most closely match the **Purport** of your input.

---

## 📊 Data
- **Hindi_Lyrics_Database**: A collection of ~1,210 Hindi song lyric files.
- **hindi_lyrics_embeddings.pt**: A pre-computed cache of semantic vectors for near-instant searching.
