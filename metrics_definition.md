
These are the four core metrics that we will try to optimise when generating a hindi lyric corresponding to an English one


### 1. Similarity of Purport (Semantic Intent)
*   **Definition**: The high-level "vibe," emotional resonance, or underlying message of the line.
*   **Success Criteria**: Captures what the line is *doing* in the song rather than just what it is *saying*.
*   **Example**: "I love you" and "You are my everything" have the same **purport**

### 2. Similarity of Actual Meaning (Semantic Accuracy)
*   **Definition**: The literal, dictionary-level translation of words and phrases.
*   **Success Criteria**: High semantic alignment between the source English and target Hindi.
*   **Example**: "I love you" and "I adore you" have very similar **actual meanings**.

### 3. Syllable Matching (Rhythmic Constraint)
*   **Definition**: A hard mathematical constraint ensuring the number of syllables (Akshars) in the Hindi line matches the English original.
*   **Success Criteria**: The Hindi line can be sung perfectly to the same melody/beat as the original English line.
*   **Example**: "I love you" (3) and "I eat that" (3) match in **syllables** regardless of meaning.

### 4. Bollywood Probability (Informality)
*   **Definition**: How likely the generated sentence is to appear in a modern Bollywood song dataset.
*   **Success Criteria**: Avoids formal, "robotic," or dictionary-style Hindi. Prioritizes modern slang, poetic tropes, and natural-sounding lyrics found in the `Hindi_Lyrics_Database`.
*   **Example**: Choosing "Dil" over "Hridaya" or "Maza" over "Ananda" to ensure it sounds like a real song, not a translation.



The final translation will be the output of a multi-objective search algorithm that maximizes the weighted sum of these four scores.
