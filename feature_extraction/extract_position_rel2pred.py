import spacy
nlp = spacy.load("en_core_web_sm")

def extract_word_position_and_voice(sentence):
    """
    Extract word positions relative to the predicate and determine if the sentence is in passive or active voice.

    Returns:
        - List of word positions: 'Before', 'After', or '_' for the predicate itself.
        - Sentence voice: 'Active' or 'Passive'.
    
    Args:
        sentence (list of dict): Each token is represented as a dictionary with an attribute 'form' (the word itself) 
                                 and 'predicate' (which is '_' if the token is not the predicate).
    """
    is_before = True
    features = []
    sentence_text = " ".join([word["form"] for word in sentence])  # Reconstruct sentence text
    doc = nlp(sentence_text)

    for word in sentence:
        if word["predicate"] != "_":
            features.append("_")  # Mark predicate position
            is_before = False
            continue
        features.append("Before" if is_before else "After")

    # Voice Detection
    is_passive = any(token.dep_ in {"nsubjpass", "auxpass"} for token in doc)
    
    return features, "Passive" if is_passive else "Active"