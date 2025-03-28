import spacy
from collections import deque

# Load a spaCy model for dependency parsing.
nlp = spacy.load("en_core_web_sm")

def generate_dependency_tree(words):
    """
    Given a list of word forms, generate a dependency tree using spaCy.
    
    Returns:
        List[dict]: A list of token dictionaries, each with:
            - 'id': token id (1-indexed)
            - 'head': the id of the head token (as a string; "0" for root)
            - 'dependency_relation': the dependency relation label
            - 'form': the word form
    """
    doc = nlp(" ".join(words))
    sentence = []
    for token in doc:
        token_dict = {
            "id": str(token.i + 1),
            # For the root token, spaCy returns the token itself as head.
            "head": "0" if token.head == token else str(token.head.i + 1),
            "dependency_relation": token.dep_,
            "form": token.text
        }
        sentence.append(token_dict)
    return sentence

def find_dependency_path(token_id, sentence, pred_index):
    """
    Extract the shortest dependency path from the token (by token_id) to the predicate.
    
    Args:
        token_id (int): 1-indexed id of the starting token.
        sentence (list of dict): The dependency tree.
        pred_index (int): 0-indexed position of the predicate in the sentence.
        
    Returns:
        List[str]: A list representing the dependency path (e.g., ["↑nsubj", "↓dobj"]).
    """
    visited = set()
    queue = deque([(token_id, [])])
    predicate_id = pred_index + 1  # Convert to 1-indexed.
    
    while queue:
        current_id, path = queue.popleft()
        if current_id in visited:
            continue
        visited.add(current_id)
        
        token = sentence[current_id - 1]
        head_id = int(token['head'])
        
        # Check if we've reached the predicate.
        if current_id == predicate_id:
            return path
        
        # Traverse upward from dependent to head.
        if head_id != 0 and head_id not in visited:
            queue.append((head_id, path + [f"↑{token['dependency_relation']}"]))
        
        # Traverse downward: from current token (head) to its dependents.
        for i, child in enumerate(sentence):
            if int(child['head']) == current_id and (i + 1) not in visited:
                queue.append((i + 1, path + [f"↓{child['dependency_relation']}"]))
    
    # Return empty if no path is found.
    return []

def extract_dependency_paths(sentence):
    """
    Given a sentence represented as a list of dictionaries (each containing a word form under 'word' and
    a 'predicate' key), reconstruct the sentence, generate its dependency tree, and extract the dependency
    paths from each token to the predicate.
    
    The predicate is identified as the token whose 'predicate' value is not '_'.
    
    Args:
        sentence (List[dict]): List of word dictionaries.
        
    Returns:
        List[str]: A list of dependency paths (or '_' if no path is found).
                   For the predicate token itself, '_' is returned.
    """
    # Extract word forms.
    words = [token['form'] for token in sentence]
    
    # Find the predicate index (0-indexed) by looking for the token where 'predicate' != '_'
    predicate_index = None
    for i, token in enumerate(sentence):
        if token.get('predicate', '_') != '_':
            predicate_index = i
            break
    if predicate_index is None:
        raise ValueError("No predicate found in sentence.")
    
    # Generate the dependency tree using spaCy.
    dep_tree = generate_dependency_tree(words)
    features = []
    
    # For each token in the dependency tree, extract the path to the predicate.
    for i, token in enumerate(dep_tree):
        if i != predicate_index:
            dep_path = find_dependency_path(int(token['id']), dep_tree, predicate_index)
            features.append(''.join(dep_path) + ':' if dep_path else '_')
        else:
            features.append('_')
    
    return features