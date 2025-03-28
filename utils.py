
import pandas as pd
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import ast
from sklearn.metrics import classification_report


def read_data_as_sentence(file_path, output_path):
    """
    Parses the CoNNL-U Plus file and returns a dataframe of sentences.
    Extracting features from the data and return a dataframe of sentence's Input_form list and sentence's arguments list.

    Returns a dataframe, where each row represents one sentence with its all words and all features of words (each columns is a list with lengh of number of words in sentence).

    file_path (str): The file path to the data to be preprocessed.
    output_path (str): The file path to the save processed dataframe.
    """
    # sentences list for all sentence in data file
    sentences = []
    # sentence list for all token in each sentence
    sentence = []  # Initialize an empty list for the current sentence
    # Open data file
    with open(file_path, 'r', encoding="utf8") as file:
        # For each line in data file
        for line in file:
            # split line by TAB (\t)
            line = line.strip().split('\t')
            # If the line starts with '#', it's a comment, ignore it
            if line[0].startswith('#'):
                continue
            # If the line is not empty
            elif line[0].strip() != '':

                # Create a token if its ID does not contain a period
                # Each token only has form, predicate, and arguments (argument per each predicate of sentence)
                if '.' not in line[0] and len(line) > 10:
                    token = {
                        'form': line[1],
                        'predicate': line[10],
                        'argument': line[11:]  # Store all remaining elements as arguments.
                    }
                    # Append the token to the sentence.
                    sentence.append(token)

            # A new line indicates the end of a sentence.
            # If line is empty one sentence has been finished.
            elif line[0].strip() == '':
                # Append the completed sentence to the sentences list.
                sentences.append(sentence)
                # Reset sentence for the next sentence.
                sentence = []

    # Iterate over all sentences. Create copies of sentences for each predicate.
    expanded_sentences = []
    for sentence in sentences:
        # Find all predicates in the sentence.
        predicates = [token['predicate'] for token in sentence if token['predicate'] != '_']

        # For every predicate, create a copy of the sentence.
        for index, predicate in enumerate(predicates):
            # For each predicate in the sentence, we make a copy of the sentence.
            sentence_copy = [token.copy() for token in sentence]
            # Finding the predicate 'form' of the sentence predicate.
            predicate_form = [token['form'] for token in sentence_copy if token['predicate'] == predicate]
            for token in sentence_copy:

                token['predicate'] = predicate_form[0]

                # Keep only the relevant argument for this predicate. Overwrite 'V' and 'C-V' with '_'.
                if token['argument'][index] == 'V' or token['argument'][index] == 'C-V':
                    token['argument'] = '_'
                else:
                    token['argument'] = token['argument'][index]
                # token['argument'] = token['argument'][index] if token['argument'][index] != 'V' else '_'
                # token['argument'] = token['argument'][index] if token['argument'][index] != 'C-V' else '_'
            expanded_sentences.append(sentence_copy)

    # Create a list to store each sentence.
    final_list = []
    # Iterate over all sentences after copy sentences for each predicate.
    for sentence in expanded_sentences:
        # Create two empty list one for input_form (token form) and one for arguments of sentence.
        form_list =[]
        argument_list =[]
        # For each word in sentence append features in their list
        # each list has all its feature for all words in sentence
        for word in sentence:
            form_list.append(word['form'])
            argument_list.append(word['argument'])
        # Creating emptylist, a list of "_" with the same length as the argument list, to control statements whose argument list is empty.
        emptylist = ['_']* (len(argument_list))##
        # append two None into argument list and emptylist. one for [SEP] and one for predicate that are appended into form list.
        argument_list.append(None)
        argument_list.append(None)
        emptylist.append(None)##
        emptylist.append(None)##
        # argument_list.append('_')
        # append [SEP] and predicate into form list
        form_list.append('[SEP]')
        form_list.append(str(sentence[0]['predicate']).split('.')[0])
        # After all words in a sentence processed, append all list of sentence to final list as a dict.
        if emptylist != argument_list:##
            final_list.append({
                        'input_form': form_list,
                        'argument': argument_list})
    # Convert list to pandas dataframe
    df = pd.DataFrame(final_list)
    # Save Dataframe to output_path
    df.to_csv(output_path)
    # return Dataframe
    return df

def get_label_mapping(train_df, test_df, dev_df):
    """
    Get the mapping of labels from the argument columns of multiple dataframes.
    Return a dictionary mapping labels to numerical values. None key stays as None value.

    Parameters:
    - train_data (DataFrame): pandas DataFrame containing the training data.
    - test_data (DataFrame): pandas DataFrame containing the testing data.
    - dev_data (DataFrame): pandas DataFrame containing the development data.
    """

    #concatenating the argument columns from all dataframes to collect all possible arguments
    all_labels = pd.concat([train_df['argument'], test_df['argument'], dev_df['argument']], ignore_index=True)

    #extract unique labels
    labels = all_labels.explode().unique()

    sorted_labels = sorted([label for label in labels if label is not None]) #converting array to list, removing None value, and sorting alphabetically
    sorted_labels.append(None) #appeding None value back again
    sorted_labels_array = np.array(sorted_labels) #converting list back to array

    #moving '_' item to position 0
    labels = np.concatenate((['_'], np.delete(sorted_labels_array, np.where(sorted_labels_array == '_'))))

    #mapping the argument labels to a number
    label_map = {label: index for index, label in enumerate(labels)}
    label_map[None] = None #converting the value of None back to None

    return label_map


def map_labels_to_numbers(label_list,label_map):
    """Map a list of labels to corresponding numerical values based on a pre-defined label mapping.
    Return a list containing the mapped numerical values corresponding to the input labels.

    Parameters:
    - label_list (list): a list of labels to be mapped to numerical values.
    - label_map (dict): a dictionary mapping labels to numerical values.
    """

    mapped_labels = [label_map[label] for label in label_list if label in label_map]
    return mapped_labels


def map_labels_in_dataframe(df,label_map):
    """
    Map labels in a specified DataFrame column to numerical values based on a pre-defined label mapping.
    Add a new column to the DataFrame with the mapped labels.
    Return a new DataFrame with an additional column containing the mapped numerical labels.

    Parameters:
    - df (DataFrame): input pandas DataFrame containing a column with lists of labels.
    - label_map (dict): a dictionary mapping labels to numerical values.
    """

    #adding new column with mapped labels to the df
    df['mapped_labels'] = df['argument'].apply(lambda x: map_labels_to_numbers(x,label_map))

    return df


def tokenize_and_align_labels(tokenizer, dataset, label_all_tokens=True):
    """
    preprocess data and tokenize it using model tokenizer.
    Extract features from the data and return a datarame.

    Return Tokenized data.

    tokenizer (transformers AutoTokenizer): tokenizer of pretrained model.
    dataset (dataframe): dataframe of list of words and list of labels of each sentence.
    label_all_tokens (boolean): Taked from tutorial if True all tokens have thier own label (some words maybe splited to more than one token).
    """

    # From tutorial with some changes to work with our data, as seen on https://huggingface.co/docs/transformers/preprocessing .
    # Use 'tolist()' function to make a list of input_form column.
    tokenized_inputs = tokenizer(dataset['input_form'].tolist(), truncation=True, is_split_into_words=True, padding=True, return_tensors="pt")

    labels = []
    # Use 'tolist()' function to make a list of mapped_labels column.
    # For each mapped ARG in 'mapped_labels' column (each sentence).
    for i, label in enumerate(dataset['mapped_labels'].tolist()):
        # Get list of tokenizer word_id for current sentence (i: index of current sentence).
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # Use previous_word_idx to store word_ids of previous token.
        previous_word_idx = None
        # Empty list for store all token (subtoken) labels.
        label_ids = []
        # For each id in word_ids
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically.
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # If label in None (that mean we add this into input_form). So We set the label to -100 so they are automatically.
            elif label[word_idx] is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def get_labels_from_map(label_map):
    """
    Get a list of labels from a label map dictionary, excluding None.
    Return a list of labels.

    Parameters:
    - label_map (dict): a dictionary mapping labels to numerical values.
    """
    return [label for label in label_map.keys() if label is not None] #getting a list of labels stored as the dictionary keys

def compute_metrics(predictions, labels, label_list):
    """
    Compute evaluation metrics for Semantic Role Labeling (SRL).
    Return a dictionary with evaluation metrics.
    """
    predict = []
    gold = []

    #predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    #removing ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    for item in true_predictions:
        predict.extend(item)

    for item in true_labels:
        gold.extend(item)

    labels = set(predict+gold) #creating the set of labels
    labels=list(labels)

    report = classification_report(gold,predict,target_names=labels,output_dict=True)

    return {
           "precision": report["macro avg"]["precision"],
    "recall": report["macro avg"]["recall"],
    "f1": report["macro avg"]["f1-score"],
    "accuracy": report["accuracy"]
    }

def load_srl_model(model_checkpoint, label_list, batch_size=16):
    """
    Load a BERT transformer model for Semantic Role Labeling (SRL).
    Return a tuple containing the loaded model, its name, and training arguments.

    Parameters:
    - model_checkpoint (str): the name of the pre-trained model.
    - label_list (list): a list of labels for token classification.
    - batch_size (int): batch size for training and evaluation.
    """

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    model_name = model_checkpoint.split("/")[-1]
    task = 'srl'
    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    return model, args

def load_dataset(tokenized_dataset):
    """
    Load tokenized dataset for training or evaluation.
    Return tokenized dataset as Dataset class.

    Parameters:
    - tokenized_dataset (dict): tokenized dataset.
    """
    dict_tokenized = Dataset.from_dict(tokenized_dataset)

    return dict_tokenized

def write_predictions_to_csv(predictions, labels, label_list, file_path):
    """
    Write true predictions and true labels to a CSV file using a DataFrame.

    Parameters:
    - predictions (np array): array of true predictions.
    - labels (np array): array of gold labels.
    - label_list (list): list with possible labels for arguments
    - file_path (str): path to the CSV file to write the predictions to.
    """
    predict = []
    gold = []
    
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    for item in true_predictions:
        predict.extend(item)

    for item in true_labels:
        gold.extend(item)
    
    #creating a DataFrame with columns 'prediction' and 'gold_label'
    df = pd.DataFrame({'prediction': predict, 'gold_label': gold})
    
    #writing the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    

def compute_evaluation_metrics_from_csv(file_path):
    """
    Compute evaluation metrics (F1 score and classification report) from a CSV file containing predictions and gold labels.
    Return F1 score and classification report.

    Parameters:
    - file_path (str): path to the CSV file containing predictions and gold labels.
    """

    #reading the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    #extracting true predictions and true labels as lists
    predictions = df['prediction'].tolist()
    true_labels = df['gold_label'].tolist()

    report = classification_report(true_labels,predictions)

    return report

def print_sentences(dataset, n=20):

    for form, argument in zip(dataset.input_form[:n], dataset.argument[:n]):
        for f, a in zip(form, argument):
            if f == '[SEP]':
                print('-' * 40)
            print(f"form: {f:<15} argument: {a}")
        print('\n' + '=' * 40 + '\n')
