from collections import defaultdict

def add_stop_token_batch(batch, sentence_column, stop_token="<stop>"):
    """Adds stop token to input dataset"""
    new_texts = [sentence + ' ' + stop_token for sentence in batch[sentence_column]]

    return {sentence_column: new_texts}


def filter_short_sentence_batch(batch, sentence_column, min_words: int=3):
    """Filters out any sentence with less than min_words"""
    new_text = [len(sentence.split()) >= min_words for sentence in batch[sentence_column]]
    return new_text

def get_rares(threshold, dataset):
    """
    This is much better than the suggested way in the assignment,
    The assignment didnt make sense at all, why the fuck should i 
    implement a remove_rares functuin that takes in two datasets?
    """
    print(f'Removing low-frequency words (count < {threshold})...')
    counter_dict = defaultdict(int)
    for sentence in dataset['train']['sentence']:
        for word in sentence.split():
                counter_dict[word] += 1

    return {word for word, count in counter_dict.items() if count < threshold}


def replace_rares_batch(batch, rares):
    """Replaces words present in rares with '<unk>'."""
    new_sentences = []
    for sentence in batch['sentence']:
        new_sentence_parts = []
        for word in sentence.split():
            if word in rares:
                new_sentence_parts.append('<unk>')
            else:
                new_sentence_parts.append(word)
        new_sentences.append(' '.join(new_sentence_parts))
    return {'sentence': new_sentences}
