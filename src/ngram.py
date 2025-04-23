import math
from tqdm import tqdm
from collections import defaultdict

def word_count(dataset):
    """
    This is just a littel helper function for unigram
    and bigram
    """
    total_words = 0
    #! i didnt know that int() returns 0 this could be good to keep in minde 
    #! BUT HERE WE NEED INT WIHTOUT () THIS IS VERY CONFUSING!!!
    model_dict = defaultdict(int)
    for sentence in dataset:
        for word in sentence.split():
            model_dict[word] += 1
            total_words += 1
    
    return total_words, model_dict


def bigram_helper(dataset):
    """
    Helper function, added later so that there is no 
    code repition. It handels the first part of the bigram functions:
    - Getting data from word_count
    - Counting word pairs in nested defaultdict
    """

    total_words, unigram_model_dict = word_count(dataset)
    # First part is just unigram
    unigram_model = {}
    for key in unigram_model_dict.keys():
        unigram_model[key] = (unigram_model_dict[key] / total_words)
    

    # Second part is bigram
    #! This is very helpfull to store the count of word pairs!
    bigram_dict = defaultdict(lambda: defaultdict(int))
    bigram_model = defaultdict(lambda: defaultdict(int))

    # Counting:
    #! This is the good part! We use zip to generate our word pairs
    #! then we iterate over the word pairs from zip and count them in 
    #! the nested defaultdict (bigram_dict)
    for sentence in dataset:
        split = sentence.split()
        word_pairs = zip(split, split[1:])

        for w_iminus1, w_i in word_pairs:
            bigram_dict[w_iminus1][w_i] += 1

    return unigram_model_dict, unigram_model, bigram_dict, bigram_model

def estimate_unigram(train_set):
    """
    Basic unigram model that simply counts all words
    and returns them as keys with there probalbility as values
    """
    total_words, model_dict = word_count(train_set)
    model = {}
    for key in model_dict.keys():
        model[key] = (model_dict[key] / total_words)
    
    return model


def estimate_bigram(dataset):
    """
    This gernerates a unigram and bigram model
    dont ask me why we are supposed to return both models.
    Makes no sense to me but prof is king i guess.
    Ok it does make sense when we go one step further :D.
    I later refractured some code from here to the helper
    function bigram_helper, this makes it look a lot cleaner to me!
    """

    unigram_model_dict, unigram_model, bigram_dict, bigram_model = bigram_helper(dataset) 
    # Generating Model:
    # This was a bit harder to wrap my brain around. 
    # The nested defaultdict is not the easyest strucktur to visiulalize

    #! First we loop thru the "outer" dict, to count up the w_i-1 (previous word)
    for w_iminus1 in bigram_dict:
        count_w_iminus1 = unigram_model_dict[w_iminus1]

        # Check for Zero devision just in case
        if count_w_iminus1 > 0:
            #! Then we loop thru the "inner" dict, to count up the word pairs
            for w_i in bigram_dict[w_iminus1]:
                pair_count = bigram_dict[w_iminus1][w_i] 

                # extra step with proba. because it is better to understand this way
                probability = pair_count / count_w_iminus1
                bigram_model[w_iminus1][w_i] = probability

    return unigram_model, bigram_model


def estimate_bigram_smoothed(dataset, alpha: float):
    """
    There is not much to tell here.
    Args:
    - dataset
    - alpha 

    returns:
    - smoothed_bigram_model
    """
    unigram_model_dict, unigram_model, bigram_dict, smoothed_bigram_model = bigram_helper(dataset)
    
    # We need the count of all words in the TRAINIG set (NOT the entire dataset) it think?
    _v_ = len(unigram_model_dict) # _v_ is suppost to represent |V|.

    #! We loop thru all words in our vocab:
    for w_iminus1 in tqdm(unigram_model_dict, desc="Calculating probabilitys..."):
        count_w_iminus1 = unigram_model_dict[w_iminus1]
        denominator = count_w_iminus1 + alpha * _v_
        #! Then we need to loop thru the entire vocab again:
        for w_i in unigram_model_dict:
            # We use .get here to get a default value IF the pair does not exist!
            count_word_pairs = bigram_dict[w_iminus1].get(w_i, 0)
            numerator = count_word_pairs + alpha

            smoothed_porbability = numerator / denominator
            smoothed_bigram_model[w_iminus1][w_i] = smoothed_porbability
    
    return unigram_model, smoothed_bigram_model


def unigram_sentence_logp(sentence, model):
    """ 
    This is part of the corss-entropy formula from the lecture/assignment,
    we look at each word of a sentence and calculate the log
    and add them up to get the probability of the sentence.
    If we dont know a word -> Prob = 0. the log would be -inf!
    """
    total_log = 0.0 #! Dont forget that we need float here!

    for word in sentence.split():
        if word in model:
            total_log += (math.log2(model[word])) #! I use -log2 here because we use -log2 in the cross-entropy formula?
        else:
            return float('-inf')
    
    return total_log

def bigram_sentence_logp(sentence, unigram_model, bigram_model):
    """
    We removed short sentences so no need to worry about empty sentences here! log2(0) and so on...
    Important Note: Because many plausible word pairs might not appear in even a large training set,
    this function (using the unsmoothed bigram model) is likely to return float('-inf') very often!
    """

    sentence = sentence.split()

    total_log = 0.0
    w1 = sentence[0]

    #! This is a very important step! I am prone to just over read the P(w1) that
    #! comes before the product in the formula
    if w1 not in unigram_model:
        #This means p_w1 = 0 and therefor:
        return float('-inf')
    else:
        p_w1 = unigram_model[w1]
        total_log += (math.log2(p_w1)) #we use negativ here because of negativ log likelihood, rightttt???
    
    word_pairs = zip(sentence, sentence[1:])
    
    # This is again a bit harder to wrap my brain around.
    for w_iminus1, w_i in word_pairs:
        if w_iminus1 in bigram_model and w_i in bigram_model[w_iminus1]:
            # Pair exists, get probability
            p_cond = bigram_model[w_iminus1][w_i]
            total_log += (math.log2(p_cond)) # Add the (negative) log prob #! DO WE NEED MINUS HERE OR NOT?
        else:
            # Pair not found in model, P=0 for sentence
            return float('-inf')
        
    return total_log


def perplexity(dataset, unigram_model, bigram_model=None):
    """
    This is the Perplexity formula from the lecture/assignment.
    We now add up the log2 of each sentence in the dataset(test set)
    and then we just divide the log with the number of words (cross-entropy).
    Then we calculate and return the perpelecity by raising 2 to the power of the cross-entropy.

    NOTE: Refractured the code to handel both unigram and bigram models.
    """

    dataset_log = 0.0
    dataset_words = 0

    for sentence in dataset['sentence']:
        if bigram_model:
            sentence_log = bigram_sentence_logp(sentence, unigram_model, bigram_model)
        else:
            sentence_log = unigram_sentence_logp(sentence, unigram_model)
        
        if sentence_log == float('-inf'): #! this is here to handle cases where the FIRST word might be unknown!
                continue
        dataset_log += sentence_log
        dataset_words += len(sentence.split())

    if dataset_words == 0:
        raise Exception("There are no valid sentences!")
    
    h = -(dataset_log / dataset_words)
    perplexity = 2**h

    return perplexity