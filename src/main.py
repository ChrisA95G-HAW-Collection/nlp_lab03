import datasets
from collections import defaultdict
from preprocessing import add_stop_token_batch, filter_short_sentence_batch, get_rares, replace_rares_batch
from ngram import estimate_unigram, unigram_sentence_logp, perplexity, estimate_bigram, estimate_bigram_smoothed, bigram_sentence_logp


def main():
    # Dataloading step
    #! Trust remote code sounds a bit unwise but i guess i have to trust
    pre_procress_dataset = datasets.load_dataset('ptb_text_only', split='train', trust_remote_code=True)

    # Preprocessing step
    dataset_stop_token = pre_procress_dataset.map(
        add_stop_token_batch,
        batched=True,
        fn_kwargs={'sentence_column': 'sentence'} 
    )

    dataset = dataset_stop_token.filter(
        filter_short_sentence_batch,
        batched=True,
        fn_kwargs={'sentence_column': 'sentence'}
    )

    dataset_split = dataset.train_test_split(test_size=0.2, shuffle=True)

    #! Part one. Unigram stuff:
    # Model before rare removal
    model = estimate_unigram(dataset_split['train']['sentence'])

    print('Part one. Unigram:')
    print('Log of test sentence1: ',unigram_sentence_logp("the the the <stop>", model))
    print('Log of test sentence2: ',unigram_sentence_logp("i love computer science <stop>", model), "\n")

    print('Model Performance before rare removal: ',perplexity(dataset_split['test'], model), "\n")
    
    rares = get_rares(5, dataset_split)

    dataset_split['train'] = dataset_split['train'].map(
        replace_rares_batch,
        batched=True,
        fn_kwargs={'rares': rares}
    )

    dataset_split['test'] = dataset_split['test'].map(
        replace_rares_batch,
        batched=True,
        fn_kwargs={'rares': rares} 
    )

    model = estimate_unigram(dataset_split['train']['sentence'])
    print("\n",'Model Performance after rare removal: ', perplexity(dataset_split['test'], model))

    #! Part two. Bigram stuff:
    unigram_model, bigram_model = estimate_bigram(dataset_split['train']['sentence'])

    print("\n Part two. Bigram:")
    print('Log of test sentence1: ', bigram_sentence_logp("the the the <stop>", unigram_model, bigram_model))
    print('Log of test sentence2: ', bigram_sentence_logp("i love computer science <stop>", unigram_model, bigram_model), "\n")

    zero_prob_counter = 0
    for sentence in dataset_split['test']['sentence']:
        prob = bigram_sentence_logp(sentence, unigram_model, bigram_model)
        if prob == float('-inf'):
            zero_prob_counter += 1
    
    print(f"There are: {zero_prob_counter}/{len(dataset_split["test"]['sentence'])} sentences with a probability of 0 in the test set! \n")

    #! Bigram with Laplac smoothing:

    unigram_model, bigram_model_smoothed = estimate_bigram_smoothed(dataset_split['train']['sentence'], alpha=0.1)

    print("Bigram with Laplac smoothing:")
    print('Log of test sentence1: ', bigram_sentence_logp("the the the <stop>", unigram_model, bigram_model_smoothed))
    print('Log of test sentence2: ', bigram_sentence_logp("i love computer science <stop>", unigram_model, bigram_model_smoothed), "\n")

    print("\n",'Model Performance after with laplac smoothing: ', perplexity(dataset_split['test'], unigram_model, bigram_model_smoothed))





if __name__ == "__main__":
    main()
