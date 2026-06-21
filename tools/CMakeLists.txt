add_executable(preprocess preprocess.c)
target_link_libraries(preprocess text_parser)

# train_bpe — read a text file, train a BPE tokenizer, save it.
add_executable(train_bpe train_bpe.c)
target_link_libraries(train_bpe bpe)

# tokenize — load a tokenizer, encode a text file to a binary token file.
add_executable(tokenize tokenize.c)
target_link_libraries(tokenize bpe)

# generate — load a trained checkpoint + tokenizer and autoregressively sample
# text from a prompt (Task 13). Needs `gpt` (the model + gpt_load) and `bpe`
# (encode the prompt / decode the output). CPU-only inference.
add_executable(generate generate.c)
target_link_libraries(generate gpt bpe)
