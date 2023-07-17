## Reproduce experiments from scratch

First, new dialogues have to be generated, then new annotations and finally a linguistic analysis on the new data can be performed. The generation of the new data is based on the same game sets we used in our paper.

### Clear directories

Before generating new data:

1. Empty the `data/generation` directory;
2. Empty also the `data/error_analysis` directory;
3. Insert OpenAI API key in `config.json`.

### Dialogues generation

To generate dialogues run:

```
python generate_dialogues.py --game_set {8-mcrae, 16-mcrae, 8-gpt, 8_mcrae_stepwise, 8_wordnet}
```

### Oracle annotation

To produce the oracle's annotation for the generated dialogues: 

```
python generate_oracle_annotations.py --game_set {8-mcrae, 16-mcrae, 8-gpt, 8_mcrae_stepwise, 8_wordnet}
```

### Guesser annotation


To produce the guesser's annotation for the generated dialogues: 

```
python generate_guesser_annotations.py --game_set {8-mcrae, 16-mcrae, 8-gpt, 8_mcrae_stepwise, 8_wordnet}
```

### Analysis

To get the analysis for the generated data, run: 

```
python analysis.py --game_set {8-mcrae, 16-mcrae, 8-gpt, 8_mcrae_stepwise, 8_wordnet}
```

The optional `log_error` argument creates files with the linguistic analysis in `data/error_analysis` 
