# Reproduce experiments from scratch

The following is a description of the steps necesassary to reproduce the experiment from scratch. In few words, first, new dialogues have to be generated, then new annotations and finally a linguistic analysis on the new data can be performed. The generation of the new data is based on the same game sets we used in our paper.

## 0. Clear directories

Before generating new data, follow these steps:

1. Empty the `data/generation` directory;
2. Empty also the `data/error_analysis` directory;
3. Insert OPENAI API key in `config.json`.

## 1. Dialogues generation

To generate dialogues run:

```
python generate_dialogues.py --game_set {8-mcrae, 16-mcrae, 8-gpt ...}
```

## 2. Oracle annotation

## 3. Guesser annotation

## 4. Linguistic analysis

