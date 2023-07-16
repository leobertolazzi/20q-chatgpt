import os
import time
import random
import re 
import argparse
import json
import openai
import numpy as np
from functools import wraps
from tqdm import tqdm
from collections import defaultdict
from analysis import Analysis

def retry_on_rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except openai.error.RateLimitError:
                print("Rate limit reached. Waiting for 10 seconds...")
                time.sleep(10)
    return wrapper

def openai_call(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=prompt,
        temperature=0.1,
    )
    return response.choices[0].message.content

def get_prompt(history):
    prompt = ([{'role': "system", 'content': "The user will provide you the history of a referential guessing game played by two agents called "\
                                                    "'assistant' and 'user'. Given the history of the conversation provided by the user, "\
                                                    "list the remaining candidates. You can only say 'The remaining candidates are:' followed"\
                                                    "by the list of remaining candidates"},
                {'role': "user", 'content': f"This is the game history:\n{history}\nList the remaining candidates."}])
    return prompt

def sample_histories(dialogues_raw, num):
    dialogues = [dial.strip("\n") for dial in dialogues_raw.split("******************")]
    dialogues = [[interaction for interaction in dial.split("\n")[1:] if interaction != ""] for dial in dialogues]
    dialogues = dialogues[1:]
    dialogues_length = [(id, np.sum([1 for i in dial if "questioner:" in i])) for id, dial in enumerate(dialogues)]
    
    histories_dict = {i:[] for i in range(1,9)} ## modify
    for id, length in dialogues_length:
        for i in histories_dict:
            if length > int(i):
                histories_dict[i].append(id)

    histories = []
    for key, value in histories_dict.items():
        samples = value #np.random.choice(value, size=10, replace=False)
        for sample in samples:
            num_qs = 0
            for pos, step in enumerate(dialogues[sample]):
                if "questioner:" in step:
                    num_qs += 1
                    if num_qs == int(key):
                        stop = pos+2
            histories.append((sample, dialogues[sample][:stop]))
            
    return histories

                
@retry_on_rate_limit
def generate_reference_set(game_set, num_candidates=8, log=True):

    if os.path.exists(f"../data/generation/{game_set}/guesser_annotations.json"):
        with open(f"../data/generation/{game_set}/guesser_annotations.json", "r") as f:
            previous_history = json.load(f)
    else:
        previous_history = []

    with open(f"../data/generation/{game_set}/dialogues.txt", "r") as f:
        dialogues_raw = f.read()

    histories = sample_histories(dialogues_raw, num_candidates)
    
    histories = histories[len(previous_history):]
    reference_set_history = previous_history

    for id, history in tqdm(histories):
        candidates = re.sub("aswerer: This is the list of candidates: ", "", history[0]).strip(".").split(", ")

        history = "\n".join(history)
        prompt = get_prompt(history)
        answer = openai_call(prompt)

        answer = answer[(answer.index(":")+1):]
        answer = re.sub("and", ",", answer)
        answer = re.sub("\.", "", answer)

        reference_set = [answer.strip(" ")] if "," not in answer else [item.strip(" ") for item in answer.split(",") if item != " "]

        if log:
            print("******************")
            print(history)
            print(f"Reference set = {reference_set}.")
        
        reference_set_history.append(
            {
                "dialogue_id" : int(id)+1, 
                "num_candidates" : len(candidates),
                "candidates" : candidates,
                "question_step" : len(re.findall("questioner:", history)),
                "reference_set" : reference_set
            }
        )
    
        with open(f"../data/generation/{game_set}/guesser_annotations.json", "w") as o:
            json.dump(reference_set_history, o, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_set", type=str, default="8_mcrae",
        choices=["8_mcrae", "16_mcrae", "8_gpt", "8_mcrae_stepwise", "8_wordnet"])
    parser.add_argument('--log_generation', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    with open("config.json") as f:
        config = json.load(f)

    openai.api_key = config["api_key"]

    num_candidates = int(re.sub(r"_.*", "", args.game_set))

    with open(f"../data/generation/{args.game_set}/dialogues.txt") as f:
        dialogues = f.read()
    
    with open(f"../data/generation/{args.game_set}/oracle_annotations.json") as f:
        annotations = json.load(f)

    generate_reference_set(args.game_set, num_candidates=num_candidates, log=args.log_generation)
