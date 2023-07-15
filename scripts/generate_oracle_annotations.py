from functools import wraps
import os
import re
import json
from tqdm import tqdm
import time
import openai
import argparse

def read_dialogues(path):
    with open(path) as f:
        dialogues = f.read().split("******************")
    dialogues = [dialogue.split("\n")[1:] for dialogue in dialogues if dialogue != ""]
    return dialogues

#Call n.1 for the Oracle
def external_oracle_call(target, question):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': "system",
                   'content': "You are playing an interactive game with the user, in which you are assigned one item from a list of candidates." \
                              "The user will have to guess which one it is by asking yes/no questions, and " \
                              "you have to respond to each question only with 'yes' or 'no'." \
                              "If the user correctly guesses your assigned item, respond with 'Yes! That's correct.'." \
                              f"The item assigned to you is {target}."},
                  {'role': "user", 'content': f'{question}'}],
        temperature=0.1,
        max_tokens=3
    )

    return {'role': response.choices[0].message.role, 'content': response.choices[0].message.content}

#Call n.2 for the Oracle, where it is forced to answer 'Yes' or 'No'
def reinforce_oracle_call(target, question):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': "system",
                   'content': "You are playing an interactive game with the user, in which you are assigned one item from a list of candidates." \
                              "The user will have to guess which one it is by asking yes/no questions, and " \
                              "you have to respond to each question only with 'yes' or 'no'." \
                              "If the user correctly guesses your assigned item, respond with 'Yes! That's correct.'." \
                              f"The item assigned to you is {target}."},
                  {'role': "user", 'content': f"{question} Please only answer 'Yes' or 'No'"}],
        temperature=0.1,
    )

    return {'role': response.choices[0].message.role, 'content': response.choices[0].message.content}

#Call n.3 for the Oracle, where it is forced to answer 'Yes' or 'No' with a simplified prompt
def reinforce2_oracle_call(target, question):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': "system", 'content': "You are playing an interactive game with the user," \
                                                "you have to answer each question about your item only with 'yes' or 'no'." \
                                                f"\nYour item is {target}."},
                  {'role': "user", 'content': f"{question} Please only answer 'Yes' or 'No'"}],
        temperature=0.1,
    )

    return {'role': response.choices[0].message.role, 'content': response.choices[0].message.content}

#Wrapper to deal with ChatGTP problems
def retry_on_rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError):
                print("ChatGPT problems. Waiting for 10 seconds...")
                time.sleep(10)
    return wrapper

@retry_on_rate_limit
def get_complete_answers(dialogues, annotation_file_path):
    #if an annotation file exists, start from the last annotation
    if os.path.exists(annotation_file_path):
        with open(annotation_file_path, 'r') as anguissa:
            annotations = json.load(anguissa)
            annotations = [annotations] if isinstance(annotations, dict) else annotations
    #if an annotation file does not exists, create a new one
    else:
        annotations = []
        for n_dial in range(1, len(dialogues)+1):
            annotation = {
                'annotation_status': 'ongoing',
                'target': str(),
                'candidates': [],
                'dialogue_id': n_dial,
                'questions': []
            }
            annotations.append(annotation)

    #last dialogue in the uncompleted annotation file
    for annotation in annotations:
        if 'annotation_status' in annotation:
            annotation_point = annotation['dialogue_id']
            break

    for i, dialogue in enumerate(tqdm(dialogues[annotation_point-1:])):

        target = re.sub("target = ", "", dialogue[0])
        candidates = re.sub("answerer: This is the list of candidates: ", "", dialogue[1]).strip(".").split(", ")
        questions = [re.sub("questioner: ", "", interaction) for interaction in dialogue if "questioner" in interaction]
        answers_per_target = [re.sub("answerer: ", "", interaction) for interaction in dialogue if
                   "answerer" in interaction and 'This is the list of candidates:' not in interaction]

        dialogue_dict = {
            'annotation_status' : "ongoing",
            'target': target,
            'candidates': candidates,
            'dialogue_id': annotation_point,
            'questions': annotations[annotation_point-1]['questions'] if annotations else [],
        }

        for n, question in enumerate(tqdm(questions[dialogue_dict['questions'][-1]['question_step'] if annotations and dialogue_dict['questions'] else 0:])):

            item_specific_answers = {}

            for candidate_as_target in candidates:

                # select the dialogue answers_per_target for the target item
                if candidate_as_target == target:
                    oracle_output = answers_per_target[dialogue_dict['questions'][-1]['question_step'] if annotations and dialogue_dict['questions'] else 0].lower()

                # generate external oracle outputs for the other candidates
                else:
                    oracle_output = external_oracle_call(candidate_as_target, question)

                    # Regenerate the answer if both 'yes' and 'no' are present
                    counter_max = 0
                    while 'yes' not in oracle_output['content'].lower() and 'no' not in oracle_output['content'].lower():
                        if counter_max <= 2:
                            oracle_output = reinforce_oracle_call(candidate_as_target, question)
                            counter_max += 1
                        elif counter_max > 2 and counter_max < 5:
                            oracle_output = reinforce2_oracle_call(candidate_as_target, question)
                            counter_max += 1
                        else:
                            with open('oracle_unable.txt', 'a+') as t:
                                t.write(f"question_id: {dialogue_dict['dialogue_id']}; question_step: {dialogue_dict['questions'][-1]['question_step']}; candidate target: {candidate_as_target}; question: {question}; Oracle answer: {oracle_output['content']}")
                                t.write('\n')
                            oracle_output['content'] = 'N/A'

                            break

                    oracle_output = oracle_output['content'].lower()
                    #print(oracle_output)

                if "yes" in oracle_output:
                    item_specific_answers[candidate_as_target] = "yes"
                elif "no" in oracle_output:
                    item_specific_answers[candidate_as_target] = "no"
                else:
                    item_specific_answers[candidate_as_target] = 'no'

            assert (len(item_specific_answers.values()) == len(candidates))

            question_dict = {
                'question_step': dialogue_dict['questions'][-1]['question_step'] + 1 if annotations and dialogue_dict['questions'] else 1,
                'question': question,
                'item_specific_answers': item_specific_answers,
            }

            dialogue_dict['questions'].append(question_dict)

            #if the dialogue is completed, delete the annotation_status and go to the next dialogue
            if len(dialogue_dict['questions']) == len(questions):
                del dialogue_dict['annotation_status']
                del annotations[dialogue_dict['dialogue_id']-1]['annotation_status']
                annotation_point += 1

            annotations[dialogue_dict['dialogue_id']-1].update(dialogue_dict)

            with open(annotation_file_path, "w") as data:
                json.dump(annotations, data, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--game_set", type=str, default="8_mcrae",
        choices=["8_mcrae", "16_mcrae", "8_gpt", "8_mcrae_stepwise", "8_wordnet"])
    args = parser.parse_args()

    with open("config.json") as f:
        config = json.load(f)

    openai.api_key = config["api_key"]

    num_candidates = int(re.sub(r"_.*", "", args.game_set))

    dialogues = read_dialogues(f'../data/generation/{args.game_set}/dialogues.txt')

    get_complete_answers(dialogues, f'../data/generation/{args.game_set}/oracle_annotations.json')
