import os
import argparse
import re
import json
import collections
from collections import Counter
from collections import defaultdict
import nltk
import numpy as np
from scipy.stats import entropy


def preprocess_dialogues(dialogues_raw):
    dialogues_raw = dialogues_raw.split("\n******************\n")
    dialogues = []
    for dialogue_raw in dialogues_raw:
        dialogue_raw = [interaction for interaction in dialogue_raw.split("\n") if interaction != ""]
        dialogue_raw = dialogue_raw[1:]
        dialogue =  []
        for i, interaction in enumerate(dialogue_raw):
            turn = {}
            if re.match("questioner:", interaction) and "correct" not in interaction and "?" in interaction:
                try :
                    question = re.sub("questioner: ", "", interaction)
                    answer = re.sub("answerer: ", "", dialogue_raw[i+1])
                    turn['question'] = question
                    turn['answer'] = answer
                    dialogue.append(turn)
                except IndexError:
                    pass
        dialogues.append(dialogue)
    return [dialogue for dialogue in dialogues if dialogue != []]

def preprocess_annotations(annotations_raw):
    annotations = []
    for dialogue in annotations_raw:
        dialogue_ans = []
        target = dialogue["target"]
        for turn in dialogue["questions"]:
            dialogue_ans.append((target, turn["question"], turn["item_specific_answers"]))
        annotations.append(dialogue_ans)
    return annotations

def preprocess_for_eig(annotations_raw):
    dialogue_id = 1
    eig_dict = {}
    # first dict
    for dialogues in annotations_raw:
        eig_dialogue = {}
        question_id = 0
        for question in dialogues['questions']:
            answer = []
            for index, answers_for_candidates in question['item_specific_answers'].items():
                answer.append(answers_for_candidates.capitalize())
            eig_dialogue[question_id] = answer
            question_id += 1
        eig_dict[dialogue_id] = eig_dialogue
        dialogue_id += 1
    # second dict
    dialogue_iden = 0
    result_dict = {}
    for dialogue in annotations_raw:
        # error: not all candidates are included
        candidates = dialogue['candidates']
        dialogue_iden += 1
        dialogue_dict = {'objects':candidates, 'qas':[]}
        target = dialogue['target']
        for i in dialogue['questions']:
            single_answ = {
                'question': i['question'],
                'answer': i['item_specific_answers'][dialogue['target']].capitalize()
            }
            dialogue_dict['qas'].append(single_answ)
        if target in dialogue_dict['qas'][-1]['question'] or target[:-1] in dialogue_dict['qas'][-1]['question']:
            dialogue_dict['status'] = 'success'
        else:
            dialogue_dict['status'] = 'failure'  
        result_dict[dialogue_iden] = dialogue_dict

    return eig_dict, result_dict

def stepwise_guesser_annotations(dialogues):
    dialogues = [dial.strip("\n") for dial in dialogues.split("******************")]
    dialogues = [[interaction for interaction in dial.split("\n") if interaction != ""] for dial in dialogues]
    dialogues = dialogues[1:]

    reference_set_history = []

    for dial_id, dialogue in enumerate(dialogues):
        target = re.sub("target = ", "", dialogue[0])
        candidates = re.sub("answerer: This is the list of candidates: ", "", dialogue[1]).strip(".").split(", ")
        num_qs = 0
        for turn_id, interaction in enumerate(dialogue):
            if "questioner" in interaction:
                num_qs += 1
                try: 
                    reference_set = [token for token in nltk.word_tokenize(dialogue[turn_id+2]) if token in candidates]
                except IndexError:
                    if target in interaction:
                        reference_set = [target]
                    else:
                        reference_set = ["wrong"]

                reference_set_history.append(
                    {
                        "dialogue_id" : int(dial_id)+1, 
                        "num_candidates" : len(candidates),
                        "candidates" : candidates,
                        "question_step" : num_qs,
                        "reference_set" : reference_set
                    }
                )
    return reference_set_history

def guesser_vs_oracle_update(reference_set_history, oracle_annotations):
    correctness = []
    distances = defaultdict(list)

    for annotation_history in reference_set_history:
        annotations = oracle_annotations[annotation_history["dialogue_id"]-1]
        reference_set = annotations["candidates"]
        for question in annotations["questions"]:
            target_answer = question["item_specific_answers"][annotations["target"]]
            reference_set = [item for item in  question["item_specific_answers"] if question["item_specific_answers"][item] == target_answer and item in reference_set]
            if question["question_step"] == annotation_history["question_step"]:
                if set(annotation_history["reference_set"]) == set(reference_set):
                    correctness.append(question["question_step"])
                    distances[question["question_step"]].append(0)
                else:
                    distances[question["question_step"]].append(len(set(annotation_history["reference_set"]).symmetric_difference(set(reference_set))))

    return correctness, distances

class Analysis():
    def __init__(
        self,
        dialogues,
        annotations,
        game_set
        ):

        self.dialogues = dialogues
        self.annotations = annotations
        self.game_set = game_set

        self.num_dialogs = len(dialogues.split("\n******************\n"))
        self.num_candidates = len(annotations[1]['candidates'])

        self.hs_questions_pos = []
        self.cs_questions_pos = []

        self.uq_ids = []
        self.tq_ids = []
        self.cq_ids = []

    def ans_error_rate(self, save=False):
        correct = 0
        dialogues = [dial for dial in self.dialogues.split("******************\n") if dial != ""]
        for i, dialogue in enumerate(dialogues):
            full_dial = dialogue.strip('\n')
            dialogue = dialogue.strip('\n').split("\n")
            target = re.sub("target = ", "", dialogue[0])
            if (target in dialogue[-2].lower() or target[:-1] in dialogue[-2].lower()) and "yes" in dialogue[-1].lower():    
                correct += 1
            else:
                if save:
                    with open(f"../data/error_analysis/{self.game_set}/answerer_errors.txt", "a") as o:
                        o.write("____________________________________________\n")
                        o.write(f"dial_id = {i}\n")
                        o.write(full_dial+"\n")
        return 100 - ((correct / self.num_dialogs) * 100)

    def average_questions(self):
        self.questions_dist = [len(dial["questions"]) for dial in self.annotations]
        avg_q = np.mean(self.questions_dist)
        return avg_q
    
    def unnecessary_questions(self, save=False):
        unnecessary_qs = []
        num_dial_unnecessary_qs = 0
        n_questions = 0
        dialogues = [dial for dial in self.dialogues.split("******************\n") if dial != ""]

        for i, dialogue in enumerate(preprocess_annotations(self.annotations)):
            _, q, reference_set = dialogue[0]
            reference_set = reference_set.keys()
            original_reference_set = reference_set
            candidates = ", ".join(reference_set)
            unnecessary = 0
            unnecessary_list = []
            unnecessary_ids = []
            for turn_id, turn in enumerate(dialogue):
                target, question, answers = turn

                turn_yes = [item for item in answers if answers[item] == "yes"]
                overlapping = any(item in reference_set for item in turn_yes)

                is_guess = any(item in question for item in original_reference_set)
                
                if target in reference_set and len(reference_set) == 1:
                    unnecessary += 1
                    if is_guess:
                        question_type = "guess"
                    else:
                        if not overlapping and turn_yes != []:
                            question_type = "contradictory"
                        elif target not in question:
                            question_type = "trivial"

                    self.uq_ids.append((i, turn_id))
                    unnecessary_ids.append(turn_id)
                    unnecessary_list.append(f"{question} - type = {question_type}")
                    
                reference_set = [item for item in answers if answers[item] == answers[target] and item in reference_set]
            if unnecessary > 1:
                unnecessary_qs.append(unnecessary - 1)
                num_dial_unnecessary_qs += 1
                if save:
                    with open(f"../data/error_analysis/{self.game_set}/unnecessary.txt", "a") as o:
                        o.write("____________________________________________\n")
                        o.write(f"dial_id = {i}\n")
                        o.write(f"candidates = {candidates}\n")
                        o.write(f"target = {target}\n")
                        o.write(f"reference set = {reference_set}\n")
                        o.write(f"start = {min(unnecessary_ids)}/{len(dialogue)}\n")
                        for q in unnecessary_list[:-1]:
                            o.write(f"questioner: {q}\n")
            n_questions += len(dialogue)
        return unnecessary_qs, (num_dial_unnecessary_qs/len(dialogues))*100, (np.sum(unnecessary_qs)/n_questions)*100
    
    def oracle_spoilers(self, save=False):
        dialogues_raw = [dial for dial in self.dialogues.split("******************\n") if dial != '']
        dialogues = []
        for dial in dialogues_raw:
            dial = [interaction for interaction in dial.split("\n") if interaction != '']
            target = dial[0]
            target = re.sub("target = ", "", target)
            oracle_answers = [ans for ans in dial[2:-2] if "answerer:" in ans]
            dialogues.append({"target" : target, "answers" : oracle_answers})

        total = 0
        spoilers = 0

        for i, dialogue in enumerate(dialogues):
            total += 1
            check = np.sum([1 for answer in dialogue["answers"] if dialogue["target"] in answer])
            if check > 0:
                spoilers += 1
                if save:
                    with open(f"../data/error_analysis/{self.game_set}/answerer_spoilers.txt", "a") as o:
                            o.write("____________________________________________\n")
                            o.write(f"dial_id = {i}\n")
                            o.write(dialogues_raw[i])
        return (spoilers / total) * 100
    
    def contradictory_questions(self, save=False):
        n_questions = 0
        inconsistent_questions = 0
        for i, dialogue in enumerate(preprocess_annotations(self.annotations)):
            _, q, reference_set = dialogue[0]
            reference_set = reference_set.keys()
            candidates = ", ".join(reference_set)
            candidates_list = reference_set
            for turn_id, turn in enumerate(dialogue):
                target, question, answers = turn
                turn_yes = [item for item in answers if answers[item] == "yes"]
                overlapping = any(item in reference_set for item in turn_yes)
                if not overlapping and turn_yes != []:
                    inconsistent_questions += 1
                    self.cq_ids.append((i, turn_id+1))
                    if save:
                        with open(f"../data/error_analysis/{self.game_set}/contradictions.txt", "a") as o:
                            o.write("____________________________________________\n")
                            o.write(f"dial_id = {i}\n")
                            o.write(f"target = {target}\n")
                            o.write(f"candidates = {candidates}\n")
                            o.write(f"turn = {turn_id+1}/{len(dialogue)}\n")
                            o.write(f"reference set = {reference_set}\n")
                            o.write(f"question = {question}\n")
                reference_set = [item for item in answers if answers[item] == answers[target] and item in reference_set]
            n_questions += len(dialogue)
        return (inconsistent_questions/n_questions)*100
    
    def trivial_questions(self, save=False):
        n_questions = 0
        trivial_questions = 0
        for i, dialogue in enumerate(preprocess_annotations(self.annotations)):
            _, q, reference_set = dialogue[0]
            reference_set = reference_set.keys()
            candidates = ", ".join(reference_set)
            candidates_list = reference_set
            for turn_id, turn in enumerate(dialogue):
                target, question, answers = turn
                reference_set_yes = [item for item in reference_set if answers[item] == "yes"]
                all_yes = len(reference_set_yes) == len(reference_set)
                if all_yes and target not in question:
                    trivial_questions += 1
                    self.tq_ids.append((i, turn_id+1))
                    if save:
                        with open(f"../data/error_analysis/{self.game_set}/trivial.txt", "a") as o:
                            o.write("____________________________________________\n")
                            o.write(f"dial_id = {i}\n")
                            o.write(f"target = {target}\n")
                            o.write(f"candidates = {candidates}\n")
                            o.write(f"turn = {turn_id+1}/{len(dialogue)}\n")
                            o.write(f"reference set = {reference_set}\n")
                            o.write(f"question = {question}\n")
                reference_set = [item for item in answers if answers[item] == answers[target] and item in reference_set]
            n_questions += len(dialogue)
        return (trivial_questions / n_questions) * 100
    
    def questions_strategies(self):
        n_questions = 0
        constraint_seeking = 0
        pseudo_constraint_seeking = 0
        hypothesis_scanning = 0
        
        for dialogue in self.annotations:
            target = dialogue['target']
            candidates = dialogue['candidates']
            for question in dialogue['questions']:
                n_questions += 1
                previous = candidates.copy()
                c = 0
                
                constraint_condition = True
                for candidate in candidates:
                    regex = r"[^a-zA-Z]" + re.escape(candidate) + r"[^a-zA-Z]"
                    if re.search(regex, question['question'], re.IGNORECASE):
                        constraint_condition = False

                if constraint_condition == True:
                    for k, v in question['item_specific_answers'].items():
                        if k in previous and v != question['item_specific_answers'][target] and len(previous) != 2:
                            previous.remove(k)
                            c += 1

                    if c == 1:
                        pseudo = question['question']
                        constraint_condition = None
                
                if constraint_condition == None:
                    pseudo_constraint_seeking += 1
                elif constraint_condition == False:
                    hypothesis_scanning += 1
                    self.hs_questions_pos.append(question["question_step"])
                elif constraint_condition == True:
                    constraint_seeking += 1
                    self.cs_questions_pos.append(question["question_step"])

        return (hypothesis_scanning/n_questions)*100, (constraint_seeking/n_questions)*100, (pseudo_constraint_seeking/n_questions)*100
    
    def compute_eig(self):
        eig_dict, result_dict = preprocess_for_eig(self.annotations)

        few_many = {8:"many"} if len(self.annotations[1]['candidates']) == 8 else {16:"many"}

        optimal_change={
            20:[10],
            19:[9,10],
            18:[9],
            17:[8,9],
            16:[8],
            15:[7,8],
            14:[7],
            13:[6,7],
            12:[6],
            11:[5,6],
            10:[5],
            9:[4,5],
            8:[4],
            7:[3,4],
            6:[3],
            5:[2,3],
            4:[2],
            3:[1,2],
            2:[1],
            1:[0]
        }

        yes_no = ['Yes', 'No']

        stop_beforehand = False
        ig_per_turn = collections.defaultdict(lambda: collections.defaultdict(list))
        best_per_turn = collections.defaultdict(list)
        ig_per_answer_overall = collections.defaultdict(list)
        yes_is_best = collections.defaultdict(list)
        yes_in_dial = collections.defaultdict(list)
        entropy_per_turn = collections.defaultdict(lambda: collections.defaultdict())

        non_informative = []
        turn_not_inf = collections.defaultdict(int)

        average_ig_per_objs = collections.defaultdict(lambda: collections.defaultdict(list))
        average_ig_per_objs_per_turn = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        zero_ig = collections.defaultdict(lambda: collections.defaultdict(list))
        count_additional_0ig = collections.defaultdict(lambda: collections.defaultdict(list))
        answers_to_0ig_model = []
        qtype_per_objs_per_turn = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        qtype_per_objs_per_turn_list = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        average_ig_per_qtype = collections.defaultdict(list)
        ig_per_game_per_turn = collections.defaultdict(lambda : collections.defaultdict())

        non_informative_ans_turn0 = []
        non_opt_and_opt_ans_turn0 = []
        non_informative_turn_0 = []

        for game_id, game_tab in eig_dict.items():
            temp = result_dict[game_id]

            if result_dict[game_id]['status'] != '':
            # if test_games[game_id]['status'] == 'failure':

                if 'N/A' in [x['answer'] for x in result_dict[game_id]['qas']]:
                    continue

                len_qas = len(result_dict[game_id]['qas'])
                len_objs = len(result_dict[game_id]['objects'])

                previous_distr = [1 / len_objs] * len_objs
                previous_etropy = entropy(previous_distr, base=2)
                previous_valid_targets = [1] * len_objs

                for turn, answers in game_tab.items():
                    gt_ans = result_dict[game_id]['qas'][int(turn)]['answer']
                    yes_in_dial[turn].append(1 if 'Yes' == gt_ans else 0)

                    previous_distr_per_ans = {}
                    previous_entropy_per_ans = {}
                    previous_valid_targets_per_ans = {}
                    num_valid_targets_per_ans = {}

                    h = {}
                    count_ans = {}
                    for current_ans in ['Yes', 'No']:
                        valid_targets = [1 if ans == current_ans else 0 for ans in answers]
                        valid_targets = [v if previous_valid_targets[idx_v] else 0 for idx_v, v in
                                        enumerate(valid_targets)]
                        num_valid_targets = sum(valid_targets)
                        if num_valid_targets == 0:
                            h[current_ans] = 0
                            count_ans[current_ans] = 0
                            continue

                        prob = 1 / num_valid_targets
                        new_distr = []
                        for idx, v in enumerate(valid_targets):
                            new_distr.append(prob if v == 1 else 0)
                        assert (1 - sum(new_distr) < 0.0001)
                        new_entropy = entropy(new_distr, base=2)
                        h[current_ans] = new_entropy
                        count_ans[current_ans] = num_valid_targets
                        previous_distr_per_ans[current_ans] = new_distr[:]
                        previous_entropy_per_ans[current_ans] = new_entropy
                        previous_valid_targets_per_ans[current_ans] = valid_targets[:]

                    h_posterior = (count_ans['Yes'] / sum(previous_valid_targets)) * h['Yes'] + (
                            count_ans['No'] / sum(previous_valid_targets)) * h['No']
                    ig = previous_etropy - h_posterior
                    ig_per_turn['model'][int(turn)].append(ig)
                    average_ig_per_objs['model'][len_objs].append(ig)
                    average_ig_per_objs_per_turn['model'][len_objs][turn].append(ig)
                    zero_ig['model'][len_objs].append(ig == 0)
                    ig_per_game_per_turn[game_id][turn] = ig

                    if int(turn) == 0:
                        non_informative_turn_0.append(1 if ig==0 else 0)

                    if ig!=0:
                        if int(turn)==0:
                            non_opt_and_opt_ans_turn0.append(1 if gt_ans =='Yes' else 0)

                    if ig==0:
                        qtype='0_IG_q'
                        if int(turn)==0:
                            non_informative_ans_turn0.append(gt_ans)
                        # qtype_per_objs_per_turn_list[few_many[len_objs]][turn]['0_IG_q'].append(1)
                    elif sum(previous_valid_targets_per_ans[gt_ans]) in optimal_change[sum(previous_valid_targets)]:
                        qtype='optimal_q'
                    else:
                        qtype='other'

                    if len_objs in few_many:
                        qtype_per_objs_per_turn[few_many[len_objs]][turn][qtype] += 1

                    # optimal agent
                    total = sum(previous_valid_targets)
                    half = int(np.around(total / 2))
                    other_half = total - half
                    answers_otpimal = ['Yes'] * half + ['No'] * other_half
                    h_yes = 0
                    h_no = 0
                    if half > 0:
                        h_yes = (half / total) * entropy([1 / half] * half, base=2)
                    if other_half > 0:
                        h_no = (other_half / total) * entropy([1 / other_half] * other_half, base=2)
                    h_posterior_otpimal = h_yes + h_no
                    ig_per_turn['optimal'][int(turn)].append(previous_etropy - h_posterior_otpimal)
                    average_ig_per_objs['optimal'][len_objs].append(previous_etropy - h_posterior_otpimal)
                    average_ig_per_objs_per_turn['optimal'][len_objs][turn].append(previous_etropy - h_posterior_otpimal)
                    zero_ig['optimal'][len_objs].append((previous_etropy - h_posterior_otpimal) == 0)
                    #########

                    # baseline agent
                    count_yes = 1
                    count_no = sum(previous_valid_targets) - 1
                    assert count_yes + count_no == total
                    total = count_yes + count_no
                    h_yes = 0
                    h_no = 0
                    if count_yes > 0:
                        h_yes = (count_yes / total) * entropy([1 / count_yes] * count_yes, base=2)
                    if count_no > 0:
                        h_no = (count_no / total) * entropy([1 / count_no] * count_no, base=2)
                    h_posterior_baseline = h_yes + h_no
                    ig_per_turn['baseline'][int(turn)].append(previous_etropy - h_posterior_baseline)
                    average_ig_per_objs['baseline'][len_objs].append(previous_etropy - h_posterior_baseline)
                    zero_ig['baseline'][len_objs].append((previous_etropy - h_posterior_baseline) == 0)
                    #########

                    previous_distr = previous_distr_per_ans[gt_ans]

                    previous_etropy = previous_entropy_per_ans[gt_ans]
                    previous_valid_targets = previous_valid_targets_per_ans[gt_ans]
                    entropy_per_turn[game_id][turn] = previous_etropy

                    if stop_beforehand:
                        if sum(previous_valid_targets) == 1:
                            count_additional_0ig['model'][len_objs].append(len_qas-int(turn)-1)
                            for new_turn in range(int(turn)+1, len_qas):
                                answers_to_0ig_model.append(result_dict[game_id]['qas'][new_turn]['answer'])
                            break
        
        return ig_per_turn



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--game_set", type=str, default="8_mcrae",
        choices=["8_mcrae", "16_mcrae", "8_gpt", "8_mcrae_stepwise", "8_wordnet"])
    parser.add_argument('--log_errors', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    with open(f"../data/generation/{args.game_set}/dialogues.txt") as f:
        dialogues = f.read()
    
    with open(f"../data/generation/{args.game_set}/oracle_annotations.json") as f:
        annotations = json.load(f)

    if args.log_errors:
        if not os.path.exists(f"../data/error_analysis/{args.game_set}"):
            os.mkdir(f"../data/error_analysis/{args.game_set}")

    num_candidates = int(re.sub(r"_.*", "", args.game_set))

    analysis = Analysis(dialogues, annotations, args.game_set)

    model = analysis.average_questions()
    print("______________________________________________________________")
    print('STRATEGY EFFICIENCY:')
    print()
    print('Average number of questions per game')
    print('GPT\t','Optim\t','Base\t')
    print('{:.2f}\t {:.2f}\t  {:.2f}\t'.format(
        model,
        (np.log2(num_candidates) + 0.5),
        num_candidates/2
        )
    )
    unnecessary_qs, percentage_unnecessary_dial, percentage_unnecessary_qs  = analysis.unnecessary_questions(save=args.log_errors)
    print()
    print('% of games with unnecessary questions')
    print('GPT\t')
    print('{:.2f}\t'.format(
        percentage_unnecessary_dial
        )
    )
    
    print()
    print("______________________________________________________________")
    print('INFORMATIVENESS OF QUESTIONS:')
    print()
    print('HS = % hypothesis scanning')
    print('CS = % constraint seeking')
    print()
    print('  \t GPT\t Optim\t Base\t ')
    hypothesis_scanning, constraint_seeking, pseudo_constraint_seeking = analysis.questions_strategies()
    print('HS\t {:.2f}\t {:.2f}\t {:.2f}\t '.format(
        pseudo_constraint_seeking + hypothesis_scanning,
        (1.5 / (np.log2(num_candidates) + 0.5)) * 100,
        100
        )
    )
    print('CS\t {:.2f}\t {:.2f}\t {:.2f}\t '.format(
        constraint_seeking,
        ( (np.log2(num_candidates) - 1) / (np.log2(num_candidates) + 0.5)) * 100,
        0
        )
    )
    print()
    print('Expected Information Gain per turn')
    print("Turn\t","GPT\t","Optim\t","Base\t")
    ig_per_turn = analysis.compute_eig()
    turns = 8 if num_candidates == 8 else 10
    ig_per_turn = {
    i : (np.mean(ig_per_turn["model"][i]), np.mean(ig_per_turn["optimal"][i]), np.mean(ig_per_turn["baseline"][i])) 
    for i in range(turns)
    }
    for i in range(turns):
        model, optimal, baseline =  ig_per_turn[i]
        print("{}\t {:.2f}\t {:.2f}\t {:.2f}\t ".format(i+1, model, optimal, baseline))

    print("______________________________________________________________")
    print('ERROR ANALYSIS:')
    print()
    print('ER = Error rate')
    print('SP = % games with spoiler answers by oracle')
    print('CQ = % contradictory questions across all games')
    print('TQ = % trivial questions across all games')
    print()
    er = analysis.ans_error_rate(save=args.log_errors)
    sp = analysis.oracle_spoilers(save=args.log_errors)
    cq = analysis.contradictory_questions(save=args.log_errors)
    tq = analysis.trivial_questions(save=args.log_errors)
    print('ER\t SP\t CQ\t','TQ\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}'.format(er, sp, cq, tq))
    
