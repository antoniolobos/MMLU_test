import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

choices = ["A", "B", "C", "D"]


subjects = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions'
    ]


subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],}


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    """
    For the first subject:
    s = ' abstract algebra'
    """
    return s


def format_example(set, idx, include_answer=True):
    """
    This function's usage:
    Given a dataframe and the id of one item, generate the prompt for this item
    """
    prompt = set[idx]['question']
    k = len(set[idx]['choices'])
    """
    For the first subject, first round:
    prompt = 'Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.'
    (this prompt is the question of this item)
    k = 4  
    (k is the number of answers)
    """
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], set[idx]['choices'][j])
    """
    Add the four choices of this item
    """
    prompt += "\nAnswer:"
    """
    Add "Answer" to the prompt
    prompt = 'Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:'
    """
    if include_answer:
        prompt += " {}\n\n".format(choices[set[idx]['answer']])
    return prompt


def gen_prompt(dev_set, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    """
    For the first subject:
    prompt = 'The following are multiple choice questions (with answers) about  abstract algebra.\n\n'
    """
    if k == -1:
        k = dev_set.num_rows
    for i in range(k):
        prompt += format_example(dev_set, i)
    """
    For the first prompt:
    prompt = 'The following are multiple choice questions (with answers) about  abstract algebra.\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B ...'
    """
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_set, test_set):
    cors = []
    all_probs = []
    """
    cors would store the True/False of each question item in this subject
    all_probs would store the probabilities of A/B/C/D of each question item in this subject
    """
    
    for i in range(test_set.num_rows):
        # get prompt and make sure it fits
        """
        For the first subject:
        test_df.shape[0] = 100
        It's the loop that traverses all the question items in a single subject
        """
        k = args.num_fewshot
        prompt_end = format_example(test_set, i, include_answer=False)
        """
        (format_example() function can generate the prompt for a question item)
        prompt_end = 'Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:'
        (prompt_end is the prompt of a single question item in the test set)
        """
        train_prompt = gen_prompt(dev_set, subject, k)
        """
        (gen_prompt() function can generate the start sentence and few_shot examples for a question item)
        train_prompt = 'The following are multiple choice questions (with answers) about  abstract algebra.\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B ...'
        (train_prompt is the few_shot context of the a single question item)
        """
        prompt = train_prompt + prompt_end
        """
        prompt is the completed prompt of a single question item
        """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)#.cuda()
        """
        input_ids.shape = [1,367]
        """

        while input_ids.shape[-1] > 2048:   # in case the prompt is too long, try to cut down the number of few-shot examples
            k -= 1
            train_prompt = gen_prompt(dev_set, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)#.cuda()

        label = choices[test_set[i]['answer']]
        """
        label = 'B'
        It's the true label of the question
        """

        logits = model(                 
            input_ids=input_ids,
        ).logits[:,-1].flatten()
        """
        (original)logits.shape = [1,367,32000]
        logits.shape = [32000]
        It's the size of the vocab
        """

        # fix the error caused by torch.float16 and torch.bfloat16. the solution is arbitrary, but considering that `probs` is not used later so I just do this.
        try:
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[-1]],
                            logits[tokenizer("B").input_ids[-1]],
                            logits[tokenizer("C").input_ids[-1]],
                            logits[tokenizer("D").input_ids[-1]],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )
        except:
            probs = torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[-1]],
                            logits[tokenizer("B").input_ids[-1]],
                            logits[tokenizer("C").input_ids[-1]],
                            logits[tokenizer("D").input_ids[-1]],
                        ]
            ).detach().cpu().to(torch.float32).numpy()

        """
        probs = array([0.12568386, 0.49708933, 0.23480837, 0.14241847], dtype=float32)
        It's the probability of the model predicting the choice as A/B/C/D
        """
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    """
    cors stores the True/False of each question item in this subject
    all_probs stores the probabilities of A/B/C/D of each question item in this subject
    acc indicates the average acc of this subject
    """
    return cors, acc, all_probs


def main(args):

    # if use model from a model file
    if isinstance(args.model, str):
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, add_bos_token=False, model_max_length=4096,padding_side="right",trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model,  torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)
    # if use an imported model(maybe a quantized model)
    elif isinstance(args.model, PreTrainedModel):
        tokenizer = args.tokenizer
        model = args.model
    else:
        print("[ERROR] Not a model file's path or an imported pretrained model!")
        exit()



    all_cors = []
    """
    all_cors would store the True/False of all the questions
    """
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    """
    subcat_cors = {     # one subcat_cor could include more than one subject, it would store the True/False of all the questions of the subjects in this subcat
        'math':[],
        'health':[],
        'physics':[],
        'business':[],
        ...
    } 
    cat_cors = {
        'STEM': [],                 # one cat_cor could include more than one subcat_cor, it would store the True/False of all the questions of the subjects in this cat
        'humanities': [], 
        'social sciences': [], 
        'other (business, hea...th, misc.)': [] 
    }

    """

    all_acc = []
    for subject in subjects:
        """
        traverse all the subjects cais/mmlu
        """
        dev_set = load_dataset('cais/mmlu', subject, split='dev')
        test_set = load_dataset('cais/mmlu', subject, split='test')
        """
        For the first subject:
        subject = 'abstract_algebra'
        dev_df.shape = [5,6]
        test_df = [100,6]
        """

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_set, test_set)
        """
        cors stores the True/False of each question item in this subject
        all_probs stores the probabilities of A/B/C/D of each question item in this subject
        acc indicates the average acc of this subject
        """        
        subcats = subcategories[subject]
        """
        For the first subject:
        subcats = ['math']
        It's the subcat of this subject
        """
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)
        all_acc.append((subject, acc))

        

    """
    After traversing all the subjects:
    """
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
    """
    Calculate the average acc of each subcat
    """

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    """
    Calculate the average acc of each cat
    """
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    """
    Calculate the average acc of all the questions
    """

    # save results
    if args.save_results:
        print('Saving results for mmlu ...')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, args.save_name) + '.md', 'w') as f:
            f.write('# Evaluation result for mmlu\n')
            f.write('metric: acc\n')
            f.write('## Result for each subject\n')
            for result in all_acc:  # result: (subject, acc)
                f.write(f'+ {result[0]}: {result[1]}\n')
            f.write('## Average result for each sub-category\n')
            for subcat in subcat_cors:
                subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
                f.write(f'+ {subcat}: {subcat_acc}\n')
            f.write('## Average result for each category\n')
            for cat in cat_cors:
                cat_acc = np.mean(np.concatenate(cat_cors[cat]))
                f.write(f'+ {cat}: {cat_acc}\n')
            f.write('## Average result for all the subjects\n')
            weighted_acc = np.mean(np.concatenate(all_cors))
            f.write(f'+ all: {weighted_acc}\n')
        print('Done.')

# for test_all.py's call
def mmlu_entry(args):
    main(args)


# can also run the script directly   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_fewshot", "-k", type=int, default=5)  # k-shot
    parser.add_argument("--save_dir", "-s", type=str, default="./mmlu_results")
    parser.add_argument(
        "--model",
        type=str,
        default="lmsys/vicuna-7b-v1.5"
    )
    args = parser.parse_args()
    main(args)
