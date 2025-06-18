import numpy as  np
from collections import Counter,defaultdict

def info_entropy(examples: list[dict], attribute: str) -> float:
    n = len(examples)
    info = 0
    cnts = Counter( ex[attribute] for ex in examples )
    for cnt in cnts.values():
        p = cnt/n
        info -= p*np.log2(p)
    return info


def info_gain(examples: list[dict], attribute: str, target_attr: str) -> float:
    Info_AT = 0
    Info_T = info_entropy(examples,target_attr)
    subsets = defaultdict(list)
    for ex in examples:
        subsets[ ex[attribute] ].append(ex)
    for subset in subsets.values():
        p = len(subset) / len(examples)
        Info_AT += p * info_entropy(subset,target_attr)
    gain = Info_T - Info_AT
    return gain

def majority(examples: list[dict], target_attr: str) -> str:
    classes = Counter( ex[target_attr] for ex in examples )
    return classes.most_common(1)[0][0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    # Your code here
    if not examples:
        return None
    target_class = [ ex[target_attr] for ex in examples]
    if len(set(target_class)) == 1:
        return target_class[0]
    if not attributes:
        return majority(examples,target_attr)
    
    gain = [info_gain(examples,attribute,target_attr) for attribute in attributes ]
    best_attribute = attributes[np.argmax(gain)]
    decision_tree = {best_attribute:{}}

    sub_group = defaultdict(list)
    for ex in examples:
        sub_group[ex[best_attribute]].append(ex)

    remaining_attributes = [attribute for attribute in attributes if attribute != best_attribute]
    for key,value in sub_group.items():
        sub_tree = learn_decision_tree(value,remaining_attributes,target_attr)
        decision_tree[best_attribute][key] = sub_tree

    return decision_tree

# print(entropy([ {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'}, \
# 			    {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},\
# 				{'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}, \
# 				{'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},\
# 				{'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\
# 				{'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'}, \
# 				{'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'}, \
# 				{'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'} ], \
# 				'PlayTennis'))

print(learn_decision_tree([ {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'}, \
			    {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},\
				{'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}, \
				{'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},\
				{'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\
				{'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'}, \
				{'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'}, \
				{'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'} ], \
				['Outlook', 'Wind'], 'PlayTennis'))





