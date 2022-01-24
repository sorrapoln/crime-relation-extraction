import pandas as pd
import numpy as np
import re
import random
import itertools
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences
from pythainlp import word_tokenize, subword_tokenize, syllable_tokenize
from sklearn.model_selection import train_test_split

def read_raw_text(filename):

    with open(filename, 'r', encoding = 'utf-8') as file:

        document = file.read()

    return document

def add_no_relation(df):

    token_df = df[df[1].str.contains('T')]
    relation_df = df[df[1].str.contains('R')]
    relation_df[3] = relation_df[2].str.split(' ')

    relation_dict = relation_df[3].apply(lambda x : dict([((x[1][5:], x[2][5:]), x[0])]))
    relation_df.drop(3, axis = 1, inplace = True)
    token_comb = list(itertools.combinations(token_df[1].tolist(), 2))
    random.shuffle(token_comb)

    no_rel_sample = 2
    c = 0
    recent_n_relation = relation_df.shape[0] + 1

    for i in range(len(token_comb)):
        if i not in relation_dict.keys():

            tok_1 = token_comb[i][0]
            tok_2 = token_comb[i][1]
            relation_df = relation_df.append({1 : f'R{recent_n_relation}',
                                              2 : f'no_relation Arg1:{tok_1} Arg2:{tok_2}'},
                                              ignore_index = True)
            c += 1
            recent_n_relation += 1

        if c== 2:
            break

    df = pd.concat([token_df, relation_df])

    return df





def read_ann_file(filename): #filename e.g. 01_nut.a/xxaa.ann

    PATH = 'data/rel_data_annotated/'

    document = read_raw_text(PATH + filename[:-4] + '.txt')
    df = pd.read_csv(PATH + filename, sep='^([^\s]*)\s', engine='python', header=None).drop(0, axis=1)

    df = add_no_relation(df)

    token_df = df[df[1].str.contains('T')]
    relation_df = df[df[1].str.contains('R')]

    list_tokens = []
    list_relations = []

    for index, row in token_df.iterrows():

        text = re.findall('\t.*', row[2])[0][1:]
        entityLabel, start, end = re.findall('.*\t', row[2])[0][:-1].split(' ')
        dict_token = {'text' : text,
                      'start' : int(start),
                      'end' : int(end),
                      'entityLabel' : entityLabel}

        list_tokens.append(dict_token)

    for index, row in relation_df.iterrows():

        relationLabel, token_id_1, token_id_2 = row[2].split(' ')
        token_id_1, token_id_2 = token_id_1[5:], token_id_2[5:]

        _, start_1, __ = re.findall('.*\t', token_df[token_df[1] == token_id_1].iloc[0][2])[0][:-1].split()
        _, start_2, __ = re.findall('.*\t', token_df[token_df[1] == token_id_2].iloc[0][2])[0][:-1].split()

        dict_relation = {'child' : int(start_2),
                         'head' : int(start_1),
                         'relationLabel' : relationLabel}

        list_relations.append(dict_relation)

    dict_ann = {'document' : document,
                'tokens' : list_tokens,
                'relations' : list_relations}


    return dict_ann


def read_all_file():

    PATH = 'data/csd_rel_data_annotated/'
    assignee_folder_list = os.listdir(PATH)[3:3+15]

    result = []
    for assignee_folder in assignee_folder_list:
        text_folder_list = sorted(os.listdir(PATH + assignee_folder))
        text_folder_list = [i for i in text_folder_list if i[-3:] in ['ann', 'txt']]
        text_folder_list = set(map(lambda x : x[:-4], text_folder_list))


        for text_folder in text_folder_list:

            filename = assignee_folder + '/' + text_folder + '.ann'

            try:
                dict_ann = read_ann_file(filename)
                result.append(dict_ann)

            except:
                print(filename)



    return result

# input น่าจะเป็น list of dicts

def get_E1(text, start_E1, end_E1):

    return text[start_E1: end_E1]

def get_E2(text, start_E2, end_E2):

    return text[start_E2: end_E2]

def get_before_E1(text, start_E1):

    return text[:start_E1]

def get_before_E2(text, start_E2):

    return text[:start_E2]

def get_after_E1(text, end_E1):

    return text[end_E1:]

def get_after_E2(text, end_E2):

    return text[end_E2:]

def get_between_E1_E2(text, start_E1, start_E2, end_E1, end_E2):

    left, right = start_E1, end_E2
    if start_E1 > end_E2:
        left, right = start_E2, end_E1

    return text[left:right]

def get_POS_E1(text):

    pass

def get_POS_E1(text):

    pass

def get_POS_E2(text):

    pass

def get_POS_E2(text):

    pass

def prep_data(text, start_E1, start_E2, end_E1, end_E2, E1_entity, E2_entity):

    E1 = get_E1(text, start_E1, end_E1)
    E2 = get_E2(text, start_E2, end_E2)
    before_E1 = get_before_E1(text, start_E1)
    before_E2 = get_before_E2(text, start_E2)
    after_E1 = get_after_E1(text, end_E1)
    after_E2 = get_after_E2(text, end_E2)
    between_E1_E2 = get_between_E1_E2(text, start_E1, start_E2, end_E1, end_E2)
    POS_E1 = get_POS_E1(text)
    POS_E2 = get_POS_E2(text)

    dict_prep_data = {'text' : text,
                      'E1' : E1, 'E2' : E2,
                      'E1_entity' : E1_entity, 'E2_entity': E2_entity,
                      'before_E1' : before_E1, 'before_E2' : before_E2,
                      'after_E1' : after_E1, 'after_E2' : after_E2,
                      'between_E1_E2' : between_E1_E2}
#                       'POS_E1' : POS_E1, 'POS_E2' : POS_E2,
#                       'NER_E1' : NER_E1, 'NER_E2' : NER_E2}

    return dict_prep_data

def p(inp):

    d = []

    for doc in inp:

        text = doc['document']
        tokens = doc['tokens']
        relations = doc['relations']

        map_start2end = dict([(i['start'], i['end']) for i in tokens])
        map_start2entity = dict([(i['start'], i['entityLabel']) for i in tokens])
        for rel in relations:

            start_E1, start_E2 = rel['head'], rel['child']
            end_E1, end_E2 = map_start2end[start_E1], map_start2end[start_E2]
            E1_entity, E2_entity = map_start2entity[start_E1], map_start2entity[start_E2]

            dict_prep_data = prep_data(text, start_E1, start_E2, end_E1, end_E2, E1_entity, E2_entity)
            dict_prep_data['label'] = rel['relationLabel']
            d.append(dict_prep_data)

    df = pd.DataFrame(d)

    return df

def tokenize(df, method = 'word',columns = None):

    tokenizers = {'word' : word_tokenize,
                 'subword' : subword_tokenize,
                 'syllable' : syllable_tokenize}

    tokenizer = tokenizers[method]

    dict_columns = dict([(name, []) for name in columns])

    for index, row in df.iterrows():

        for name in columns:

            tokenized_text = tokenizer(row[name])
            dict_columns[name].append(tokenized_text)

    max_len = 0

    for name in columns:

        df[name] = dict_columns[name]
        max_len_column = max([len(i) for i in dict_columns[name]])
        max_len = max(max_len, max_len_column)

    return df, max_len

def build_map_token_to_index(df, columns = None):

    token_list = []

    for name in columns:

        tok = [j for i in df[name] for j in i]
        print(tok)
        token_list += tok

    token_set = sorted(set(token_list))
    map_tok2ind = dict([(v, k) for k, v in enumerate(token_set)])
    map_tok2ind['<UNK>'] = len(map_tok2ind)
    map_tok2ind['<PAD>'] = len(map_tok2ind)

    return map_tok2ind

def convert_to_index(df, map_tok2ind, columns = None ):

    for name in columns:

        df[name] = df[name].apply(lambda x: np.array([map_tok2ind[i] for i in x]))


    return df

def pad_sequences_(df, max_len, map_tok2ind, columns = None):

    for name in columns:


        padded_seq = pad_sequences(df[name],
                                 maxlen = max_len,
                                 dtype ='int32',
                                 padding ='post',
                                 value = map_tok2ind['<PAD>'])
        _ = []

        for seq in padded_seq:
            _.append(list(seq))

        df[name] = _


    return df

def preprocessing(df, method = 'word', columns = None):

    tokenized_df, max_len = tokenize(df = df,
                                     columns = columns,
                                     method = method)

    map_tok2ind = build_map_token_to_index(df = tokenized_df,
                                           columns = columns)

    indexed_df = convert_to_index(df = tokenized_df,
                                  columns = columns,
                                  map_tok2ind = map_tok2ind)

    padded_df = pad_sequences_(df = indexed_df,
                               columns = columns,
                               map_tok2ind = map_tok2ind,
                               max_len = max_len)
    return padded_df, map_tok2ind, max_len

def convert_tok_to_index_for_test_set(df, map_tok2ind, columns):

    dict_columns = dict([(v, []) for k,v in enumerate(columns)])

    for index, row in df.iterrows():
        for name in columns:
            res = []
            for i in row[name]:
                if i in map_tok2ind:
                    res.append(map_tok2ind[i])
                else:
                    res.append(map_tok2ind['<UNK>'])
            dict_columns[name].append(res)

    for name in columns:
        df[name] = dict_columns[name]

    return df



def return_train_test(df):

    columns = ['E1', 'E2', 'before_E1', 'before_E2', 'after_E1', 'after_E2', 'between_E1_E2']
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)
    train_df, map_tok2ind, max_len = preprocessing(train_df, method = 'subword', columns = columns)

    test_df, _ = tokenize(test_df, method = 'subword', columns = columns)
    test_df = convert_tok_to_index_for_test_set(test_df, map_tok2ind, columns)
    test_df = pad_sequences_(test_df, max_len , map_tok2ind, columns = columns)

    return train_df, test_df

if __name__ == '__main__':
    result = read_all_file()
    df = p(result)
    df.to_csv('prepared_df.csv')
    train, test = return_train_test(df)










