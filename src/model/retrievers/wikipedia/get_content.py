"""
This script creates a jsonl file containing the extracted wikipedia content of the demanded wikipedia identifiers.
 The input to script must be a txt file with each wikipedia identifier in a separate line.
"""
import sys
import wikipedia
import jsonlines
from tqdm import tqdm
import mwparserfromhell
import os
import json
import requests

input_file_name = sys.argv[1]
vocab_mentions = set(
    [line.strip().replace('_', ' ') for line in open(input_file_name, 'r').readlines() if line.strip()])

with open('wikipedia_redirects.json', 'r') as file:
    wikipedia_redirects = json.load(file)


mentions = []

if os.path.exists(f'{input_file_name}.jsonl'):
    with jsonlines.open(f'{input_file_name}.jsonl', 'r') as reader:
        for obj in reader:
            mention = list(obj.keys())[0]
            mentions.append(mention)

print(f'Pre-loaded {len(mentions)} articles ...')

page_errors = []
with jsonlines.open(f'{input_file_name}.jsonl', 'a') as writer:
    for mention in tqdm(vocab_mentions):
        if mention in mentions:
            continue
        if mention.replace(' ', '_') in wikipedia_redirects:
            mention = wikipedia_redirects[mention.replace(' ', '_')]
        try:
            w_page = wikipedia.page(mention.replace('_', ' '))
            print(w_page.title)
            print('------------------'*4)
            wikicode = mwparserfromhell.parse(w_page.content)
            main_text = ""
            for section in wikicode.filter_text():
                main_text += str(section)
            main_text += "".join([f"\nCategory:{x}" for x in w_page.categories])
            writer.write({mention: main_text})
        except wikipedia.exceptions.PageError:
            page_errors.append(mention)
        except wikipedia.exceptions.DisambiguationError:
            page_errors.append(mention)
        except KeyError:
            page_errors.append(mention)
        except requests.exceptions.ReadTimeout:
            page_errors.append(mention)


with open('page_errors.txt', 'w') as f:
    for error in page_errors:
        f.write(error+'\n')
