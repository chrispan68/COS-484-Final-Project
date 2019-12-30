# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 00:33:12 2019

@author: evefl
"""

from bs4 import BeautifulSoup
from selenium import webdriver           # pip install selenium, install chromedriver, and change driver path before use
from html.parser import HTMLParser
import re
import pickle
from collections import OrderedDict
import random
import wordsegment
from PyDictionary import PyDictionary

def get_paragraphs_from_metadata(fpath):
    
    driver = webdriver.Chrome("C:/Users/evefl/.spyder-py3/chromedriver_win32/chromedriver.exe")
    text = open(fpath, 'r', encoding='utf8')
    print(text.readline())  # do not remove this line
    
    data = []
    ctr = 0
    for line in text:
        new_data = []
        items = re.split('(,)(?=(?:[^"]|"[^"]*")*$)', line)
        
        try:
          year = int(items[16])
        except ValueError:
          print("NOT INT", items)
          continue

        text = items[14]
        text = text.replace('"', ' ')
        city = items[18].strip("[").strip("]").strip(":")
        city = "".join(city.split())        
    
        text_id = items[0]
        print(text_id, items)
        
        url = "https://prosody.princeton.edu/archive/" + text_id + "/?query=book"
    
        driver.get(url)
    
        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")

        snippets = []
        tag_re = re.compile(r'<[^>]+>')
        for item in soup.find_all("p", "snippet"):
            
            print(item)
            snippet = tag_re.sub('', str(item))
            snippet = snippet.replace("\n", "").replace("-  ", "").replace('\"', '').replace("'", "").replace("•", "")
            snippets.append(snippet)
            
            # Problems: *rv&gt;
        
        print(snippets)
        
        new_data = {"id": items[0], "text": text, "long_text": snippets, "year": year, "city": items[18], "region": None, "period": None} # [id, text, date, place] 
        data.append(new_data)
        
        
    with open('ppa_long_text_dict.pickle', 'wb') as handle:
        pickle.dump(data, handle)
        
        
def clean_full_data(data):
    wordsegment.load()
        
    # Get region: British, American, or other
    us_list = ["New York", "Chicago", "Philadelphia", "New-York", "N.Y.", "N.H.", "N.J.", "Calif.",
                "N. Y.", "Sacramento", "MN", "Mich", "Washington", "Sacramento", "Pa.", "Columbus", 
                "Madison", "Albany", "Ind.", "Tex", "Tenn" "Ohio", "Urbana", "Portland","Boston", 
                "St. Louis", "Cincinnati", "Baltimore", "Washington, D.C.", "Andover", "Richmond", 
                "Rochester", "Louisville", "New Haven", "Indianapolis", "Greensboro", "Raleigh", 
                "Nashville", "San Francisco", "Los Angeles", "Mass.", "Me.", "New Orleans", "Tenn.",
                "Atlanta", "Minn", "Berkeley", "Kansas", "Syracuse", "Providence", "Lincoln", "Wis.", 
                "New-Haven", "Buffalo", "Pittsburgh", "Ill.", "Detroit", "Ann Arbor"]
    uk_list = ["London", "Cambridge", "Oxford", "Glasgow", "Liverpool", "Edinburg", 
                "Eng.", "York", "Edinburgh", "Westminster","Belfast", "Dublin", "Manchester", 
                "Abingdon", "Birmingham", "Cork", "Newcastle", "Stratford-upon-Avon", "Perth", 
                "Southampton", "Bath", "Eton", "Lond."]
    europe_list = ["Paris", "Berlin", "Leipzig", "Strassburg", "Zurich", "Lund", "Heidelberg",
                    "Madrid", "Bologna", "Göttingen", "Groningen", "Uppsala"] #do NOT switch order of US/UK checks
    
    counts = [0,0,0, 0]

    for index, doc in enumerate(data):
        
        
        categorized = False
        for city in us_list:
            if city in doc["city"]:
                doc["region"] = "US"
                counts[0]+=1 
                categorized = True
    
        if not categorized:
            for city in uk_list:
                if city in doc["city"]:
                    doc["region"] = "UK"
                    counts[1]+=1 
                    categorized = True

        if not categorized:
            for city in europe_list:
                if city in doc["city"]:
                    doc["region"] = "Europe"
                    counts[2]+=1 
                    categorized = True

        if not categorized:
            doc["region"] = "UNK"
            counts[3] += 1
        #print(doc["city"], doc["region"], doc["year"])


        


    print("Initial country-of-origin counts before evening out:", counts[0], "US", counts[1], "UK", counts[2], "Europe", counts[3], "Unknown")
    # Get time period counts
    period_counts = OrderedDict({(1700, 1842): [0, 0], (1843, 1874): [0, 0], (1875, 1901): [0, 0],
                              (1902, 1923): [0, 0]})                            # CHANGED START FROM 1550 TO 1700 (doesn't change data distribution)


    for index, doc in enumerate(data):
        for period in period_counts:
            if doc['year'] >= period[0] and doc['year'] <= period[1]:
                data[index]['period'] = period
        
                if doc['region'] == "US":
                    period_counts[period][0] += 1
                else:
                    period_counts[period][1] += 1
                break

    print("Overall period counts, before sampling:", period_counts)

    # Keep only British and American documents; sample evenly from US and UK
    random.shuffle(data)

    uk_data = [x for x in data if (x['region']=="UK" and len(x['text'].split()) < 25)]          # maybe truncate later if more examples necessary
    us_data = [x for x in data if (x['region']=="US" and len(x['text'].split()) < 25)][:len(uk_data)]


    split = int(.8*len(uk_data))
    train_set = uk_data[:split] + us_data[:split]
    test_set = uk_data[split:] + us_data[split:]

    random.shuffle(train_set)
    random.shuffle(test_set)


#    print("---INFO---")
#    print("Train set data by period ("+ str(len(train_set))+ " total examples):")
#    period_counts = OrderedDict({(1700, 1842): [0, 0], (1843, 1874): [0, 0], (1875, 1901): [0, 0],
#                              (1902, 1923): [0, 0]})
    
    for index, doc in enumerate(train_set):
        
        if index % 50 == 0:
            print(index)
        
        # Clean up long text OCR
        
        #print(doc["long_text"])
        new_sentences = []
        replacements = {"&amp;c": "", ". . . .": "", "&amp;c.": "", "**": "", "---":""}
        
        
        for sentence in doc["long_text"]:
            
            for mistake in replacements:
                new_sentence = sentence.replace(mistake, replacements[mistake])
                #if new_sentence != sentence:
                #    print("FIxED MISTAKE", sentence, new_sentence)
            
            new_words = []
            for word in new_sentence.split():
                
                #if dictionary.meaning(word) != None:
                #print(word)
                #print("IN DICTIONARY")
                #    new_words.append(word)
                
                capped = word.istitle()
                
                segmentation = wordsegment.segment(word)
                #print(segmentation)
                
                if capped and len(segmentation)>=1:
                    #print(segmentation)
                    segmentation[0] = segmentation[0].capitalize()
                new_words += segmentation
                
                #if len(segmentation) > 1:
                #    print("ADDED SEGMENTED", word, segmentation)
            
            new_sentence = " ".join(new_words)
            new_sentences.append(new_sentence)
        
        #print("OLD", doc["long_text"])
        #print("NEW", new_sentences)
        doc["long_text"] = new_sentences
        
#        for period in period_counts:
#            if doc['year'] >= period[0] and doc['year'] <= period[1]:
#                if doc['region'] == "US":
#                    period_counts[period][0] += 1
#                else:
#                    period_counts[period][1] += 1
#            break
#    print(period_counts)

#    print("Test set data by period ("+ str(len(test_set)) + " total examples):")
#    period_counts = OrderedDict({(1700, 1842): [0, 0], (1843, 1874): [0, 0], (1875, 1901): [0, 0],
#                                 (1902, 1923): [0, 0]})
    
    for index, doc in enumerate(test_set):
        
        if index % 50 == 0:
            print(index)
        
        # Clean up long text OCR
        new_sentences = []
        replacements = {"&amp;c": "", ". . . .": "", "&amp;c.": "", "**": "", "---":""}
        
        
        for sentence in doc["long_text"]:
            
            for mistake in replacements:
                new_sentence = sentence.replace(mistake, replacements[mistake])
            
            new_words = []
            for word in new_sentence.split():
                
                
                capped = word.istitle()
                
                segmentation = wordsegment.segment(word)
                
                if capped and len(segmentation)>=1:
                    segmentation[0] = segmentation[0].capitalize()
                new_words += segmentation
                
            new_sentence = " ".join(new_words)
            new_sentences.append(new_sentence)
        
        #print("OLD", doc["long_text"])
        #print("NEW", new_sentences)
        doc["long_text"] = new_sentences
        
#        for period in period_counts:
#            if doc['year'] >= period[0] and doc['year'] <= period[1]:
#                
#                if doc['region'] == "US":
#                    period_counts[period][0] += 1
#                    
#                else:
#                    period_counts[period][1] += 1
#            break

    print("TRAIN SET EXAMPLES:", train_set[:3], train_set[:3])
    print("TEST SET EXAMPLES:", test_set[:3], test_set[:3])
    #print(period_counts)

    # Then return val dict and train dict

    return train_set, test_set, period_counts




#get_paragraphs_from_metadata("corpus.mm.metadata")


full_dict = pickle.load( open('ppa_long_text_dict.pickle', "rb" ) )
train_set, test_set, period_counts = clean_full_data(full_dict)

with open('full_train_set.pickle', 'wb') as handle:
    pickle.dump(train_set, handle)
    
with open('full_test_set.pickle', 'wb') as handle:
    pickle.dump(test_set, handle)

with open('full_period_counts.pickle', 'wb') as handle:
    pickle.dump(period_counts, handle)