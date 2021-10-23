# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:26:51 2021

@author: Owner
"""

# =============================================================================
# Data Processing
# =============================================================================
import spacy
import unidecode
import re
import string
import pandas as pd
from sklearn.utils import resample
from word2number import w2n
import en_core_web_sm
nlp = en_core_web_sm.load()
nlp = spacy.load("en_core_web_lg")
nlp = spacy.load('en_core_web_sm') 
import pickle

auto_replay= pd.read_json(r"C:\Users\Owner\Desktop\פרויקט גמר\data\inbound_data_auto_reply.json")
manual_replay= pd.read_json(r"C:\Users\Owner\Desktop\פרויקט גמר\data\inbound_data_manual_reply.json")
extracted_data= pd.read_csv(r"C:\Users\Owner\Desktop\פרויקט גמר\data\extracted_data.csv")
TaggedData = pd.read_csv(r"C:\Users\Owner\Desktop\פרויקט גמר\data\taggedData.csv")

# =============================================================================
# reorgenize data
# =============================================================================
auto_replay.columns = ["id", "replay"]
manual_replay.columns = ["id", "replay"]
extracted_data.columns = ["id", "replay"]
TaggedData.columns = ["id","replay","class"]
TaggedData.drop(['class'], axis='columns', inplace=True)
data_before_shuffle = pd.concat([auto_replay, manual_replay,extracted_data,TaggedData])
data_before_shuffle.head()

data = resample(data_before_shuffle,random_state =123) # same seed

# =============================================================================
# Finding patterns
# =============================================================================
def clean_text(df):
    all_rep = list()
    emails = df["replay"].values.tolist()
    for text in emails:
        if isinstance(text, str):
            link_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            text = link_pattern.sub('_link_', text) ##Identifies a word that begins with an Internet prefix and replaces with a placed sign
            link_pattern2 = re.compile('(?i)((?:https?://|www\d{0,3}[.])?[a-z0-9.\-]+[.](?:(?:international)|(?:construction)|(?:contractors)|(?:enterprises)|(?:photography)|(?:immobilien)|(?:management)|(?:technology)|(?:directory)|(?:education)|(?:equipment)|(?:institute)|(?:marketing)|(?:solutions)|(?:builders)|(?:clothing)|(?:computer)|(?:democrat)|(?:diamonds)|(?:graphics)|(?:holdings)|(?:lighting)|(?:plumbing)|(?:training)|(?:ventures)|(?:academy)|(?:careers)|(?:company)|(?:domains)|(?:florist)|(?:gallery)|(?:guitars)|(?:holiday)|(?:kitchen)|(?:recipes)|(?:shiksha)|(?:singles)|(?:support)|(?:systems)|(?:agency)|(?:berlin)|(?:camera)|(?:center)|(?:coffee)|(?:estate)|(?:kaufen)|(?:luxury)|(?:monash)|(?:museum)|(?:photos)|(?:repair)|(?:social)|(?:tattoo)|(?:travel)|(?:viajes)|(?:voyage)|(?:build)|(?:cheap)|(?:codes)|(?:dance)|(?:email)|(?:glass)|(?:house)|(?:ninja)|(?:photo)|(?:shoes)|(?:solar)|(?:today)|(?:aero)|(?:arpa)|(?:asia)|(?:bike)|(?:buzz)|(?:camp)|(?:club)|(?:coop)|(?:farm)|(?:gift)|(?:guru)|(?:info)|(?:jobs)|(?:kiwi)|(?:land)|(?:limo)|(?:link)|(?:menu)|(?:mobi)|(?:moda)|(?:name)|(?:pics)|(?:pink)|(?:post)|(?:rich)|(?:ruhr)|(?:sexy)|(?:tips)|(?:wang)|(?:wien)|(?:zone)|(?:biz)|(?:cab)|(?:cat)|(?:ceo)|(?:com)|(?:edu)|(?:gov)|(?:int)|(?:mil)|(?:net)|(?:onl)|(?:org)|(?:pro)|(?:red)|(?:tel)|(?:uno)|(?:xxx)|(?:ac)|(?:ad)|(?:ae)|(?:af)|(?:ag)|(?:ai)|(?:al)|(?:am)|(?:an)|(?:ao)|(?:aq)|(?:ar)|(?:as)|(?:at)|(?:au)|(?:aw)|(?:ax)|(?:az)|(?:ba)|(?:bb)|(?:bd)|(?:be)|(?:bf)|(?:bg)|(?:bh)|(?:bi)|(?:bj)|(?:bm)|(?:bn)|(?:bo)|(?:br)|(?:bs)|(?:bt)|(?:bv)|(?:bw)|(?:by)|(?:bz)|(?:ca)|(?:cc)|(?:cd)|(?:cf)|(?:cg)|(?:ch)|(?:ci)|(?:ck)|(?:cl)|(?:cm)|(?:cn)|(?:co)|(?:cr)|(?:cu)|(?:cv)|(?:cw)|(?:cx)|(?:cy)|(?:cz)|(?:de)|(?:dj)|(?:dk)|(?:dm)|(?:do)|(?:dz)|(?:ec)|(?:ee)|(?:eg)|(?:er)|(?:es)|(?:et)|(?:eu)|(?:fi)|(?:fj)|(?:fk)|(?:fm)|(?:fo)|(?:fr)|(?:ga)|(?:gb)|(?:gd)|(?:ge)|(?:gf)|(?:gg)|(?:gh)|(?:gi)|(?:gl)|(?:gm)|(?:gn)|(?:gp)|(?:gq)|(?:gr)|(?:gs)|(?:gt)|(?:gu)|(?:gw)|(?:gy)|(?:hk)|(?:hm)|(?:hn)|(?:hr)|(?:ht)|(?:hu)|(?:id)|(?:ie)|(?:il)|(?:im)|(?:in)|(?:io)|(?:iq)|(?:ir)|(?:is)|(?:it)|(?:je)|(?:jm)|(?:jo)|(?:jp)|(?:ke)|(?:kg)|(?:kh)|(?:ki)|(?:km)|(?:kn)|(?:kp)|(?:kr)|(?:kw)|(?:ky)|(?:kz)|(?:la)|(?:lb)|(?:lc)|(?:li)|(?:lk)|(?:lr)|(?:ls)|(?:lt)|(?:lu)|(?:lv)|(?:ly)|(?:ma)|(?:mc)|(?:md)|(?:me)|(?:mg)|(?:mh)|(?:mk)|(?:ml)|(?:mm)|(?:mn)|(?:mo)|(?:mp)|(?:mq)|(?:mr)|(?:ms)|(?:mt)|(?:mu)|(?:mv)|(?:mw)|(?:mx)|(?:my)|(?:mz)|(?:na)|(?:nc)|(?:ne)|(?:nf)|(?:ng)|(?:ni)|(?:nl)|(?:no)|(?:np)|(?:nr)|(?:nu)|(?:nz)|(?:om)|(?:pa)|(?:pe)|(?:pf)|(?:pg)|(?:ph)|(?:pk)|(?:pl)|(?:pm)|(?:pn)|(?:pr)|(?:ps)|(?:pt)|(?:pw)|(?:py)|(?:qa)|(?:re)|(?:ro)|(?:rs)|(?:ru)|(?:rw)|(?:sa)|(?:sb)|(?:sc)|(?:sd)|(?:se)|(?:sg)|(?:sh)|(?:si)|(?:sj)|(?:sk)|(?:sl)|(?:sm)|(?:sn)|(?:so)|(?:sr)|(?:st)|(?:su)|(?:sv)|(?:sx)|(?:sy)|(?:sz)|(?:tc)|(?:td)|(?:tf)|(?:tg)|(?:th)|(?:tj)|(?:tk)|(?:tl)|(?:tm)|(?:tn)|(?:to)|(?:tp)|(?:tr)|(?:tt)|(?:tv)|(?:tw)|(?:tz)|(?:ua)|(?:ug)|(?:uk)|(?:us)|(?:uy)|(?:uz)|(?:va)|(?:vc)|(?:ve)|(?:vg)|(?:vi)|(?:vn)|(?:vu)|(?:wf)|(?:ws)|(?:ye)|(?:yt)|(?:za)|(?:zm)|(?:zw))(?:/[^\s()<>]+[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019])?)', re.IGNORECASE)
            text = link_pattern2.sub('_link_', text) ##Identifies a word that begins with an Internet prefix and replaces with a placed sign
    
            email_pattern2 = re.compile(r'[\w_\-\.]+@[\w_\-\.]+\.[a-zA-Z]{2,5}')
            text = email_pattern2.sub('_email_', text) ##Identifies a word that begins with an Internet prefix and replaces with a placed sign
            email_pattern3 = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
            text = email_pattern3.sub('_email_', text) ##Identifies a word that begins 
            email_pattern= re.compile("([a-z0-9!#$%&'+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-][a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", re.IGNORECASE)
            text = email_pattern.sub('_email_', text) ##Identifies a word that begins 
                                
            
            Phone_number_pattern= re.compile(r'^\[2-9]\d{2}\-\d{3}\-\d{4}$')
            Phone_number_pattern2= re.compile(r'\(([0-9]{2}|0{1}((x|[0-9]){2}[0-9]{2}))\)\s*[0-9]{3,4}[- ]*[0-9]{4}')
            Phone_number_pattern4= re.compile(r'^\s*\d{3}-\d{3}-\d{4}\s*$')
            Phone_number_pattern5 = re.compile(r'(\+420)?(\s*)?\d{3}(\s*)?\d{3}(\s*)?\d{3}')
            Phone_number_pattern6= re.compile(r'\+?\d{1}-\d{3}-\d{3}-\d{4}')
            Phone_number_pattern7= re.compile(r'\d\d\d\-\d\d\d\-\d\d\d\d')
            Phone_number_pattern8= re.compile('''((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))''')
            Phone_number_pattern9 = re.compile('((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))', re.IGNORECASE)
    
            text =Phone_number_pattern.sub('_Phone_number_',text)
            text =Phone_number_pattern2.sub('_Phone_number_',text)
            text =Phone_number_pattern4.sub('_Phone_number_',text)
            text =Phone_number_pattern5.sub('_Phone_number_',text)
            text =Phone_number_pattern6.sub('_Phone_number_',text)
            text =Phone_number_pattern7.sub('_Phone_number_',text)
            text =Phone_number_pattern8.sub('_Phone_number_',text)
            text =Phone_number_pattern9.sub('_Phone_number_',text)
    
    
    
            time_pattren =re.compile('\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE)
            text =time_pattren.sub('_time_',text)
            time_pattren =re.compile('\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE)
            text =time_pattren.sub('_time_',text)
    
            date_pattern= re.compile(r'^\d{1,2}\/\d{1,2}\/\d{4}$')
            date_pattern2= re.compile(r'^[0,1]?\d{1}\/(([0-2]?\d{1})|([3][0,1]{1}))\/(([1]{1}[9]{1}[9]{1}\d{1})|([2-9]{1}\d{3}))$')
            date_pattern3= re.compile(r'^(([1-9])|(0[1-9])|(1[0-2]))\/((0[1-9])|([1-31]))\/((\d{2})|(\d{4}))$')
            date_pattern4= re.compile(r'((0?[13578]|10|12)(-|\/)((0[0-9])|([12])([0-9]?)|(3[01]?))(-|\/)((\d{4})|(\d{2}))|(0?[2469]|11)(-|\/)((0[0-9])|([12])([0-9]?)|(3[0]?))(-|\/)((\d{4}|\d{2})))')
            date_pattern5= re.compile('(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}')
            date_pattern5= re.compile(r'(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}')
            date_pattern6= re.compile(r'[0-9]{2}/[0-9]')
            date_pattern7= re.compile(r'[0-9]/[0-9]{2}')
            date_pattern8 =re.compile('[0-9]{2} [am|pm]')
            
            text =date_pattern.sub('_date_',text)
            text =date_pattern2.sub('_date_',text)
            text =date_pattern3.sub('_date_',text)
            text =date_pattern4.sub('_date_',text)
            text =date_pattern5.sub('_date_',text)
            text =date_pattern6.sub('_date_',text)
            text =date_pattern7.sub('_date_',text)
            text =date_pattern8.sub('_date_',text)
          
            all_rep.append(text)
    return all_rep


def remove_common_words(text_vector):
        """Removes 50 most common words in the uk english.
        source: http://www.bckelk.ukfsn.org/words/uk1000n.html
        wuthout no,not,but
        """
        common_words = set(['the', 'and', 'to', 'of', 'a', 'I', 'in',
            'was', 'he', 'that', 'it', 'his', 'her', 'you', 'as',
            'had', 'with', 'for', 'she',  'at', 'be',
            'my', 'on', 'have', 'him', 'is', 'said', 'me', 'which',
            'by', 'so', 'this', 'all', 'from', 'they', 'were',
            'if', 'would', 'or', 'when', 'what', 'there', 'been',
            'one', 'could', 'very', 'an', 'who'])
        return [word for word in text_vector if word not in common_words]


# exclude words from spacy stopwords list
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def lower(email):
    email = email.lower()
    
def process_for_email(email):
    email = lower(email)

def remove_accented_chars(email):
    """remove accented characters from text, e.g. café"""
    email = unidecode.unidecode(email)
    return email

def remove_whitespace(email):
    """remove extra whitespaces from text"""
    email = email.strip()
    return " ".join(email.split())

CONTRACTION_MAP = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
}


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


    
def text_preprocessing(email, accented_chars=True, contractions=True, 
                       convert_num=True, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_num=True, special_chars=True, 
                       stop_words=True,remove_common_words = True):
    """preprocess text with default option set to true for all steps"""
    if extra_whitespace == True: #remove extra whitespaces
        email = remove_whitespace(email)
    if accented_chars == True: #remove accented characters
        email = remove_accented_chars(email)
    if contractions == True: #expand contractions
        email = expand_contractions(email)
    if lowercase == True: #convert all characters to lowercase
        email = email.lower()
   
    doc = nlp(email) #tokenise text
    # if (isinstance(doc,bool) == False and remove_common_words == True) :
    #     remove_common_words(doc)
    clean_text = []
    
    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True: 
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
        and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)        
    return clean_text

def main_data_process(data,tagged):
    emails_list = list()
    tagged_list = list()
    #emails_after_process = pd.DataFrame(columns = "email")
    emails_after_process_list = list()
    emails_list = clean_text(data) #patterns
    for e in emails_list:
        em = text_preprocessing(e)
        emails_after_process_list.append(em)
    
    tagged_after_process_list = list()
    tagged_list = clean_text(tagged) #patterns
    for t in tagged_list:
        em = text_preprocessing(t)
        tagged_after_process_list.append(em)
    return emails_after_process_list , tagged_after_process_list


    
emails_after_process_list , tagged_after_process_list  = main_data_process(data,TaggedData)
with open('listfile.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(emails_after_process_list, filehandle)


with open('tagged_after_process.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(tagged_after_process_list, filehandle)





    