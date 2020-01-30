
"""
CAS IE - PROJECT IR: 
    
    *** MOVIE RECOMMENDER ***

Christophe Otter & Severin Trösch, 
January 2020

"""


#%% packages & settings

import pandas as pd
import time
import kaggle
import os
import re
import nltk #language processing
#from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#from nltk.corpus import wordnet
stopw = set(stopwords.words('english')) # set of stopwords
#import spacy #Named-entity recognition
#import en_core_web_sm # - You can now load the model via spacy.load('en_core_web_sm')
from cachetools import cached, TTLCache  # import the "cached" decorator and the "TTLCache" object from cachetools
import math
import json

#%% load and cache file

## prepare cache
cache_folder = 'C:\\Users\\sever\\Google Drive\\Dokumente PC\\ZHAW\\CAS Information Engineering\\Modul IR\\project'
os.chdir(cache_folder)
#os.getcwd()

cachetime = 1000 
mycache = TTLCache(maxsize = 1, ttl = cachetime)  # create the cache object.

# use cache dacorator on function that loads data
@cached(cache = mycache)  # decorate the method to use the cache system
def get_kaggle_ds(dbname = "jrobischon/wikipedia-movie-plots"): #my function to load data from url using pandas
    
    ''' 
    dbname = name / path of kaggle dataset (in format [owner]/[dataset-name]), e.g.(default) "jrobischon/wikipedia-movie-plots"
    '''
    
    print("loading kaggle dataset: {}".format(dbname))
    print()
    
    
    try: 
        # load data from kaggle api
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset = dbname, 
                                  path = os.path.join(cache_folder,"data"),
                                  unzip=True)  
        
        # then load df load from local disk - path stimmt noch nicht (ev. os.path.join braichen)
        data = pd.read_csv(os.path.join(cache_folder,"data",'wiki_movie_plots_deduped.csv'), delimiter=',')

        print("Dataset is cached. {} s cache remaining.".format(cachetime))
        print()
        print()
        
        #return data
        return(data)
    
    except IOError as detail:
        print("Error! Dataset could not be loaded:")
        print()
        print(detail)
        
# test
# moviez = get_kaggle_ds() #test

#%% get data and randomly subset it
movies = get_kaggle_ds()

# for development: random subset of movies
#movies = movies.sample(frac=0.1, replace=False, random_state=1)# small, random subset of movies for development


#%% write all other vars of data frame together with "Plot" in "All" variable - baloon them (overweight factor)

other_var_weight_factor = 10 #how many times should the non-Plot variables be repeated

movies["All"] = movies['Plot'].map(str) + " " + \
                (movies['Title'].map(str) + " ")*other_var_weight_factor + \
                (movies['Origin/Ethnicity'].map(str) + " ")*other_var_weight_factor +\
                (movies['Director'].map(str) + " ")*other_var_weight_factor + \
                (movies['Cast'].map(str) + " " )*other_var_weight_factor+ \
                (movies['Genre'].map(str))*other_var_weight_factor

# make list to iterate over later
#movies_dev_ls = movies_dev.All.tolist()
movies_ls = movies.All.tolist()


#%% save movie df as csv
#movies.to_csv("movies_df.csv")



###################################################################### DEFINE HELPER FUNCTIONS #########


#%% normalizer function

#nlp = en_core_web_sm.load()

def nor(txt):
    
    ''' doc:
       input (txt) = text string that needs to be normalised
       output = a list of normalised tokens - NOT UNIQUE!
    '''
    
    ## tokenize
    txt_2 = nltk.word_tokenize(txt) #tokenize
    
    ## remove stopwords
    txt_2 = [word for word in txt_2 if word not in stopw]
    
    ## remove special characters and capitals
    txt_2 = [re.sub(r'\W+', '', word).lower() for word in txt_2]
    
    ## remove empty indexes
    txt_2 = [word for word in txt_2 if len(word)>1]
    
#    # stemming ------------------------------------------ does that not make it worse?
#    ps = PorterStemmer()
#    ttry_2 = [ps.stem(word) for word in ttry_2]
    
    # return result: a list of normalised tokens - NOT UNIQUE! - and joined with named entities
    return txt_2
    

#%% idf

# function that calculates idf for tokens in a collection of documents
# returns a dict (key = token, velue = idf)

def idf(coll):
    
    """doc:
    in: coll = list of of documents (strings of multiple tokens)
    out: dictionary of tokens to idf of token in collection
    """ 
    ts1 = round(time.time(),0) #timestamp 1
    
    N = len(coll)  
    print("Analysing {} documents...".format(N))
    print()
    
    dic = {}
    
    # get a list of sets of normalised tokens
    lss = [set(nor(doc)) for doc in coll] #returns a list of sets
    
    ts2 = round(time.time(),0) #timestamp 2
    print("step 1 (normalising & TF of all movie-plot strings) done ... (took {} s)".format(ts2-ts1))
    
    # set of all words in coll
    settot = set()
    for s in lss:
        settot = settot.union(s)
        
    ts3 = round(time.time(),0) #timestamp 3
    print("step 2 (merge all sets of normalised strings to one set) done ... (took {} s)".format(ts3-ts2))
        
    # for all tokens in collection 
    for tok in list(settot):
        nrdoc = 0
        
        # for all sets of tokens (i.e. documents)
        for tokset in lss:
            if tok in tokset:
                nrdoc += 1 #nrdoc: in how many docs is the token?
        
        dic[tok] = N/nrdoc #idf = N / number of docs that contain tok
    
    ts4 = round(time.time(),0) #timestamp 4 
    print("step 3 (for all tokens in all movies: determine idf) done ... (took {} s)".format(ts4-ts3))
    print()
    print("finished. (overall analysis time: {} s)".format(ts4 - ts1))
    return dic


#%% create function for TF out of list of not-unique normalised tokens (result from nor() function)
    
def tf(lis):
    
    ''' doc:
       input: 
           lis = list of not-unique normalised tokens (result from nor() function)
       output = a dict  with token and term freqencies
    '''
        
    res = {} #dict of tokens in lis (key) and # occurances in lis (value)
    cindex = set() #set of tokens in txt
    
    nt = len(lis) # number of tokens in txt
    
    # for every word in list
    for t in lis:
        if t not in cindex:
            cindex.add(t)
            res[t] = 1/nt # /nt gives the normalisation for the term-frequency
        else:
            res[t] += 1/nt
    
    ## return result - a dict of token and term freqencies
    return res


#%% combine with idf - function takes string and gives dict of tokens:tfidf

def tfidf(txt, idfs):
    """ Compute TF-IDF
    Args:
        txt: string
        idfs (dictionary): token to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tfs = tf(nor(txt)) #a dict of token and term freqencies
    
    resdict = {}
    
    for t in tfs:
        if t in idfs:
            resdict[t] = tfs[t] * idfs[t] #calculate TF-IDF
        else:
            resdict[t] = 0 #zero if word is NOT in idfs reference
            
    return resdict


#%% build first function to compare query and corpus  - and make trivial ranking (adding up of tfidf weights)
    
def get_ranking_add(query, ref_dict, threshold = 1):
    
    """ calculate and return a ranking of movies relative to searchstring (query)
    input:
        query = searchstring
        ref_dict = reference dictionary of movies (title:dict(token:tfidf))
        threshold = threshold reference score (default = 1) above which results are returned
    output:
        a sorted listof tuples (title, relevance score) with top-10 movies in decreasing similarity scores-order
    """
    
    ## first, apply tf() and nor() function to query
    q = tf(nor(query)) #dict of query tokens and term freq
    
    ## second, iterate over ref_dict and compare query to values (calculate relevance score)
        ## the result is a dict with movie titles as key and the relevance score (rs) as value
    res = {}
    
    for film in ref_dict: #all films in ref_dict
        
        rs = 0
        
        filmwords = [] #prepare list of film-words - new for every film
        
        for filmword in ref_dict[film]: #for all normalised words of film
            filmwords.append(filmword) #append words of film
        
        for queryword in q: #all words in query
            if queryword in filmwords: #check if any word of query is in filmwords (list of words in one film)
                rs += ref_dict[film][queryword] #if yes, add weight of word to rs
        
        # add relevance score for film to res-dict
        res[film] = rs 
        
    ## define a sorted list with decreasing relevance scores
    result1 = sorted(res.items(), key=lambda kv: kv[1], reverse = True)
    # and choose items over threshold
    result2 = [mov[0] for mov in result1 if mov[1] > threshold]
             
    ## finally, return result
    return result2

#%% implement cosine-similarity function(s)

# dot product and normalizing funs
def dotprod(a, b):
    return sum([a[t] * b[t] for t in a if t in b])

def norm(a):
    return math.sqrt(dotprod(a, a))

## then, the cossim fun
#def cossim(a, b):
#    return dotprod(a, b) / norm(a) / norm(b)

# and finally, the actual fun to calculate cossim from two dicts
def cossim(dict1, dict2):
    """ Compute cosine similarity between two strings
    Args:
        dict1: first dict of tokens and tf-idf values
        dict2: second dict of tokens and tf-idf values

    Returns:
        cossim: cosine similarity value
    """

    return dotprod(dict1, dict2) / norm(dict1) / norm(dict2)
    
#test function
cossim(res_dict['Americana'],tfidf("james bond American idiot",idfs)) #works

#%% build second ranking-function to compare query and corpus  - using cosine similarity
    
def get_ranking_cossim(query, ref_dict, threshold = 0, idf_dict = idfs):
    
    """ calculate and return a ranking of movies relative to searchstring (query)
    input:
        query = searchstring
        ref_dict = reference dictionary of movies (title:dict(token:tfidf))
        threshold = threshold reference score (default = 1) above which results are returned
        idfs = reference dict with term idfs
    output:
        a sorted listof tuples (title, relevance score) with top-10 movies in decreasing similarity scores-order
    """
    
    ## first, apply tf() and nor() function to query
    q = tfidf(query, idf_dict) #dict of query tokens:tfidf
    
    ## second, iterate over ref_dict and compare query to values (calculate relevance score)
        ## the result is a dict with movie titles as key and the relevance score (rs) as value
    res = {}
    
    for film in ref_dict: #all films in ref_dict
        
        
        # calculate relevance score (cosine similarity)
        rs = cossim(q,ref_dict[film])
        
        # add relevance score for film to res-dict
        res[film] = rs 
        
    ## define a sorted list with decreasing relevance scores
    result1 = sorted(res.items(), key=lambda kv: kv[1], reverse = True)
    # and choose items over threshold
    result2 = [mov[0] for mov in result1 if mov[1] > threshold]
             
    ## finally, return result
    return result2

###################################################################### NOW THE ACTUAL WORK #########
#%% make dict of idfs for movies (for lookup)
    
idfs = idf(movies_ls) # this is a one-time step that takes a while (approx. 15 min)



#%% apply functions to  dataset - a dict results (title:list of tuples(title,TF))

res_dict = {} #dictionary with titles (keys) and result dict (values). result dict is token:tfidf
    
for i, row in movies.iterrows():

    txt = row["All"]
    print("analysing ",row["Title"],"...")
    print()
    res_dict[row["Title"]] = tfidf(txt, idfs) 


#%% save the two relevant dicts as json
    
# res_dict (dict of titles:dict(token:tfidf))
with open('res_dict.json', 'w') as outfile:
    json.dump(res_dict, outfile)
    
# idfs (dict of tokens:idfs)
with open('idfs.json', 'w') as outfile2:
    json.dump(idfs, outfile2)
    
###################################################################### AND FINALLY, THE APPLICATION #########

#%% check the thing with a first query:
qu1 = "tarantino"

get_ranking_add(qu1, res_dict,  threshold = 1)
get_ranking_cossim(qu1, res_dict, threshold = 0.01)

#%% or this:

get_ranking_add(input(), res_dict)
