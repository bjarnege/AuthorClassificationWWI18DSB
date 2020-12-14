# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:02:33 2020

@author: Bjarne Gerdes
"""
from textblob import TextBlob
from tqdm import tqdm
import spacy
import os
import re
import html

path = os.path.dirname(os.path.abspath(__file__))
# Initialize spacy with english language model
nlp = spacy.load(
    f'{path}/data/en_core_web_sm-2.2.5/en_core_web_sm/en_core_web_sm-2.2.5')
# Set max RAM for space on 10 gb
nlp.max_length = 10_000_000
# Read the stopwordslist
stopwords = [l.replace("\n", "")
             for l in open(f"{path}/data/stopwords.txt").readlines()]



# # Text-Preprocessing
class Preprocessing():

    def __init__(
            self,
            stopwords: list = stopwords,
            nlp=nlp) -> None:
        """
        This class performs the needed preprocessing

        Parameters:
            stopwords: A list of stopwords.

            nlp: The fitted nlp-Function from spaCy
        """
        self.stopwords = stopwords
        self.nlp = nlp
        self.noStopWords = None
        self.lemmatizedWords = None

        
    def Cleaning(self, document):
        """
        Clean the documents to provide clear structures
        """
        # remove any kind of links:
        document = document.replace("urlLink","")
        document = re.sub(r'http\S+', '', document)
    
        # decode html entities
        return html.unescape(document)
    
    def ExtractWords(self, document) -> str:
        """
        Removing unusual letters etc.
        """
        self.onlyWords = "".join(
            [f"{w} " for w in TextBlob(document).words])

        return self.onlyWords

    def RemoveStopwords(self) -> str:
        """
        Removing stopwords using the list of stopwords from self.stopwords
        """
        self.noStopWords = "".join(
            [f"{word} " for word in self.onlyWords.split() if not word.lower() in self.stopwords])

        return self.noStopWords

    def Lemmatizer(self) -> str:
        """
        Lemmatize words using spaCy
        """
        doc = self.nlp(self.noStopWords)

        self.lemmatizedWords = "".join(
            [f"{token.lemma_.lower()} " for token in doc])
        
        return self.lemmatizedWords

    def ProcessOne(self, document) -> str:
        """
        Run through the whole preprocessing pipeline
        """
        document = self.Cleaning(document)
        self.ExtractWords(document)
        self.RemoveStopwords()

        return self.Lemmatizer()

    def ProcessMany(self, document_list) -> list:
       """
       Processes a list of documents

       Parameters
       ----------
       document_list : list
           List of raw documents.

       Returns
       -------
       list
           List of preprocessed documents.

       """
       return [self.ProcessOne(document) for document in tqdm(document_list)]
   


    