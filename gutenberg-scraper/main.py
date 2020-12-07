# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 18:27:16 2020

@author: Bjarne Gerdes
"""
from pymarc import MARCReader
from urllib.request import urlopen
from tqdm import tqdm
import pymongo

class ScrapeGutenberg:
    
    def __init__(self, catalog_path):
        self.catalog_path = catalog_path
        self.catalog_file = open(catalog_path, "rb")
        self.reader = MARCReader(self.catalog_file)
        self.mongo_client = pymongo.MongoClient("mongodb://admin:password@h2877813.stratoserver.net:12321/")
        self.db = self.mongo_client["authordatabase"]
        self.collection = self.db["bookcontents"]
        
    def getContent(self, gutenberg_id):
        try:
            return str(urlopen(f"http://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt.utf8").read())
        except:
            try:
                return str(urlopen(f"http://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt").read())
            except:
                return None
        
    def getData(self, record):
        author = record.author()
        title = record.title()
        gutenberg_id = record.get_fields()[0].data
        url_gutenberg = f"http://www.gutenberg.org/cache/epub/{gutenberg_id}"
        
        payload = {"author": author,
                "title": title,
                "url_gutenberg": url_gutenberg,
                "gutenberg_id": gutenberg_id,
                 "content": self.getContent(gutenberg_id)}

        self.collection.insert_one(payload)


    def getAll(self):
        for record in tqdm(self.reader):
            self.getData(record)


scraper = ScrapeGutenberg("catalog.marc")
scraper.getAll()