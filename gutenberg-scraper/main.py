# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 18:27:16 2020

@author: Bjarne Gerdes
"""
from pymarc import MARCReader
from urllib.request import urlopen
from tqdm import tqdm
import csv

class ScrapeGutenberg:
    
    def __init__(self, catalog_path):
        self.storage_path = "./data/data.csv"
        self.catalog_path = catalog_path
        self.catalog_file = open(catalog_path, "rb")
        self.reader = MARCReader(self.catalog_file)
        
        
    def getContent(self, gutenberg_id):
        """

        Scrapes the content of the book by making use of the gutenberg id
        ----------
        gutenberg_id : int
            ID of the book

        Returns
        -------
        str
            Content of the book.

        """
        try:
            return str(urlopen(f"http://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt.utf8").read())
        except:
            try:
                return str(urlopen(f"http://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt").read())
            except:
                return None
        
    def getData(self, record):
        """
        Takes the MARC-record and extracts the metadata

        Parameters
        ----------
        record : 
            MARC-recard of th ecatalog.marc file.

        Returns
        -------
        None.

        """
        author = record.author()
        title = record.title()
        gutenberg_id = record.get_fields()[0].data
        url_gutenberg = f"http://www.gutenberg.org/cache/epub/{gutenberg_id}"
        
        payload = {"author": author,
                "title": title,
                "url_gutenberg": url_gutenberg,
                "gutenberg_id": gutenberg_id,
                 "content": self.getContent(gutenberg_id)}
            
        # Store data in a csv format.
        try:
            self.writer.writerow(payload)
        except IOError:
            print("I/O error")
            
    def getAll(self):
        """
        Iterates over all MARC-Records and processes their contents.

        Returns
        -------
        None.

        """
        # Create the data.csv file
        with open(self.storage_path, 'w') as csvfile:
            self.writer = csv.DictWriter(csvfile,\
                                         fieldnames=["author","title","url_gutenberg","Gutenberg_id","content"])
            self.writer.writeheader()

        # process all books, book by book
        for record in tqdm(self.reader):
            self.getData(record)


scraper = ScrapeGutenberg("catalog.marc")
scraper.getAll()