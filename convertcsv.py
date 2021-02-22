# import the MongoClient class
from pymongo import MongoClient
# import the Pandas library
import pandas

# these libraries are optional
import json
import time
from pyexcel.cookbook import merge_all_to_a_book
# import pyexcel.ext.xlsx # no longer required if you use pyexcel >= 0.2.2 
import glob


# build a new client instance of MongoClient
mongo_client = MongoClient('mongodb://<account-name>:<password>@cluster0-shard-00-00.zwnev.mongodb.net:27017,cluster0-shard-00-01.zwnev.mongodb.net:27017,cluster0-shard-00-02.zwnev.mongodb.net:27017/<project-name>?ssl=true&replicaSet=atlas-bevp92-shard-0&authSource=admin&retryWrites=true&w=majority')

# create new database and collection instance
db = mongo_client.myproject
col = db.Retail

# start time of script
start_time = time.time()

# make an API call to the MongoDB server
cursor = col.find()

# extract the list of documents from cursor obj
mongo_docs = list(cursor)

# restrict the number of docs to export
mongo_docs = mongo_docs[:10] # slice the list
print ("total docs:", len(mongo_docs))

# create an empty DataFrame for storing documents
docs = pandas.DataFrame(columns=[])

# iterate over the list of MongoDB dict documents
for num, doc in enumerate(mongo_docs):
# convert ObjectId() to str
	doc["_id"] = str(doc["_id"])
	# get document _id from dict
	doc_id = doc["_id"]
	# create a Series obj from the MongoDB dict
	series_obj = pandas.Series( doc, name=doc_id )
	# append the MongoDB Series obj to the DataFrame obj
	docs = docs.append(series_obj)


# export the MongoDB documents as a JSON file
docs.to_json("./datas/Retail.json")

# have Pandas return a JSON string of the documents
json_export = docs.to_json() # return JSON data
# print ("\nJSON data:", json_export)

# export MongoDB documents to a CSV file
docs.to_csv("./datas/Retail.csv", ",") # CSV delimited by commas

# export MongoDB documents to CSV
csv_export = docs.to_csv(sep=",") # CSV delimited by commas
print ("\nCSV data:", csv_export)

# create IO HTML string
import io
html_str = io.StringIO()

# export as HTML
docs.to_html(
buf=html_str,
classes='table table-striped'
)

# print out the HTML table
print (html_str.getvalue())

# save the MongoDB documents as an HTML table
docs.to_html("./datas/Retail.html")
merge_all_to_a_book(glob.glob("./datas/Retail.csv"), "./datas/Retail.xls")


print ("\n\ntime elapsed:", time.time()-start_time)