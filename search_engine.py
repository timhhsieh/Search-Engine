#-------------------------------------------------------------------------
# AUTHOR: Tim Hsieh
# FILENAME: search_engine.py
# SPECIFICATION: calculates precision and recall from CSV file
# FOR: CS 4250- Assignment #1
# TIME SPENT: 5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard arrays

#importing some Python libraries
import csv
import math

documents = []
labels = []

#reading the data in a csv file
with open('collection.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            documents.append (row[0])
            labels.append(row[1].strip())

#Conduct stopword removal.
stopWords = {'I', 'and', 'She', 'They', 'her', 'their'}

for i in range(len(documents)):
    words = documents[i].split()
    documents[i] = ' '.join([word for word in words if word not in stopWords])


#Conduct stemming.
stemming = {
  "cats": "cat",
  "dogs": "dog",
  "loves": "love",
}

for i in range(len(documents)):
    words = documents[i].split()
    stemmed_words = [stemming.get(word, word) for word in words]
    documents[i] = ' '.join(stemmed_words)

#Identify the index terms.
terms = []

for doc in documents:
    words = doc.split()
    for word in words:
        if word not in terms:
            terms.append(word)

#Build the tf-idf term weights matrix.
docMatrix = []

for doc in documents:
    tfidf_vector = []
    words = doc.split()
    for term in terms:
        tf = words.count(term) / len(words)
        idf = math.log((len(documents) / (sum([1 for doc in documents if term in doc.split()]))), 10)
        tfidf = tf * idf
        tfidf_vector.append(tfidf)
    docMatrix.append(tfidf_vector)
#print(docMatrix)

#Calculate the document scores (ranking) using document weigths (tf-idf) calculated before and query weights (binary - have or not the term).
query = 'cat and dogs'

query_terms = query.split()
query_terms = [term for term in query_terms if term not in stopWords]
query_terms = [stemming.get(term, term) for term in query_terms]

query_vector = [1 if term in query_terms else 0 for term in terms]

docScores = []

for doc_vector in docMatrix:
    score = sum([doc_vector[i] * query_vector[i] for i in range(len(terms))])
    #print(doc_vector)
    #print(query_vector)
    docScores.append(score)
#print(docScores)   


#Calculate and print the precision and recall of the model by considering that the search engine will return all documents with scores >= 0.1.
threshold = 0.1

# Count the number of relevant documents (true positives) above the threshold
relevant_documents = [i for i, score in enumerate(docScores) if score >= threshold]
#print("Relevant documents: " + str(relevant_documents))

total_retrieved = len(relevant_documents)
#print("Total Retrieved: " + str(total_retrieved))

# Count the number of true positives (retrieved and relevant)
true_positives = sum(1 for i in relevant_documents if labels[i] == 'R')
#print("True positives: " + str(true_positives))

# Count the total number of relevant documents in the collection
total_relevant = labels.count('R')
#print("Total Relevant: " + str(total_relevant))

precision = true_positives / total_retrieved if total_retrieved > 0 else 0
recall = true_positives / total_relevant if total_relevant > 0 else 0

print("Precision:", precision)
print("Recall:", recall)