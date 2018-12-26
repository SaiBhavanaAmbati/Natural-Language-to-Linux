from flask import Flask, render_template, request,redirect
app = Flask(__name__,static_folder='static',template_folder = 'templates')
from operator import itemgetter
import cPickle as pickle 
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import string
stop_words = set(stopwords.words('english'))
vsm_dict_file = open('vsm_dict_file.pkl', 'rb')
vsm_dict = pickle.load(vsm_dict_file)
base_avg_file = open('base_avg_file.pkl', 'rb')
base_avg = pickle.load(base_avg_file)
final_avg_file = open('final_avg_file.dat', 'rb')
final_avg = pickle.load(final_avg_file)
base_class_file = open('base_class_file.pkl', 'rb')
base_class = pickle.load(base_class_file)
import math
import os
def compute_tf_idf_query(query) :
    body = re.sub(r'[^\w\s]',' ',query)
    word_tokens = word_tokenize(body) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    query_freq = len(filtered_sentence)
    tf_idf_query = {}
    word_freq_query = {} 
    stemmed_words = [str(ps.stem(word)) for word in filtered_sentence]
    for  word in vsm_dict.keys():
        word_freq_query[word] = 0
    for  word in vsm_dict.keys():
        tf_idf_query[word] = 0.0
    for word in stemmed_words :
        if word in vsm_dict.keys() :
            word_freq_query[word] =+ 1
    for stemmed_word in stemmed_words :
        if stemmed_word in vsm_dict.keys() :
            tf_val = 1 + math.log10(float(word_freq_query[stemmed_word]))
            idf_val = math.log10(11964.0/float(vsm_dict[stemmed_word][0]))
            tf_idf_query[stemmed_word] = tf_val*idf_val
    return tf_idf_query 

def compute_similarity(query_vec,doc_vec) :
    sum_tf_idf = 0.0
    squares_tf_idf = 0.0
    for term in query_vec.keys():
        if term in doc_vec.keys() :
          sum_tf_idf = sum_tf_idf + query_vec[term]*doc_vec[term]
          squares_tf_idf = squares_tf_idf + (doc_vec[term])**2
    if squares_tf_idf != 0 :
      return sum_tf_idf/((squares_tf_idf)**(1/2))
    else : return 0 


@app.route("/")
def home() :
	return render_template('manager_dash1.html')

@app.route('/answer1', methods=['GET','POST'])
def answer():
	query = request.form.get('problem_statement')
	tf_idf_query = compute_tf_idf_query(query)
	similarities = []
	for key,value in base_avg.items() :
	  similarities.append((key,compute_similarity(tf_idf_query,value)))
	similarities = sorted(similarities, key = lambda x: x[1], reverse = True) 
	#print(similarities[:5])
	top_base_class = similarities[:5]
	final_similarities = []
	for (key,value) in top_base_class :
	  for f_command in base_class[key] :
	    final_similarities.append((f_command,compute_similarity(tf_idf_query,final_avg[f_command])))
	final_similarities = sorted(final_similarities, key = lambda x: x[1], reverse = True) 
	#print(final_similarities[:5])
	problem = [[],[]]
	problem[0].append('Similarity')
	problem[0].append(top_base_class)
	for i in range(5) :
		problem[0][1].append(final_similarities[i])
	problem[1].append('SVM')
	ans = os.popen("curl http://localhost:5002/"+query.replace(' ','-')).read()
	problem[1].append([(str(ans),'')])
	#print(problem)
	return 	render_template('manager_dash1.html',problems = problem,query_1 = query)




if __name__ == "__main__" :
	app.config['TEMPLATES_AUTO_RELOAD'] = True
	app.run(debug=True,port = 5003)
	


