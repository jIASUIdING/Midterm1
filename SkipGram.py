import numpy as np
#import matplotlib.pyplot as plt 
from collections import defaultdict


class word2vec():
    def __init__(self, windowSize, dimension, epochs, learningRate):
        self.window = windowSize    #window size
        self.n = dimension          #embedding dimension
        self.epochs = epochs        #number of training epochs
        self.alpha = learningRate   #learning rate 
        pass
    
    
    #GENERATE CORPUS FROM TEXT FILE
    def text_to_corpus(self, filename):
        corpus=[]
        with open(filename, "r") as training_file:
            sentence_list = training_file.readlines()
            for sentence in sentence_list:
                corpus.append(sentence.split(" "))                
                
        return corpus
    
    
    # GENERATE TRAINING DATA VECTOR MATRIX (self, filename)
    # RETURNS [Center Word Vector[Context Word Vectors]] sorted 
    def generate_training_data(self, filename):
        
        #GENERATE CORPUS FROM TEXT FILE
        corpus = self.text_to_corpus(filename)
        
        # GENERATE WORD COUNTS
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(word_counts.keys()),reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):
                
                #w_target  = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j!=i and j<=sent_len-1 and j>=0:
                        w_context.append(self.word2onehot(sentence[j]))
                w_context = np.sum(w_context, axis = 0)
                training_data.append([w_target, w_context])
        return np.array(training_data)
    
    
    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec


    # LIKELIHOOD FUNCTION P（u_o/v_c)
    def probability(self, w_c):
        v_c = np.dot(self.v.T, w_c)     #(d, 1)Transpose matrix v and picks the column vector of the center word 
        p = np.dot(self.u, v_c)         #(N, 1)u * v_c
        p = self.softmax(p)             #(N, 1)apply softmax
        return p, v_c
    
    
    # SOFTMAX FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    
    # GRADIENT FUNCTION
    def gradient(self, p, v_c, w_t)    :
        p.shape = (p.size, 1)               #N x 1
        v_c.shape = (v_c.size, 1)           #d x 1
        w_t.shape = (w_t.size, 1)           #N x 1
        j_u = -v_c + np.dot(v_c, p.T)       #d x N
        j_v = -self.u + np.dot(p.T, self.u) #N x d
        
        #multiply by the context array to make the gradient of irrelavent words 0
        j_u = np.multiply(j_u, w_t.T)
        j_v = np.multiply(j_v, w_t)

        return j_u.T, j_v
    
    # GRADIENT DESCENT： apply gradient descent till the gradient is at minimum
    def gradient_descent(self, w_c, w_t, j_u, j_v):
        min_grad = np.abs(sum(sum(j_u)))+np.abs(sum(sum(j_v)))
        self.u -= self.alpha * j_u
        self.v -= self.alpha * j_v
        p, v_c = self.probability(w_c)
        j_u, j_v = self.gradient(p, v_c, w_t)
        cur_grad = np.abs(sum(sum(j_u)))+np.abs(sum(sum(j_v)))
        if cur_grad < min_grad:
            self.gradient_descent(w_c, w_t, j_u, j_v)
        return None

    
    # Fuction to calculate the sum log-likelihood function given a center word
    def negative_log_likelihood(self, w_c, w_t):
        p, v_c = self.probability(w_c)      #get probability vector
        p_u = np.multiply(p, w_t)           #extract context words only
        j = -sum(np.log(p_u[p_u !=0]))      #negative log-likelihood function
        return j
    
    
    #TRAIN W2V MODEL
    def train(self, training_data):
        
        #Step 1 INITIALIZE WEIGHT MATRICES
        self.u = np.random.uniform(0, 1, (self.v_count, self.n))    # context matrix
        self.v = np.random.uniform(0, 1, (self.v_count, self.n))     # center matrix
    
        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            j = 0
            error = 0
            
            # CYCLE THROUGH EACH CENTER WORD
            for w_c, w_t in training_data:  #center vector, context matrix
                j = 0
                
                # Step 2 MAXIMUM LIKELIHOOD
                p, v_c = self.probability(w_c)
                
                # Calculate Error
                error += np.sum(np.square(np.subtract(p, w_c)))/2/self.v_count
            
                # Step 3 CALCULATE GRADIENT
                j_u, j_v = self.gradient(p, v_c, w_t)
                
                # Step 4 GRADIENT DESCENT
                self.u -= self.alpha * j_u
                self.v -= self.alpha * j_v
                #self.gradient_descent(w_c, w_t, j_u, j_v)
                
                # Step 5 Compute J
                j += self.negative_log_likelihood(w_c, w_t)                
                
            print(i + 1, error)

        return self.u + self.v


# set the seed for reproducibility 
np.random.seed(0)                      


# INITIALIZE W2V MODEL: windowSize, dimension, epochs, learningRate
w2v = word2vec(2,128,50,0.01)

# generate training data
training_data = w2v.generate_training_data('training data.txt')

# train word2vec model
embedding_matrix = w2v.train(training_data)
