import tensorflow as tf
import pandas as pd
import numpy as np
import numpy.core.multiarray

#import matplotlib.pyplot as plt

#from tensorflow.contrib import learn

#from sklearn.pipeline import Pipeline
#from sklearn import datasets, linear_model

from tkinter import *

print ('Initiallize Program')

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("Bacteria Classification Program")
        self.pack(fill=BOTH, expand=1)

        bar = Menu(self.master)
        self.master.config(menu = bar)

        Label_Main = Label(self.master, text='Please Enter Data Below', fg='red')
        Label_Main.place(x=120, y=2)

        file = Menu(bar)
        file.add_command(label='Exit', command=self.client_exit)
        bar.add_cascade(label='File', menu=file)

        xdim = 150
        xdimL = 5

        mEntry1 = Entry(self,textvariable=ent1)
        mEntry1.place(x=xdim,y=30)

        Label1 = Label(self.master, text = 'Biotic Relationship')
        Label1.place(x=xdimL, y=30)

        mEntry2 = Entry(self,textvariable=ent2)
        mEntry2.place(x=xdim,y=60)

        Label2 = Label(self.master, text = 'Cell Arrangement')
        Label2.place(x=xdimL, y=60)

        mEntry3 = Entry(self,textvariable=ent3)
        mEntry3.place(x=xdim, y=90)

        Label3 = Label(self.master, text = 'Cell Shape')
        Label3.place(x=xdimL, y=90)

        mEntry4 = Entry(self,textvariable=ent4)
        mEntry4.place(x=xdim,y=120)

        Label4 = Label(self.master, text = 'Ecosystem')
        Label4.place(x=xdimL, y=120)

        mEntry5 = Entry(self,textvariable=ent5)
        mEntry5.place(x=xdim, y=150)

        Label5 = Label(self.master, text = 'Ecosystem Category')
        Label5.place(x=xdimL, y=150)

        mEntry6 = Entry(self,textvariable=ent6)
        mEntry6.place(x=xdim, y=180)

        Label6 = Label(self.master, text = 'Ecosystem Subtype')
        Label6.place(x=xdimL, y=180)

        mEntry7 = Entry(self,textvariable=ent7)
        mEntry7.place(x=xdim, y=210)

        Label2 = Label(self.master, text = 'Ecosystem Type')
        Label2.place(x=xdimL, y=210)

        mEntry8 = Entry(self,textvariable=ent8)
        mEntry8.place(x=xdim, y=240)

        Label8 = Label(self.master, text = 'Energy Source')
        Label8.place(x=xdimL, y=240)

        mEntry9 = Entry(self,textvariable=ent9)
        mEntry9.place(x=xdim, y=270)

        Label9 = Label(self.master, text = 'Gram Staining')
        Label9.place(x=xdimL, y=270)

        mEntry10 = Entry(self,textvariable=ent10)
        mEntry10.place(x=xdim, y=300)

        Label10 = Label(self.master, text = 'Habitat')
        Label10.place(x=xdimL, y=300)

        mEntry11 = Entry(self,textvariable=ent11)
        mEntry11.place(x=xdim, y=330)

        Label11 = Label(self.master, text = 'Motility')
        Label11.place(x=xdimL, y=330)

        mEntry12 = Entry(self,textvariable=ent12)
        mEntry12.place(x=xdim, y=360)

        Label12 = Label(self.master, text = 'Oxygen Requirement')
        Label12.place(x=xdimL, y=360)

        mEntry13 = Entry(self,textvariable=ent13)
        mEntry13.place(x=xdim, y=390)

        Label13 = Label(self.master, text = 'Sporulation')
        Label13.place(x=xdimL, y=390)

        mEntry14 = Entry(self,textvariable=ent14)
        mEntry14.place(x=xdim, y=420)

        Label14 = Label(self.master, text = 'Temperature Range')
        Label14.place(x=xdimL, y=420)

        entButton = Button(self, text="Run Network", command=self.enter_value)
        entButton.place(x=130,y=460)


    def client_exit(self):
        exit()

    def enter_value(self):
        mtext1 = ent1.get()
        mtext2 = ent2.get()
        mtext3 = ent3.get()
        mtext4 = ent4.get()
        mtext5 = ent5.get()
        mtext6 = ent6.get()
        mtext7 = ent7.get()
        mtext8 = ent8.get()
        mtext9 = ent9.get()
        mtext10 = ent10.get()
        mtext11 = ent11.get()
        mtext12 = ent12.get()
        mtext13 = ent13.get()
        mtext14 = ent14.get()


        test_X = np.array([mtext1,mtext2,mtext3,mtext4,mtext5,mtext6,mtext7,mtext8,mtext9,
                          mtext10,mtext11,mtext12,mtext13,mtext14])#.reshape(1,14)


        print ('Entry:', test_X )


        df = pd.read_csv('C:/Users/Bharat purohit/Desktop/export.csv')
        data_ = df.drop(['ID','Species'], axis=1)


        n_classes = data_["Phylum"].nunique()

        dim = 14
        learning_rate = 0.0001
        display_step = 10
        n_hidden_1 = 2000
        n_hidden_2 = 1500
        n_hidden_3 = 1000
        n_hidden_4 = 500

        X = tf.placeholder(tf.float32, [None, dim])

        train_set = data_.sample(frac=1)
        test_set = data_.loc[~data_.index.isin(train_set.index)]

        train_size = train_set.size

        inputY_test = pd.get_dummies(test_set['Phylum'])
        inputY_train = pd.get_dummies(train_set['Phylum'])

        train_X = train_set.iloc[:train_size, 5:].as_matrix()
        train_X = pd.DataFrame(data=train_X)
        train_X = train_X.fillna(value=0).as_matrix()

        train_Y = inputY_train.as_matrix()
        train_Y = pd.DataFrame(data=train_Y)
        train_Y = train_Y.fillna(value=0).as_matrix()

        #test_X = test_set.iloc[:, 5:].as_matrix()
        #test_X = pd.DataFrame(data=test_X)
        #test_X = test_X.fillna(value=0).as_matrix()
        test_X = pd.DataFrame(data=test_X)
        test_X = test_X.replace('', np.nan)
        test_X = test_X.fillna(value=0).as_matrix()
        test_X.resize(1,14)


        test_Y = inputY_test.as_matrix()
        test_Y = pd.DataFrame(data=test_Y)
        test_Y = test_Y.fillna(value=0).as_matrix()

        n_samples = train_Y.size
        total_len = train_X.shape[0]
        n_input = train_X.shape[1]
        batch_size = 5


        W = tf.Variable(tf.zeros([dim, n_classes]))
        b = tf.Variable(tf.zeros([n_classes]))


        def multilayer_perceptron(x, weights, biases):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)

            # Hidden layer with RELU activation
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)

            # Hidden layer with RELU activation
            layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
            layer_3 = tf.nn.relu(layer_3)

            # Hidden layer with RELU activation
            layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
            layer_4 = tf.nn.relu(layer_4)

            # Output layer with linear activation
            out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
            return out_layer

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
            'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
            'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
            'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
            'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
            'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
        }

        # Construct model
        pred = multilayer_perceptron(X, weights, biases)
        pred1 = tf.argmax(pred,1)

        y = tf.placeholder(tf.float32, [None, n_classes])
        #cost = -tf.reduce_sum(y*tf.log(tf.clip_by_value(pred,1e-10,1.0)))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        optimizer =  tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        hm_epochs = 20

        tf.set_random_seed(1234)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init) 

            for epoch in range(hm_epochs):
                avg_cost = 0
                total_batch = int(total_len/batch_size)
                for i in range(total_batch-1):
                    batch_x = train_X[i*batch_size:(i+1)*batch_size]
                    batch_y = train_Y[i*batch_size:(i+1)*batch_size]

                    _, c, p = sess.run([optimizer, cost, pred], feed_dict={X: batch_x,
                                                                                   y: batch_y})
                    avg_cost += c / total_batch

                label_value = batch_y
                estimate = p
                err = label_value-estimate

                if epoch % display_step == 0:
                    print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
                    print ("[*]----------------------------------------------------")
                    for i in xrange(3):
                        print ("label value:", label_value[i], \
                                "estimated value:", estimate[i])
                    print ("=======================================================")

            print ("Optimization Finished!")

            #Test model
            #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            #Calculate accuracy
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            #print ("Accuracy:", accuracy.eval({X: test_X, y: test_Y}))


            feed_dict = {X: test_X}
            classification = pred.eval(feed_dict)
            print ("Network Prediction:", classification)
            print ("Classification: ", np.argmax(classification) + 1)

            #probabilities=pred1
            #print "probabilities", probabilities.eval(feed_dict={X: test_X}, session=sess)


root = Tk()

ent1 = StringVar()
ent2 = StringVar()
ent3 = StringVar()
ent4 = StringVar()
ent5 = StringVar()
ent6 = StringVar()
ent7 = StringVar()
ent8 = StringVar()
ent9 = StringVar()
ent10 = StringVar()
ent11 = StringVar()
ent12 = StringVar()
ent13 = StringVar()
ent14 = StringVar()

root.geometry("400x600")
app = Window(root)
root.mainloop()
