# =================================================================
# libraries
# =================================================================
from   numpy             import array
from   keras.models      import Sequential
from   keras.layers      import Dense
import pandas            as     pd
import matplotlib.pyplot as     plt
import math
import toml
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# =================================================================
# functions
# =================================================================
# load file
def load_file(nameFile):
    # print("Hello")
    # print(os.path.dirname(__file__) + '/' + nameFile)
    dfData = pd.read_csv(
        filepath_or_buffer = os.path.dirname(__file__) + '/' + nameFile,
        sep                = ';',
        header             = 0,
        encoding           = 'utf-8')
    
    return dfData

# split a univariate sequence into samples
def split_sequenceUStep(sequence, n_steps_in):
    X, y = list(), list()
    
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        
    return array(X), array(y)

# split a multivariate sequence into samples
def split_sequenceMStep(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
    
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
        
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
            
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
        
	return array(X), array(y)
    
# split a multivariate sequence into samples
def split_sequenceMStep1(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    
    out_end_ix = 0
    for i in range(len(sequence)):
        # find the end of this pattern
        start_ix = i + out_end_ix//10
        end_ix   = start_ix + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[start_ix:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    return array(X), array(y)    

# =================================================================
# main function
# =================================================================
def main():
    # reading toml file. It is the configuration file
    configuration_toml = toml.load(os.path.dirname(__file__) + '/' + 'configuration.toml') 
    print(configuration_toml)
    
    # reading input: historic time series
    dfData = load_file(configuration_toml['dataName'])
    
    # number of time n_steps_in
    n_steps_in = configuration_toml['n_steps_in']
    
    # number of time n_steps_out
    n_steps_out = configuration_toml['n_steps_out']
    
    # option one-step or multi-step ahead
    multistep = configuration_toml['multistep']
    
    # univariate time series
    for var in configuration_toml['variables']:
        raw_seq  = dfData[var].tolist()
        raw_seq1 = raw_seq.copy()
        
        # training set
        raw_seq  = raw_seq[0:math.floor(0.8*len(raw_seq))]


        if multistep:
            # split into samples
            X, y = split_sequenceMStep1(raw_seq, n_steps_in, n_steps_out)
            
            # summarize the data
            for i in range(len(X)):
                print(X[i], y[i])
            
            # define model
            model = Sequential()
            model.add(Dense(200, activation='relu', input_dim=n_steps_in))
            model.add(Dense(200, activation='sigmoid'                   ))
            model.add(Dense(n_steps_out))
            model.compile(optimizer='adam', loss='mse')
            
            # fit model
            model.fit(X, y, epochs=200, verbose=0)
            i = 1
            while i < 100:
               model.fit(X, y, epochs=100, verbose=0)
               if i > 80:
                  print(model.get_weights()[0])
               i += 1

            weights = model.get_weights()
            print(weights)
            
            # demonstrate prediction
            # split into samples
            print('\n test \n')
            X, y = split_sequenceMStep1(raw_seq1, n_steps_in, n_steps_out)
            for i in range(len(X)):
                print(X[i], y[i])
            
            print('\n prediction \n')
            x_input = X[-1]
            print(x_input)
            x_input = array(x_input)
            x_input = x_input.reshape((1, n_steps_in))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            
            # split into samples
            X, y = split_sequenceUStep(raw_seq1, n_steps_in)
            predictions = list()
            for x_input in X:
               x_input = array(x_input)
               x_input = x_input.reshape((1, n_steps_in))
               yhat = model.predict(x_input, verbose=0)
               predictions.append(yhat[0][0])

            training = raw_seq [n_steps_in:len(raw_seq )]
            original = raw_seq1[n_steps_in:len(raw_seq1)]
            plt.plot(original   , label = "original"   )
            plt.plot(training   , label = "train"      )
            plt.plot(predictions, label = "predictions")
            plt.legend()
            plt.show()
            
        else:
            # split into samples
            X, y = split_sequenceUStep(raw_seq, n_steps_in)

            # summarize the data
            # for i in range(len(X)):
                # print(X[i], y[i])
                
            # define model
            model = Sequential()
            model.add(Dense(200, activation='relu', input_dim=n_steps_in))
            model.add(Dense(200, activation='sigmoid'                   ))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer='adam') # loss = 'mse' loss='mean_squared_logarithmic_error'

            # fit model
            model.fit(X, y, epochs=200, verbose=0)
            i = 1
            while i < 100:
               model.fit(X, y, epochs=100, verbose=0)
               if i > 80:
                  print(model.get_weights()[0])
               i += 1

            weights = model.get_weights()
            print(weights)

            # demonstrate prediction
            # split into samples
            X, y = split_sequenceUStep(raw_seq1, n_steps_in)
            predictions = list()
            for x_input in X:
               x_input = array(x_input)
               x_input = x_input.reshape((1, n_steps_in))
               yhat = model.predict(x_input, verbose=0)
               predictions.append(yhat[0][0])

            training = raw_seq [n_steps_in:len(raw_seq )]
            original = raw_seq1[n_steps_in:len(raw_seq1)]
            plt.plot(original   , label = "original"   )
            plt.plot(training   , label = "train"      )
            plt.plot(predictions, label = "predictions")
            plt.show()


# =============================================================================================================================================================
# script execution
# =============================================================================================================================================================
if __name__ == '__main__':
    main()



