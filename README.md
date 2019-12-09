
This repo contains Keras implementation of parameter-reduced LSTMs. Base implementation of LSTM in Keras 2.2 is used to generate inherited layers of reduced gated and cell parameters. Sample script for importing the slim22 module and executing a training/testing example is provided.

LSTM uses gating mechanism to control the signal flow. It possess three gating signals driven by 3 main components, namely, the external input signal, the previous state, and a bias. We have proposed nine variants of the LSTM model, aiming at reducing the number of (adaptive) parameters in each gate, and thus reduce computational cost. 

As can be seen, Setting learning rate equal to 0.0001 results in an almost fluctuation-free profile with close results to standard LSTM.


![comparison](https://raw.githubusercontent.com/atrakriv/Slim-LSTMs/master/i1sig0001.png)


For more information please refer to  
[Simplified Long Short-term Memory Recurrent Neural Networks: part I](https://arxiv.org/abs/1707.04619)  
[Simplified Long Short-term Memory Recurrent Neural Networks: part II](https://arxiv.org/abs/1701.05923)  
[Simplified Long Short-term Memory Recurrent Neural Networks: part III](https://arxiv.org/abs/1707.04626).



# Requirements
python 3.6  
TensorFlow 1.13.1  
Keras 2.2.4  

# Usage
python slim22-driver.py  
You can set corresponding parameters in config.py or pass them as arguments:  
python slim22-driver.py LSTM4 32 100 0.0001  
in which LSTM4 is the model we wanted to run, 32 is batch-size, and 100 is number of epochs and 0.0001 is learning parameter.

# Citation
If you use this code in your project/paper/research and would like to cite this work, please use the below.

@article{akandeh2017simplified,
	title="Simplified Long Short-term Memory Recurrent Neural Networks: part I",
	author="Atra {Akandeh} and Fathi M. {Salem}",
	journal="World Congress in Computer Science, Computer Engineering,
    \& Applied Computing",
	year="2017"
}

@article{akandeh2017simplified,
	title="Simplified Long Short-term Memory Recurrent Neural Networks: part II",
	author="Atra {Akandeh} and Fathi M. {Salem}",
	journal="World Congress in Computer Science, Computer Engineering,
    \& Applied Computing",
	year="2017"
}

@article{akandeh2017simplified,
	title="Simplified Long Short-term Memory Recurrent Neural Networks: part III",
	author="Atra {Akandeh} and Fathi M. {Salem}",
	journal="World Congress in Computer Science, Computer Engineering,
    \& Applied Computing",
	year="2017"
}

