class args:

    # training
    seed = 42
    epochs = 70 # Number of training epochs

    # data
    train_path = "../input/train_mainfile.csv"
    test_path =  "../input/test_mainfile.csv"
    sample_rate = 16000 # sample rate for the data/model features 
    batch_size = 20 # Batch size of traing
    num_workers = 4 # Number of workers used in data-loading
    window_size = .02 # Window size for spectrogram generation (seconds)
    window_stride = .01 # Window stride for spectrogram generation (seconds)
    window = "hamming" # Window type for spectrogram generation

    # model
    rnn_type = "lstm" # Type of RNN to use in model, rnn/gru/lstm are supported
    hidden_size = 1024 # Hidden size of RNN Layer
    hidden_layers = 5 # Number of RNN layers
    bidirectional = True # use BidirectionalRNNs.

    #optimizer
    learning_rate = 3e-4 # Initial Learning Rate
    weight_decay = 1e-5 # Initial Weight Decay
    momentum = 0.9
    eps = 1e-8
    beta = (0.9, 0.999) # Adam betas
    max_norm = 400 # Norm cutoff to prevent explosion of gradients
    learning_anneal: 1.1 # Annealing applied to learning rate after each epoch
        
        


