<h1 style="color:green">Business/Real World Problem</h1>

<h2 style="color:blue">Description</h2>


<p style="color:gray;font-size:15px">Artificially intelligent machines are becoming smarter in every day. Deep learning and machine learning techniques enable machines to perform many tasks at the human level. In some cases, they even surpass human abilities. Machine intelligence can analyze big data faster and more accurately than a human possibly can. Even though they cannot think yet, they see, sometimes better than humans (read our computer vision and machine vision articles), they can speak, and they are also good listeners. Known as “automatic speech recognition” (ASR), “computer speech recognition”, or just “speech to text” (STT) enables computers to understand spoken human language.</p>

<p style="color:gray;font-size:15px"><b>Note:</b>Speech recognition and speaker recognition are different terms. While speech recognition is to understand what is told, speaker recognition is to know the speaker instead of understanding the context of the speech that can be used for security measures. These two terms are confusing and voice recognition is often used for both.</p> 


<h2 style="color:blue">Problem Statement</h2>
<p style="color:gray;font-size:15px">This is <b>END TO END</b> model, given audio data that convert Analog-to-Digital using (ADC) converter, then extract features form audio using some Signinal-Processing algorithms like Sort-Time-Fourier-Transform(STFT), Then using some Deep-Learning based techniques (like CNN's, LSTM's and GRU's) convert audio features into text representation</p>

<h2 style="color:blue">Source/Useful Links</h2>

<p style="color:gray;font-size:15px"> Some articles and reference blogs about ths problem statement</p>

<p style="color:gray;font-size:16px"> We are referred to some research papers and open source projects/repositories maintained below </p>

<h3 style="color:red">Research Papers</h3>

<ul style="color:gray">
  <li><a href="https://arxiv.org/pdf/1512.02595.pdf">Deep Speech 2: End-to-End Speech Recognition in
English and Mandarin</a></li>
  <li><a href="https://arxiv.org/pdf/1904.03288.pdf">Jasper: An End-to-End Convolutional Neural Acoustic Model</a></li>
  <li><a href="https://arxiv.org/pdf/1508.01211.pdf">Listen,Attend and Spell</a></li>
</ul>

<h3 style="color:red">Open Source Projects</h3>

<ul style="color:gray">
  <li><a href="https://arxiv.org/pdf/1512.02595.pdfhttps://github.com/SeanNaren/deepspeech.pytorch">deepspeech.pytorch</a></li>
  <li><a href="https://github.com/NVIDIA/OpenSeq2Seq/tree/master/open_seq2seq">NVIDIA OpenSeq2Seq</a></li>
  <li><a href="https://github.com/foamliu/Listen-Attend-Spell-v2">Listen,Attend and Spell</a></li>
</ul>


<h2 style="color:blue">Objective</h2>

<p style="color:gray;font-size:16px">Our objective is to build End-To-End Speech Recognition System using existing research and Try verious architectures, Then find out which one works better for us.</p>


<h2 style="color:blue">Constrains</h2>

<ul style="color:gray">
    <li><b style="color:red">Latency:</b> Given a audio (.wav) file the model predict Text what's spoken in that audio file, depending on application what you are using latency important</li>
    <li><b style="color:red">Interpretability:</b> As long as the speaker has spoken he/she wanted to check what are they spoken, they don't what to know how the model predicting that, so in this case, interpretability not importent.</li>
  <li><b style="color:red">Word Error Rate: </b> Word error rate (WER) is a common metric of the performance of a speech recognition or machine translation system. The general difficulty of measuring performance lies in the fact that the recognized word sequence can have a different length from the reference word sequence (supposedly the correct one).</li></ul>

<p><b style="color:green"> Our goal is to train best model that gives low Word Error Rate(WER) </b></p>
