Download anaconda from this sight.

https://www.anaconda.com/products/distribution
once installed open anaconda prompt and create 2 environments with these commands.

conda create -n mypython385 python=3.8.5
conda create -n mypython35 python=3.5

So now check how much environment we have by using 
conda info --envs
see both environments are there of name mypython385 and mypython35

if yes then activate python3.5 version with command
conda activate mypython35
then go to directory of files copy directory 
cd path offiles
Then dir  it will show you all files and run command
pip install –r requirements.txt
then run
pip install torch===1.3.1 torchvision==0.4.1 –f https://download.pytorch.org/whl/torch_stable.html
pip install pyyaml

### Download Dataset
* By default, we expect the data to be stored in `./data`. replace train and test files which I will send you.



### Train and inference
You can change epoch in config_trian.yml num_epoch: 50 
And set the paths according to your system 
Run `python train.py`. Training result will be output to `./training_output`.
You can change epoch in config_Inference.yml num_epoch: 50 and `test_model_path: 'Your Model Path'`. Run `python infernce.py`.Generated responses will be output to `./inference_output`.

### Concept Selection

For concept selection, edit `config_sort.yml`, `test_model_path: 'Your Selector Path' `. Run `python sort.py`. The sorted two-hop concepts will be output to `selected_concept.txt` with ascending order.



Now you have to activate another environment which is python385
conda activate python385
Then install these commands
Install these versions

pip install tensorflow==2.9.0
pip install tensorflow-gpu==2.8.0
pip install torch==1.9.1 torchvision==0.10.1  -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.19.2
pip install typer==0.4.0
pip install typing-extensions==3.7.4.3
pip install tqdm==4.50.2
pip install pandas
pip install shutil
pip install pathlib
pip install sklearn
pip install glob
pip install pickle


Run command
python check.py
then open Trainchat.py
python Trainchat.py
once done then 
python Bot.pt  
"here bot will work you can use chat for exit press y"

you can copy files ref.txt and hyp.txt
for evaluation you have to go inside rouge folder. 
Go to same directory on console
Run python setup.py install

Then go inside tests folder 
Here paste files ref.txt and hyp.txt. Then run "Python test.py"
It will give you rouge-1 rouge-2 and rouge-l with precision recall and f score.
After that come back 2 step then go inside folder DSTC7-End-to-End-Conversation-Modeling-master and then evaluation and src folder. Here also paste ref.txt and hyp.txt.
Come back 1 step and install perl
https://strawberryperl.com/
download stawberryperl 64 bit or 32 bit
install then run 
python score.py and it will give you results


