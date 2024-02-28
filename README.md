# Mental-Health-Chatbot
Introducing our Mental Health Chatbot: combining NLP, ML, HTML, CSS, and JavaScript. It offers empathetic conversations, personalized support, and resources. 

# If you want to clone this then do this step :

##Clone repo and create a virtual environment :
```
$ git clone https://github.com/harsuuu/Mental-Health-Chatbot
$ python3 -m venv venv
$ for windows -->  . \Scripts\activate
$ for macbook -->  . venv/bin/activate
```
Install dependencies
```
$ (venv) pip install Flask torch torchvision nltk
```
Install nltk package
```
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
```
Modify `intents.json` with different intents and responses for your Chatbot

Run
```
$ (venv) python train.py
```
This will dump data.pth file. And then run the following command to test it in the console.
```
$ (venv) python chat.py
```
- At last use this : 
```
$ (venv) python app.py
```
- Video :

https://github.com/harsuuu/Mental-Health-Chatbot/assets/107912873/e253618f-1035-41f9-9804-787fbdf89891

