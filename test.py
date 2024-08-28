import pandas as pd
from nltk.tokenize import RegexpTokenizer
from selenium import webdriver
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
import pickle
import uvicorn
from fastapi import FastAPI
import joblib

# Read the csv files
legimate_df = pd.read_csv("legimate.csv")
phishing_url = pd.read_csv("verified_online.csv")
phish_data = pd.read_csv('phishing_site_urls.csv')


#Add new column
a = []
for i in legimate_df.index:
    a = 'no'

legimate_df = legimate_df.assign(verified = a)


legi = phish_data[phish_data.Label == 'good']
just_phish = phish_data[phish_data.Label == 'bad']

#rename
just_phish.rename(columns={'Label' : 'verified', 'URL' : 'url'},inplace=True)
legi.rename(columns={'Label' : 'verified', 'URL' : 'url'},inplace=True)

just_phish['verified'] = just_phish['verified'].map({'bad': 'yes', 'good': 'no'})
legi['verified'] = legi['verified'].map({'bad': 'yes', 'good': 'no'})

#index resetting
legi.reset_index(inplace=True)
just_phish.reset_index(inplace=True)

just_phish.drop(columns = {'index'},inplace=True)
legi.drop(columns = {'index'},inplace=True)

#Get random sample
just_phish = just_phish.sample(n = 50000, random_state = 12).copy()
just_phish = just_phish.reset_index(drop=True)

legi = legi.sample(n = 20000, random_state = 12).copy()
legi = legi.reset_index(drop=True)


phishing_url = phishing_url[['url', 'verified']]

legimate_df = legimate_df.sample(n = 5000, random_state = 12).copy()
legimate_df = legimate_df.reset_index(drop=True)

#merge
final_df = phishing_url.append(legimate_df, ignore_index=True)
final_df = final_df.append(just_phish, ignore_index=True)
final_df = final_df.append(legi, ignore_index=True)

#Delete http:// and https:// with regex
final_df['url'] = final_df['url'].str.replace(r'http:\/\/', '', regex=True)
final_df['url'] = final_df['url'].str.replace(r'https:\/\/', '', regex=True)

#Words tokenizer
tokenizer = RegexpTokenizer(r'[A-Za-z]+')

#Add new column
final_df['text_tokenized'] = final_df.url.map(lambda t: tokenizer.tokenize(t))


final_df['text_sent'] = final_df['text_tokenized'].map(lambda l: ' '.join(l))

#Chromedriver
browser = webdriver.Chrome(r"chromedriver.exe")

#Check any websites
list_urls = ['https://candybok.com/','https://bit.ly/3ISi1dw']
links_with_text = []
for url in list_urls:
    browser.get(url)
    soup = BeautifulSoup(browser.page_source,"html.parser")
    for line in soup.find_all('a'):
        href = line.get('href')
        links_with_text.append([url, href])

        print(line)

for url in list_urls:
    browser.get(url)
    soup = BeautifulSoup(browser.page_source,"html.parser")
    for line in soup.find_all('a'):
        href = line.get('href')
        links_with_text.append([url, href])

        print(line)


df = pd.DataFrame(links_with_text, columns=["from", "to"])


# create lr object
lr = LogisticRegression()


#Create pipeline 
from sklearn.pipeline import make_pipeline 

pipeline_ls = make_pipeline(
    CountVectorizer(
        tokenizer = RegexpTokenizer(
            r'[A-Za-z]+').tokenize,stop_words='english'),
             LogisticRegression())

#Create train test object then fit
trainX, testX, trainY, testY = train_test_split(final_df.url, final_df.verified)
pipeline_ls.fit(trainX,trainY)

#Take a pkl file
pickle.dump(pipeline_ls,open('final_test.pkl','wb'))

loaded_model = pickle.load(open('final_test.pkl', 'rb'))

result = loaded_model.score(testX,testY)

#Print accuracy
print("         -------------------------Accuracy of the model-----------------------")
print(result)

#Test with few websites
predict_yes = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_no = ['youtube.com/','https://www.youtube.com/watch?v=Lh2uRnbapVA&ab_channel=KemalSunalFilmleri','https://www.geeksforgeeks.org/','https://edition.cnn.com/']
loaded_model = pickle.load(open('final_test.pkl', 'rb'))

#Print the result
result = loaded_model.predict(predict_yes)
result2 = loaded_model.predict(predict_no)
print('This is a phishing website ' + result)
print("--"*20)
print('This is not a phishing website ' + result2)

#/////////      TESTING     //////////

#Create a simple website with FastAPI
app = FastAPI()

#Load the pkl file
phish_model = open('final_test.pkl','rb')
phish_model_ls = joblib.load(phish_model)

#Test any website using the pkl file
@app.get('/predict/{feature}')
async def predict(features):
	X_predict = []
	X_predict.append(str(features))
	y_Predict = phish_model_ls.predict(X_predict)
	if y_Predict == 'yes':
		result = "This is a Phishing Site"
	else:
		result = "This is not a Phishing Site"

	return (features, result)
if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1",port=8000)
    

