import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud

#Importing dataset using pandas 
df = pd.read_csv("Uncleaned_DS_jobs.csv")

#dropping columns not necessary for analysis
df = df.drop(["Founded","Type of ownership","Competitors","Revenue","Size"],axis=1)

print(df["Salary Estimate"].dtype)
# Word to remove
word_to_remove = '(Glassdoor est.)'

# Remove the word from the Salary column
def clean_salary(salary_str):
    try:
        # Remove extra characters and split the string
        salary_str = salary_str.split('(')[0].strip()
        min_salary, max_salary = [int(s.strip('K').strip('$').strip()) for s in salary_str.split('-')]
        result = f"{min_salary} - {max_salary}"
        return result
    except (ValueError, AttributeError, IndexError):
        return None

df['Salary Estimate'] = df['Salary Estimate'].apply(clean_salary)

# Split the 'Salary Estimate' column into two separate columns
df[['Min Salary', 'Max Salary']] = df['Salary Estimate'].str.split('-', expand=True)

# Convert the columns to numeric type
df['Min Salary'] = pd.to_numeric(df['Min Salary'])
df['Max Salary'] = pd.to_numeric(df['Max Salary'])

#create average column
df['Avg Salary']= ((df['Max Salary'] + df['Min Salary'])/2).astype(int) 

# Remove '\n' and numbers from the 'Company' column
df['Company Name'] = df['Company Name'].str.replace('\n', '')  # Remove '\n'
df['Company Name'] = df['Company Name'].str.replace(r'\d+(\.\d+)?', '')  # Remove numbers
# Split the 'Location' column into 'City' and 'State'
# Define a regular expression pattern to match state abbreviation
pattern = r'\b[A-Z]{2}\b'

# Extract state abbreviation from 'Location' column using regular expression
df['State'] = df['Location'].apply(lambda x: re.findall(pattern, x)[0] if re.findall(pattern, x) else None)

# Extract city by removing state abbreviation and comma from 'Location' column
df['City'] = df['Location'].str.replace(pattern, '').str.strip(', ')

# Text preprocessing
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())  # Tokenize and lowercase
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]  # Remove non-alphabetic and stopwords
    return filtered_words

df['Processed_Description'] = df['Job Description'].apply(preprocess_text)

# Word frequency analysis
all_words = [word for desc_words in df['Processed_Description'] for word in desc_words]
fdist = FreqDist(all_words)
print("Most common words:")
print(fdist.most_common(10))

# Word cloud visualization
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Descriptions')
plt.show()

print(df[:10])