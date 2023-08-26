import sys
import gensim
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import pandas as pd
import joblib
import pandas as pd
import os
import warnings
warnings.filterwarnings(action = 'ignore')
import gensim
from gensim.models import Word2Vec
import string
import numpy as np
from itertools import groupby, count
import re
import subprocess
import os.path
import sys
import logging
import joblib
from gensim.models.phrases import Phrases, Phraser
import pytesseract
import cv2
from pdf2image import convert_from_path
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import aspose.words as aw
import fitz
def _skills_in_box(image_gray,threshold=60):
  '''
  Function for identifying boxes and identifying skills in it: Given an imge path,
        returns string with text in it.
        Parameters:
            img_path: Path of the image
            thresh : Threshold of the box to convert it to 0
  '''
  img = image_gray.copy()
  thresh_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
  # Blur the image
  blur = cv2.GaussianBlur(thresh_inv,(1,1),0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  # find contours
  contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
  mask = np.ones(img.shape[:2], dtype="uint8") * 255
  available = 0
  for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    if w*h>1000:
        cv2.rectangle(mask, (x+5, y+5), (x+w-5, y+h-5), (0, 0, 255), -1)
        available = 1

  res = ''
  if available == 1:
    res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    res_final[res_final<=threshold]=0
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    res_fin = cv2.filter2D(src=res_final, ddepth=-1, kernel=kernel)
    vt = pytesseract.image_to_data(255-res_final,output_type='data.frame')
    vt = vt[vt.conf != -1]
    res = ''
    for i in vt[vt['conf']>=43]['text']:
      res = res + str(i) + ' '
  print(res)
  return res

def _image_to_string(img):
  '''
  Function for converting images to grayscale and converting to text: Given an image path,
  returns text in it.
  Parameters:
      img_path: Path of the image
  '''
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  res = ''
  string1 = pytesseract.image_to_data(img,output_type='data.frame')
  string1 = string1[string1['conf'] != -1]
  for i in string1[string1['conf']>=43]['text']:
    res = res + str(i) + ' '
  string3 = _skills_in_box(img)
  return res+string3

def _pdf_to_png(pdf_path):
    '''
    Function for converting pdf to image and saves it in a folder and
    convert the image into string
    Parameter:
        pdf_path: Path of the pdf
    '''
    string = ''
    images = convert_from_path(pdf_path)
    for j in tqdm(range(len(images))):
        # Save pages as images in the pdf
        image = np.array(images[j])
        string += _image_to_string(image)
        string += '\n'
    return string
def ocr(paths):
    '''
    Function for checking the pdf is image or not. If the file is in .doc it converts it into .pdf
    if the pdf is in image format the function converts .pdf to .png
    Parameter:
        paths: list containg paths of all pdf files
    '''
    text = ""
    res = ""
    try:
        doc = fitz.open(paths)
        for page in doc:
            text += page.get_text()
        if len(text) <=10 :
            res = _pdf_to_png(paths)
        else:
            res = text
    except:
        doc = aw.Document(paths)
        doc.save("Document.pdf")
        doc = fitz.open("Document.pdf")
        for page in doc:
            text += page.get_text()
        if len(text) <=10 :
            res = _pdf_to_png("Document.pdf")
        else:
            res = text
        os.remove("Document.pdf")
    return res

def to_la(L):
  k=list(L)
  l=np.array(k)
  return l.reshape(-1, 1)

def cos(A, B):
  dot_prod=np.matmul(A,B.T)
  norm_a=np.reciprocal(np.sum(np.abs(A)**2,axis=-1)**(1./2))
  norm_b=np.reciprocal(np.sum(np.abs(B)**2,axis=-1)**(1./2))
  norm_a=to_la(norm_a)
  norm_b=to_la(norm_b)
  k=np.matmul(norm_a,norm_b.T)
  return list(np.multiply(dot_prod,k))

def check(path,skills,l2,w2v_model1,phrases,pattern):
  text = ocr(path)
  text = re.sub(r'[^\x00-\x7f]',r' ',text)
  text = text.lower()
  text = re.sub("\\\|,|/|:|\)|\("," ",text)
  t2 = text.split()
  l_2=l2.copy()
  match=list(set(re.findall(pattern,text)))
  sentences=phrases[t2]
  resume_skills_dict={}
  res_jdskill_intersect=list(set(sentences).intersection(set(l_2)))
  if(len(match)!=0):
    for k in match:
      k=k.replace(' ','_')
      resume_skills_dict[k]=1
      try:
        l_2.remove(k)
      except:
        continue
  l6=list(set(l_2).intersection(skills['word']))
  l6_minus_skills=list(set(l_2).difference(skills['word']))
  for i in l6_minus_skills:
    resume_skills_dict[i]=0
  if(len(l6)==0):
    return resume_skills_dict
  l4=list(set(sentences).intersection(skills['word']))
  arr1 = np.array([w2v_model1.wv.get_vector(i) for i in l6])
  arr2 = np.array([w2v_model1.wv.get_vector(i) for i in l4])
  similarity_values=cos(arr1,arr2)
  count=0
  for i in similarity_values:
    k=list(filter(lambda x: x<0.38, list(i)))
    if(len(k)==len(i)):
      resume_skills_dict[l6[count]]=0
    else:
      resume_skills=[s for s in range(len(i)) if(i[s])>0.38]
      resume_skills_dict[l6[count]]=1
    count+=1
  return resume_skills_dict

def Convert(string):
    li = list(string.split())
    return list(set(li))

def preprocess(string):
  string = string.replace(",",' ')
  string= string.replace("'",' ')
  string = Convert(string)
  return string

if __name__ == "__main__":
   #Arg 1 = vocabulary, Arg 2 = model, Arg 3 = phrases object, Arg 4 = JD's Mandatory Skills, Arg 5 = Resume Path
   argv = sys.argv[1:]
   w2v_model1 = joblib.load(argv[0])
   skills=pd.read_csv(argv[1])
   mapper = {}
   underscore=[]
   jd_skills=argv[3]
   jd_skills=" ".join(jd_skills.strip().split())
   jd_skills=jd_skills.replace(', ',',')
   pattern=jd_skills.replace(',','|').lower()
   for i in jd_skills.split(','):
    if '_' in i:
      underscore.append(i)
      mapper[i.lower().replace('_',' ')] = i
   jd_skills=jd_skills.replace(' ','_')
   jd_skills=jd_skills.replace(',',', ')
   for i in jd_skills.split(', '):
    if i not in underscore:
      if '_' in i:
        mapper[i.lower().replace('_',' ')] = i.replace('_',' ')
      elif '-' in i:
        mapper[i.lower().replace('-',' ')] = i
      else:
        mapper[i.lower()] = i
   jd_skills=jd_skills.replace('-','_')
   phrases=Phrases.load(argv[2])
   lines = [preprocess(jd_skills.lower().rstrip())]
   phrases=Phrases.load(argv[2])
   final_jd_skills=list(set(lines[0]).intersection(skills['word']))
   path = argv[4]
   res=check(path,skills,lines[0],w2v_model1,phrases,pattern)
   print('skills_matched :',res)
   perc = ((sum(res.values()))*100)//(len(res))
   print(f"Percentage of match: {perc}%")
   if perc > 60:
    print(f"This resume is eligible for the role as it matches {perc} which is more than 60 percent of the job requirements")
   else:
    print(f"This resume is not eligible for the role as it matches {perc} which is less than 60 percent of the job requirements")
