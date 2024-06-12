from subprocess import call
import speech_recognition as sr
import serial
import RPi.GPIO as GPIO      
import os, time
from urllib.request import urlopen
import nltk
import feedparser as fp
import newspaper
from newspaper import Article
from bs4 import BeautifulSoup as soup
import sys, os, subprocess, picamera, json
import json
import chainer
import argparse
import numpy as np
import math
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import serializers
import cv2
import serial              
from time import sleep
import sys
lat=None
longi=None
received_data=None

sys.path.append('./code')
from CaptionGenerator import CaptionGenerator

#gps start
ser = serial.Serial ("/dev/ttyS0")
gpgga_info = "$GPGGA,"
GPGGA_buffer = 0
NMEA_buff = 0

def convert_to_degrees(raw_value):
    decimal_value = raw_value/100.00
    degrees = int(decimal_value)
    mm_mmmm = (decimal_value - int(decimal_value))/0.6
    position = degrees + mm_mmmm
    position = "%.4f" %(position)
    return position

def location():
     GPGGA_data_available = received_data.find(gpgga_info)   #check for NMEA GPGGA string
     if (GPGGA_data_available>0):
            GPGGA_buffer = received_data.split("$GPGGA,",1)[1]  #store data coming after “$GPGGA,” string
            NMEA_buff = (GPGGA_buffer.split(','))
            nmea_time = []
            nmea_latitude = []
            nmea_longitude = []
            nmea_time = NMEA_buff[0]                    #extract time from GPGGA string
            nmea_latitude = NMEA_buff[1]                #extract latitude from GPGGA string
            nmea_longitude = NMEA_buff[3]               #extract longitude from GPGGA string
            print("NMEA Time: ", nmea_time,'\n')
            lat = (float)(nmea_latitude)
            lat = convert_to_degrees(lat)
            longi = (float)(nmea_longitude)
            longi = convert_to_degrees(longi)
            print ("NMEA Latitude:", lat,"NMEA Longitude:", longi,'\n')      

#gps end
devnull = open('os.devnull', 'w')

caption_generator=CaptionGenerator(
    rnn_model_place='./data/caption_en_model40.model',
    cnn_model_place='./data/ResNet50.model',
    dictonary_place='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json',
    beamsize=3,
    depth_limit=50,
    gpu_id=-1,
    first_word= "<sos>",
    )


r= sr.Recognizer()
text = {}
text1 = {}
#location()
#location="Mumbai"+lat+longi

def listen1():
    with sr.Microphone(device_index = 1) as source:
               r.adjust_for_ambient_noise(source)
               print("Say Something");
               call(["espeak", "-s140  -ven+18 -z" , "Say Something"])
               audio = r.listen(source)
               print("got it");
    return audio

def voice(audio1):
       try: 
         text1 = r.recognize_google(audio1)
         print ("you said: " + text1)
         call(["espeak", "-s140  -ven+18 -z" , "you said " +text1])
         return text1; 
       except sr.UnknownValueError: 
          call(["espeak", "-s140  -ven+18 -z" , "Google Speech Recognition could not understand"])
          print("Google Speech Recognition could not understand") 
          return 0
       except sr.RequestError as e: 
          print("Could not request results from Google")
          return 0
        
def main(text):
       audio1 = listen1() 
       text = voice(audio1)
       site = 'https://news.google.com/news/rss'
       op = urlopen(site)
       rd = op.read()
       op.close()
       sp_page = soup(rd, 'xml')
       news_list = sp_page.find_all('item')  # finding news
       if 'news' in text:
           call(["espeak", "-s140  -ven+18 -z" , "Fetching Latest News"])
           for news in news_list:
               print(news.title.text)
               url = news.link.text
               article = Article(url)
               article.download()
               article.parse()
               mytext = news.title.text
               call(["espeak", "-s140  -ven+18 -z" , mytext])
               call(["espeak", "-s140  -ven+18 -z" , "    Sir, Do you want me to continue?"])
               audio1 = listen1()
               text = voice(audio1)
               if text=="yes please":
                   call(["espeak", "-s140  -ven+18 -z" , "    Ok Sir, Fetching another news"])
                   continue
               else:
                    call(["espeak", "-s140  -ven+18 -z" , "    Sorry Sir, Going to main menu"])
                    break
       elif 'location' in text:
           #location()
           #location=lat+longi
           location="Navi Mumbai"
           call(["espeak", "-s140  -ven+18 -z" , "You Current location is "+ location])
       elif "what is in front of me":
           cap = cv2.VideoCapture(0)
           frame=cap.read()[1]
           cv2.imwrite("image.jpg", frame)
           cap.release()
           call(["espeak", "-s140  -ven+18 -z" ,"this might take a while, please wait.."])
           captions = caption_generator.generate("image.jpg")
           word = " ".join(captions[0]["sentence"][1:-1])
           print(word)
           call(["espeak", "-s140  -ven+18 -z" ,word])
       text = {}

       
if __name__ == '__main__':
 while(1):
     audio1 = listen1() 
     text = voice(audio1)
     if text == 'Ok Google': 
         text = {}
         call(["espeak", "-s140  -ven+18 -z" ," Okay Sir, waiting for your command"])
         try:
             main(text)
         except TypeError:
             call(["espeak", "-s140  -ven+18 -z" , "Try Again"])
                 
     else:
         call(["espeak", "-s140 -ven+18 -z" , " Please repeat"])
