#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import xmltodict
import json
import codecs

def pythonXmlToJson(xmlStr):
  
    #1.Xml to Jason
#    xmlStr = """
#<student>
#    <stid>10213</stid>
#    <info>
#        <name>name</name>
#        <mail>xxx@xxx.com</mail>
#        <sex>male</sex>
#    </info>
#    <course>
#        <name>math</name>
#        <score>90</score>
#    </course>
#    <course>
#        <name>english</name>
#        <score>88</score>
#    </course>
#</student>
#"""
#    xmlStr = """
#<annotation>
#	<folder>VOC2007</folder>
#	<filename>000001.jpg</filename>
#	<source>
#		<database>The VOC2007 Database</database>
#		<annotation>PASCAL VOC2007</annotation>
#		<image>flickr</image>
#		<flickrid>341012865</flickrid>
#	</source>
#	<owner>
#		<flickrid>Fried Camels</flickrid>
#		<name>Jinky the Fruit Bat</name>
#	</owner>
#	<size>
#		<width>353</width>
#		<height>500</height>
#		<depth>3</depth>
#	</size>
#	<segmented>0</segmented>
#	<object>
#		<name>dog</name>
#		<pose>Left</pose>
#		<truncated>1</truncated>
#		<difficult>0</difficult>
#		<bndbox>
#			<xmin>48</xmin>
#			<ymin>240</ymin>
#			<xmax>195</xmax>
#			<ymax>371</ymax>
#		</bndbox>
#	</object>
#	<object>
#		<name>person</name>
#		<pose>Left</pose>
#		<truncated>1</truncated>
#		<difficult>0</difficult>
#		<bndbox>
#			<xmin>8</xmin>
#			<ymin>12</ymin>
#			<xmax>352</xmax>
#			<ymax>498</ymax>
#		</bndbox>
#	</object>
#</annotation>
#"""
 
    convertedDict = xmltodict.parse(xmlStr);
    jsonStr = json.dumps(convertedDict, indent=1);
#    print "jsonStr=",jsonStr;

    return jsonStr
    

def pythonJsonToXml(jsonStr):     
    #2.Json to Xml
#    dictVal = {
#        'page': {
#            'title': 'King Crimson',
#            'ns': 0,
#            'revision': {
#                'id': 547909091,
#            }
#        }
#    };

#    dictVal = {
#    "annotation": {
#  "folder": "VOC2007", 
#  "filename": "000001.jpg", 
#  "source": {
#   "database": "The VOC2007 Database", 
#   "annotation": "PASCAL VOC2007", 
#   "image": "flickr", 
#   "flickrid": "341012865"
#  }, 
#  "owner": {
#   "flickrid": "Fried Camels", 
#   "name": "Jinky the Fruit Bat"
#  }, 
#  "size": {
#   "width": "353", 
#   "height": "500", 
#   "depth": "3"
#  }, 
#  "segmented": "0", 
#  "object": [{
#    "name": "dog", 
#    "pose": "Left", 
#    "truncated": "1", 
#    "difficult": "0", 
#    "bndbox": {
#     "xmin": "48", 
#     "ymin": "240", 
#     "xmax": "195", 
#     "ymax": "371"
#    }
#   }, 
#   {
#    "name": "person", 
#    "pose": "Left", 
#    "truncated": "1", 
#    "difficult": "0", 
#    "bndbox": {
#     "xmin": "8", 
#     "ymin": "12", 
#     "xmax": "352", 
#     "ymax": "498"
#    }
#   }]
# }
#}
    

#    convertedXml = xmltodict.unparse(dictVal, pretty=True)
#    print "convertedXml=",convertedXml   
    convertedXml = xmltodict.unparse(jsonStr, pretty=True)
    return convertedXml
 
###############################################################################
#if __name__=="__main__":
#    pythonXmlToJson();
#    pythonJsonToXml();