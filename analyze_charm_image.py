#!/usr/bin/python3
from google.cloud import vision
from skimage.metrics import structural_similarity
import cv2
import imutils
import io
import json
import numpy
import os.path
import sys

class Charm:
    # 1500, 230, (330, 330)
    x=1500
    y=230
    w=330
    h=330

    x1=210
    y1=85
    w1=40
    h1=30

    first_zero_sample_image = cv2.imread('firstsocket/zero.jpg')
    first_one_sample_image = cv2.imread('firstsocket/one.jpg')
    first_two_sample_image = cv2.imread('firstsocket/two.jpg')
    first_three_sample_image = cv2.imread('firstsocket/three.jpg')
    second_zero_sample_image = cv2.imread('secondsocket/zero.jpg')
    second_one_sample_image = cv2.imread('secondsocket/one.jpg')
    second_two_sample_image = cv2.imread('secondsocket/two.jpg')
    third_zero_sample_image = cv2.imread('thirdsocket/zero.jpg')
    third_one_sample_image = cv2.imread('thirdsocket/one.jpg')
    

    skill_map = json.loads(''.join(open('data.json').readlines()))

    def __init__(self, filepath):
        self.filepath = filepath
        self.cv2image = cv2.imread(filepath)
        self.socket_results='' # something like 0-0-0, 3-2-0, 2-1-1, etc
        self.skill0_name = ''
        self.skill0_level = 0
        self.skill1_name = ''
        self.skill1_level = 0

    def sanitize_string(self, string):
        map = {'ν':'v','、':'','伴':'佯','カ':'力'}
        result = string.lower().strip()
        for k,v in map.items():
            result = result.replace(k, v)
        return result

    def analyze(self):
        ocr_result_filepath = self.filepath.replace('.jpg', '.txt')
        if  os.path.isfile(ocr_result_filepath):
            ocr_result_file = open(ocr_result_filepath)
            ocr_result = ''.join(ocr_result_file.readlines())
            ocr_result_file.close()
        else:
            ocr_result = self.ocr()
            ocr_result_file = open(ocr_result_filepath, 'w')
            ocr_result_file.write(ocr_result)
            ocr_result_file.close()
        parse_result = self.parse_string(ocr_result)
        if not parse_result:
            return False
        first_socket = Charm.analyze_first_socket(self.cv2image)
        second_socket = Charm.analyze_second_socket(self.cv2image)
        third_socket = Charm.analyze_third_socket(self.cv2image)
        self.socket_results='%d-%d-%d' % (first_socket, second_socket, third_socket)
        return True

    def __str__(self):
        skill0 = Charm.skill_map[self.skill0_name]
        if self.skill1_level > 0:
            skill1 = Charm.skill_map[self.skill1_name]
            return '[(%s-LV%d)(%s-LV%d)(%s)]' % (skill0, self.skill0_level, skill1, self.skill1_level, self.socket_results)
        else:
            return '[(%s-LV%d)(%s)]' % (skill0, self.skill0_level, self.socket_results)

    def toArray(self):
        skill0 = Charm.skill_map[self.skill0_name]
        if self.skill1_level > 0:
            skill1 = Charm.skill_map[self.skill1_name]
            return [skill0, self.skill0_level, skill1, self.skill1_level, self.socket_results]
        else:
            return [skill0, self.skill0_level, '', self.skill1_level, self.socket_results]


    def parse_string(self, string):
        parts = self.sanitize_string(string).split('lv')
        if len(parts) == 3:
            # two skills
            try:
                self.skill0_name = parts[0]
                self.skill0_level = int(parts[1][0])
                self.skill1_name = parts[1][1:]
                self.skill1_level = int(parts[2][0])
            except:
                print('Cannot parse this:[%s] for file %s' % (string ,self.filepath))    
        elif len(parts) == 2:
            # one skills
            try:
                self.skill0_name = parts[0]
                self.skill0_level = int(parts[1])
            except:
                print('Cannot parse this:[%s] for file %s' % (string ,self.filepath))
        else:
            print('Cannot parse this:[%s] for file %s' % (string ,self.filepath))
            return False
        return True
        
    def ocr(self):
        client = vision.ImageAnnotatorClient()
        # [START vision_python_migration_text_detection]
        with io.open(self.filepath, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        result = []
        for text in texts:
            should_skip_this_one = False
            for vertex in text.bounding_poly.vertices:
                if vertex.y < 170:
                    should_skip_this_one = True
            if should_skip_this_one:
                continue
            result.append(text.description)
        return ''.join(result).replace('-', '･')

    @staticmethod
    def similarity(image_a, image_b):
        grayA = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        return score
    
    @staticmethod
    def socket_type(score0=-1, score1=-1, score2=-1, score3=-1):
        if score0 > score1 and score0 > score2 and score0 > score3:
            return 0
        if score1 > score0 and score1 > score2 and score1 > score3:
            return 1
        if score2 > score0 and score2 > score1 and score2 > score3:
            return 2
        if score3 > score0 and score3 > score1 and score3 > score2:
            return 3

    @staticmethod
    def analyze_first_socket(image):
        first_socket_image = image[Charm.y1:Charm.y1 + Charm.h1, Charm.x1:Charm.x1 + Charm.w1]
        winner = -1
        score0 = Charm.similarity(first_socket_image, Charm.first_zero_sample_image)
        score1 = Charm.similarity(first_socket_image, Charm.first_one_sample_image)
        score2 = Charm.similarity(first_socket_image, Charm.first_two_sample_image)
        score3 = Charm.similarity(first_socket_image, Charm.first_three_sample_image)
        return Charm.socket_type(score0,score1,score2,score3)

    @staticmethod
    def analyze_second_socket(image):
        second_socket_image = image[Charm.y1:Charm.y1 + Charm.h1, Charm.x1 + Charm.w1:Charm.x1 + Charm.w1 * 2]
        winner = -1
        score0 = Charm.similarity(second_socket_image, Charm.second_zero_sample_image)
        score1 = Charm.similarity(second_socket_image, Charm.second_one_sample_image)
        score2 = Charm.similarity(second_socket_image, Charm.second_two_sample_image)
        return Charm.socket_type(score0,score1,score2)

    @staticmethod
    def analyze_third_socket(image):
        third_socket_image = image[Charm.y1:Charm.y1 + Charm.h1, Charm.x1 + Charm.w1 * 2:Charm.x1 + Charm.w1 * 3]
        winner = -1
        score0 = Charm.similarity(third_socket_image, Charm.third_zero_sample_image)
        score1 = Charm.similarity(third_socket_image, Charm.third_one_sample_image)
        return Charm.socket_type(score0,score1)


if __name__ == "__main__":
    #analyze_image(cv2.imread(sys.argv[1]))
    results = []
    for filename in os.listdir('.'):
        if filename.endswith('.jpg'):
            a = Charm(filename)
            if a.analyze():
                print(a)
                results.append(a.toArray())
            else:
                print('Parse %s failed' % filename)
    with open('result.json', 'w') as fp:
        json.dump(results, fp)
