"""Performs face alignment and stores face thumbnails in the output directory."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import random
import align_dlib  # @UnresolvedImport
import dlib
import facenet
import cv2

import mxnet as mx
import pdb
from lightened_moon import lightened_moon_feature
import numpy as np
import face_recognition
import json

#two tasks to do

#COMPLETE 1) given a video, output JSON of all the features for each face found
    #probably also need to assign each face an ID, and keep recognition throughout the frames

#IN PROGRESS 2) given a video and a JSON containing that information, produce a frame-by-frame visualization of the faces/attributes detected

def main(args):

    #MOON feature extractor; not sure how to make this a modular component
    symbol = lightened_moon_feature(num_classes=40, use_fuse=True)

    #the detector passed in from the command line; requires files in facenet/data/
    detector = align_dlib.AlignDlib(os.path.expanduser(args.dlib_face_predictor))

    #fixed landmark indices; could make modular, but later
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE

    video = cv2.VideoCapture(args.input_video)

    #TODO: this is the main problem: cropped_face refers to the already-aligned faces
    # in this code, this face is used for a few things
    #  1) to feed into the attribute classifier
    #cropped_face = cv2.imread(args.cropped_img)
    #scale = float(args.face_size) / args.image_size
    #for now, solution is to ignore altogether

    devs = mx.cpu()

    #begin to iterate over the frames and process them
    ret, frame = video.read()

    #a list of dictionaries containing face_output for each frame
    total_output = []

    #maps encoding matrix to id number
    known_faces_dict = dict()
    known_faces_encoding = []
    id_count = 0

    while ret is True:
        face_boxes = detector.getAllFaceBoundingBoxes(frame)

        id_attr, known_faces_dict, known_faces_encoding, id_count = processFrame(args, frame, known_faces_dict, known_faces_encoding, id_count, symbol, detector, landmarkIndices, devs, face_boxes)

        total_output.append(id_attr)

        ret, frame = video.read()

    #==========CONVERT TO JSON FILE===============
    print('done processing; converting to json')
    #print(total_output)
    #ith element represents the ith frame
    frame_num = 0
    json_output = '{\r\n"video":\r\n{\r\n"frames":\r\n[\r\n'
    for frame_info in total_output:
        #begin the num-faces entry
        json_output += '{\r\n"num": '+str(frame_num)+',\r\n'
        if len(frame_info.keys()) == 0:
            #if this still isnt valid, try doing "faces": 0 and closing the field
            # remove last occurrence of comma
            k = json_output.rfind(',')
            json_output = json_output[:k] + json_output[k+1:]
            json_output += '},\r\n' #close the num-faces entry; no faces field
            frame_num += 1
            continue
        json_output += '"faces":\r\n[\r\n'
        # process the face information in frame_info in a loop
        for face in frame_info.keys():
            #get actual content, which is a list
            #content shouldnt ever be empty, because there exists a key
            #TODO may be a bug bc of this assumption
            content = frame_info[face]
            pid = content[0]

            #check if content is length > 1
            #there may be an individual with 0 yes-attributes
            if len(content) == 3:
                #attributes will contain the topleft,bottomright coordinates,
                #followed by the attributes themselves
                attributes = content[1:len(content)-1]
                attributes.extend(['Negatives'])
                d = {pid:attributes} #looks like 0:[]
                json_output += json.dumps(d)+',\r\n'
                continue

            #attributes will contain the topleft, bottomright coordinates
            #followed by the attributes themselves
            attributes = content[1:len(content)-1]
            d = {pid:attributes}
            #now we have the proper split
            json_output += json.dumps(d)+',\r\n'
        #outside of loop
        # remove last occurrence of comma
        k = json_output.rfind(',')
        json_output = json_output[:k] + json_output[k+1:]
        json_output += ']\r\n' #close the faces array
        json_output += '},\r\n' #close the num-faces entry
        frame_num += 1
    # remove last occurrence of comma
    k = json_output.rfind(',')
    json_output = json_output[:k] + json_output[k+1:]
    json_output += '\r\n]\r\n}\r\n}'

    d = json.loads(json_output)
    json_output = json.dumps(d, indent=4, separators=(',', ': '))

    #write out to file
    print('done converting to json; writing to file')
    f = open('output.json', 'wb')
    f.write(json_output)
    f.close()
    print('done!')

def processFrame(args, frame, known_faces_dict, known_faces_encoding, id_count, symbol, detector, landmarkIndices, devs, face_boxes):

    if len(face_boxes) == 0:
        print('cannot find faces')

    #list where first entry is id, 2nd and 3rd entries are bounding box coordiantes, 
    #and rest are attributes for that id
    #key is the face number, but that is not too relevant
    id_attr = dict()

    face_num = 0
    for box in face_boxes:

        #=========CROP FACE==========
        pad = [0.25, 0.25, 0.25, 0.25]
        left = int(max(0, box.left() - box.width()*float(pad[0])))
        top = int(max(0, box.top() - box.height()*float(pad[1])))
        right = int(min(frame.shape[1], box.right() + box.width()*float(pad[2])))
        bottom = int(min(frame.shape[0], box.bottom()+box.height()*float(pad[3])))

        cropped_face = frame[top:bottom, left:right]

        #align if specified
        if args.align == 1:
            scale = float(args.face_size) / args.image_size
            cropped_face = detector.align(args.image_size, cropped_face, landmarkIndices=landmarkIndices, skipMulti=False, scale=scale)

        if cropped_face is None:
            print('failed to align! will not save this face\'s attributes or ID')
            continue

        #array of two points: top left, bottom right
        bounding_box_coordinates = [(top,left), (bottom,right)]

        #=========DRAW BOUNDING BOX=========
        '''
        pad = [0.25, 0.25, 0.25, 0.25]
        left = int(max(0, box.left() - box.width()*float(pad[0])))
        top = int(max(0, box.top() - box.height()*float(pad[1])))
        right = int(min(img.shape[1], box.right() + box.width()*float(pad[2])))
        bottom = int(min(img.shape[0], box.bottom()+box.height()*float(pad[3])))

        cv2.rectangle(img, (left, top), (right, bottom), (0,255,0),3) 
        '''
        #========EXTRACT ATTRIBUTES=======
        # crop face area and resize as feature input
        gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))/255.0
        temp_img = np.expand_dims(np.expand_dims(gray, axis=0), axis=0)
        # get pred
        _, arg_params, aux_params = mx.model.load_checkpoint(args.model_load_prefix, args.model_load_epoch)
        arg_params['data'] = mx.nd.array(temp_img, devs)
        exector = symbol.bind(devs, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exector.forward(is_train=False)
        exector.outputs[0].wait_to_read()
        output = exector.outputs[0].asnumpy()
        text = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald", "Bangs","Big_Lips","Big_Nose",
                "Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee",
                "Gray_Hair", "Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",
                "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
                "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
        #pred = np.ones(40)

        attr_list = []
        i = 0
        for num in output[0]:
            if num > 0:
                attr_list.append(text[i])
            i+=1
        #=========WRITE ATTRIBUTES W/ YES NEXT TO BOUNDING BOX============
        '''
        yes_attributes = []
        index = 0

        for num in output[0]:
            if num > 0:
                yes_attributes.append(text[index])
            index+=1

        pad = 20
        for attr in yes_attributes:
            cv2.putText(img, attr, (right, top), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255), 2)
            top = top + pad
        '''
        #========WRITE ATTRIBUTES AND ID TO DICT===========
        face_enc = face_recognition.face_encodings(cropped_face)

        if face_enc is None:
            #print('didnt catch this face')
            continue

        if len(face_enc) == 0:
            #print('didnt catch this face with face_recognition')
            continue

        #print('got the encoding; flattening and using first element as hash index')
        face_enc = face_enc[0]
                #print(face_enc.shape)
        face_enc_hashable = face_enc.flatten()[0]
        #print(face_enc_hashable[0])

        if len(known_faces_encoding) == 0:
            #print('first known face!')
            known_faces_dict[face_enc_hashable] = id_count
            known_faces_encoding = [face_enc]
            id_l = [id_count]
            id_l.extend(bounding_box_coordinates)
            id_l.extend(attr_list)

            #save into the dictionary; id_l format is [id, (topleft), (bottomright), attr...]
            id_attr[face_num] = id_l
            id_count += 1
            #print('added to dict of known faces')
            continue

        #print('comparing with list of known faces')
        compare_results = face_recognition.compare_faces(known_faces_encoding, face_enc)

        index = 0
        identifier = None
        #print(compare_results)
        #print('done comparisons on known faces; looking for a match')
        while index < len(compare_results):
            result = compare_results[index]
            if result:
                identifier = known_faces_encoding[index]
                break
            index += 1

        if identifier is None:
            #print('no match; adding this face to the list with new id')
            #add to dict and known encodings
            known_faces_encoding = np.append(known_faces_encoding, [face_enc], axis=0)
            known_faces_dict[face_enc_hashable] = id_count
            id_l = [id_count]
            id_l.extend(bounding_box_coordinates)
            id_l.extend(attr_list)
            #print('This should have at least one element: ' + str(id_l))
            id_attr[face_num] = id_l
            id_count += 1
            #print(known_faces_dict)
            #cv2.imshow('cropped face', cropped_face)
            #cv2.waitKey(0)
        else:
            #print('we have a match! getting id from match')
            #get the encoding that was True via the index and add to json dict
            similar_encoding = known_faces_encoding[index]
            similar_encoding_hash = similar_encoding.flatten()[0]
            projected_id = known_faces_dict[similar_encoding_hash]
            id_l = [projected_id]
            id_l.extend(bounding_box_coordinates)
            id_l.extend(attr_list)
            #print('This should have at least one element: ' + str(id_l))
            id_attr[face_num] = id_l

        face_num += 1

    #after running on sample.mp4, should ideally have only 2 entries
    #print(id_attr)

    return id_attr, known_faces_dict, known_faces_encoding, id_count

#currently, the following supporting files from mxnet-face and facenet are needed:
# ../data/shape_predictor_68_face_landmarks.dat from facenet (downloadable from repo)
# lightened_moon.py from mxnet-face (should be in attribute)
# lightened_moon folder containing lightened_moon_fuse from model folder in mxnet-face

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_video', type=str, help='Target image.')
    #parser.add_argument('cropped_img', type=str, help='Cropped face')
    parser.add_argument('--dlib_face_predictor', type=str,
        help='File containing the dlib face predictor.', default='../data/shape_predictor_68_face_landmarks.dat')

    # custom options added by me
    parser.add_argument('--align', type=int,
        help='Indicate whether faces should be aligned for feature extraction, default is 0. 0=No, 1=Yes. If yes, specify --image_size and --face_size if needed.', default=0)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=110)
    parser.add_argument('--face_size', type=int,
        help='Size of the face thumbnail (height, width) in pixels.', default=96)
    # parser.add_argument('--use_center_crop', 
    #     help='Use the center crop of the original image after scaling the image using prealigned_scale.', action='store_true')
    # parser.add_argument('--prealigned_dir', type=str,
    #     help='Replace image with a pre-aligned version when face detection fails.', default='')
    # parser.add_argument('--prealigned_scale', type=float,
    #     help='The amount of scaling to apply to prealigned images before taking the center crop.', default=0.87)

    # parser.add_argument('--size', type=int, default=128,
    #                     help='the image size of lfw aligned image, only support squre size')
    # #parser.add_argument('--opencv', type=str, default='~/Desktop/mxnet-face/model/opencv/cascade.xml',
    # #                    help='the opencv model path')
    # parser.add_argument('--pad', type=float, nargs='+',
    #                              help="pad (left,top,right,bottom) for face detection region")
    parser.add_argument('--model-load-prefix', type=str, default='lightened_moon/lightened_moon_fuse',
                        help='the prefix of the model to load')
    parser.add_argument('--model-load-epoch', type=int, default=82,
                        help='load the model on an epoch using the model-load-prefix')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
