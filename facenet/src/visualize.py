import json
import cv2
import sys
import argparse

def main(args):
    #TODO read in JSON as dict; then for each key in "frame", aka each "num",
    #draw bounding box and list ID, features
    #then display

    if args.input_json is None or args.input_video is None:
        print('Need BOTH json and video files')
        return

    f = open(args.input_json, 'rb')
    data = json.load(f)
    f.close()

    video = cv2.VideoCapture(args.input_video)

    #we only care about the frames themselves
    #data is now a list of dictionaries, each containing info about the frame
    data = data["video"]["frames"]
    #print data
    for frame_dict in data:

        #read in the corresponding video frame
        ret, frame = video.read()

        #if there are faces, process and display
        if "faces" in frame_dict:
            #print frame_dict["faces"]
            #process each face in a nested for loop
            #this list contains a dictionary of all faces
            face_dicts = frame_dict["faces"]

            for face_dict in face_dicts:
                #just as a test, print the id only
                #we know there is only one key bc face_dicts
                #is a list of dicts with one key
                face_id = face_dict.keys()[0]
                info = face_dict[face_id]

                #splice the first two entries into box coordinates
                topleft = info[0]
                bottomright = info[1]
                
                #convert topleft/bottomright from lists to tuples
                top = topleft[0]
                left = topleft[1]
                bottom = bottomright[0]
                right = bottomright[1]
                temp = info[2:len(info)-1]
                attributes = ['ID: ' + str(face_id)]
                attributes.extend(temp)

                #now we can draw the bounding boxes and attributes onto the frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0),3)

                pad = 20
                for attr in attributes:
                    cv2.putText(frame, attr, (right, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    top = top + pad

        cv2.imshow('frame', frame)
        cv2.waitKey(0)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_json', type=str, help='JSON file output from detect.py')
    parser.add_argument('input_video', type=str, help='video file processed into input_json')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
