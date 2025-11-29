import cv2
import cv2
import pickle
import numpy as np
import os
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data=[]

i=0

name=input("Enter Your Name: ")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==100:
        break
video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100, -1)


if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)




        private static String nextWordOrSeparator(String text, int position) {
        assert text != null : "Violation of: text is not null";
        assert 0 <= position : "Violation of: 0 <= position";
        assert position < text.length() : "Violation of: position < |text|";

    
        Set<Character> newSet = new Set1L<Character>();

        for (int i = 0; i < SEPARATORS.length(); i++) {
            char c = SEPARATORS.charAt(i);
            if (!newSet.contains(c)) {
                newSet.add(c);
            }
        }
        boolean isSep = newSet.contains(text.charAt(position));
        while (endIndex < text.length()] && newSet.contains(text.charAt(endIndex)) == isSep) {
            endIndex++;
        }

        return text.substring(position, endIndex);


 Set<Character> newSet = new Set1L<Character>();
        for (int i = 0; i < SEPARATORS.length(); i++) {
            char c = SEPARATORS.charAt(i);
            if (!newSet.contains(c)) {
                newSet.add(c);
            }
        }
        Queue<String> queueOfTokens = new Queue1L<String>();
        while (!in.atEOS()) {
            int position = 0;
            String line = in.nextLine();
            while (position < line.length()) {
                String token = nextWordOrSeparator(line, position);
                if (!newSet.contains(line.charAt(position))) {
                    queueOfTokens.enqueue(token);
                }
                position += token.length();
            }
        }
        queueOfTokens.enqueue(END_OF_INPUT);

        return queueOfTokens;