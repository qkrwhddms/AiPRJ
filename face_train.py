# 설치한 OpenCV 패키지 불러오기
import cv2
import numpy as np

# 학습된 얼굴 정면검출기 사용하기
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# 카메라로부터 이미지 가져오기
vcp = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 얼굴 학습을 위한 LBPHF 알고리즘 선언
model = cv2.face.LBPHFaceRecognizer_create()

# 학습할 데이터구조 선언(학습데이터, 라벨링)
training_data, labels = [], []

count = 0

# 얼굴 이미지 학습 횟수가 100번 될 떄까지, while문을 계속 반복함
while True:
    ret, my_image = vcp.read()

    # 카메라로부터 프레임(이미지)를 잘 받았으면 실행함
    if ret is True:
        # 카메라의 프레임을 얼굴인식율을 높이기 위해 흑백으로 변경함
        gray = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)

        # 변환한 흑백사진으로부터 히스토그램 평활화
        gray = cv2.equalizeHist(gray)

        # 얼굴 검출 수행(정확도 높이는 방법의 아래 파라미터를 조절함)
        # 얼굴 검출은 히스토그램 평황화한 이미지 사용
        # scaleFactor : 1.5
        # minNeighbors : 인근 유사 픽셀 발견 비율이 5번 이상
        # flags : 0 => 더이상 사용하지 않는 인자 값
        # 분석할 이미지의 최소 크기 : 가로 100, 세로 100
        faces = face_cascade.detectMultiScale(gray, 1.5, 5, 0, (20, 20))

        # 인식된 얼굴의 수
        facesCnt = len(faces)

        # 얼굴 인식이 되었으면
        if facesCnt == 1:

            count += 1

            # 동영상 속 사람은 첫번째 인식된 1명
            x, y, w, h = faces[0]

            face_image = gray[y:y + h, x:x + w]

            # 학습을 위해 데이터구조에 학습할 얼굴이미지 및 라벨링 값 저장하기
            training_data.append(face_image)  # 얼굴이미지 학습
            labels.append(count)  # 학습되는 얼굴 이미지의 고유한 라벨링 값 정의

            print(training_data)  # 학습되는 얼굴이미지 데이터 출력
            print(labels)  # 학습되는 얼굴이미지의 라벨링 출력

            # 학습하기
            model.train(training_data, np.array(labels))

            # 학습한 결과를 학습모델 파일로 생성하기
            model.save("model/face-trainner.yml")

            # 얼굴 검출 사각형 그리기
            cv2.rectangle(my_image, faces[0], (255, 0, 0), 4)

        #else:
            #print("얼굴 미검출")

        # 사이즈 변경된 이미지로 출력하기
        cv2.imshow("train_my_face", my_image)

    if cv2.waitKey(1) == 13 or count == 100:
        break

vcp.release()

cv2.destroyAllWindows()