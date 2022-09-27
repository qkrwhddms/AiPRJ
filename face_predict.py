# 설치한 OpenCV 패키지 불러오기
import cv2

# 학습된 얼굴 정면검출기 사용하기
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# 카메라로부터 이미지 가져오기
vcp = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 얼굴 학습을 위한 알고리즘 선언
model = cv2.face.LBPHFaceRecognizer_create() #LBPH를 사용할 새 변수 생성

# 학습된 모델 가져오기
model.read("model/face-trainner.yml")   #저장된 값 가져오기

# 카메라로부터 이미지 가져오기
while True:
    ret, my_image = vcp.read()

    # 동영상으로부터 프레임(이미지)를 잘 받았으면 실행함
    if ret is True:

        # 동영상의 프레임을 얼굴인식율을 높이기 위해 흑백으로 변경함
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

            # 동영상 속 사람은 첫번째 인식된 1명
            x, y, w, h = faces[0]

            face_image = gray[y:y + h, x:x + w]

            # 유사도 분석
            id_, res = model.predict(face_image)

            # 예측결과 문자열
            result = "result : " + str(res) + "%"

            # 예측결과 문자열 사진에 추가하기
            cv2.putText(my_image, result, (x, y - 15), 0, 1, (255, 0, 0), 2)

            # 얼굴 검출 사각형 그리기
            cv2.rectangle(my_image, faces[0], (255, 0, 0), 4)

        # 사이즈 변경된 이미지로 출력하기
        cv2.imshow("predict_my_face", my_image)

    # 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
    if cv2.waitKey(1) > 0:
        break

vcp.relsease()

cv2.destoryAllWindows()