# 설치한 OpenCV 패키지 불러오기
import cv2

# 분석하기 위한 이미지 불러오기
image = cv2.imread("image/newjeans2.jpg", cv2.IMREAD_COLOR)

# 흑백사진으로 변경
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 변환한 흑백사진으로부터 히스토그램 평활화
gray = cv2.equalizeHist(gray)

if image is None: raise Exception("이미지 읽기 실패")

# 학습된 얼굴 정면검출기 사용하기
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# 얼굴 검출 수행(정확도 높이는 방법의 아래 파라미터를 조절함)
# 얼굴 검출은 히스토그램 평황화한 이미지 사용
# scaleFactor : 1.5
# minNeighbors : 인근 유사 픽셀 발견 비율이 5번 이상
# flags : 0 => 더이상 사용하지 않는 인자값
# 분석할 이미지의 최소 크기 : 가로 100, 세로 100
faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (100, 100))

# 인식된 얼굴의 수
facesCnt = len(faces)

# 인식된 얼굴의 수 출력
print(len(faces))

# 얼굴이 검출되었다면,
if facesCnt > 0:

    for face in faces:
        #얼굴 위치 값을 가져오기
        x, y, w, h = face

        #얼굴 검출 사각형 그리기
        cv2.rectangle(image, face, (255, 0, 0), 4)

else: print("얼굴 미검출")

# 사이즈 변경된 이미지로 출력하기
cv2.imshow("MyFace", image)

# 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
cv2.waitKey(0)