import cv2

from models.find_face import circle_faces

live_url = 'rtmp://play.live.frankxxj.top/oldcare/source1'
source = cv2.VideoCapture(live_url)
counter = 0
while source.isOpened():
    ret, frame = source.read()
    if not ret:
        print('Source has stopped!')
        break
    if counter == 0:
        frame = circle_faces(frame)
    cv2.imshow('frame', frame)
    counter = (counter + 1) % 5
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
source.release()