import time

from models.find_face import mark_faces_video

live_url = 'rtmp://play.live.frankxxj.top/oldcare/source1'
live_url = './video1.mp4'
start = time.time()
mark_faces_video(live_url)
end = time.time()
print('cost time:', end - start, ' seconds')
