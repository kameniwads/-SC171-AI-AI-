import cv2

# 打开USB摄像头，通常设备号是0，1，2等，取决于你的设备
cap = cv2.VideoCapture(1)  # 0是默认摄像头设备号，如果有多个摄像头可以试试1, 2等

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 从摄像头捕获帧
    ret, frame = cap.read()

    if not ret:
        print("无法接收到视频帧")
        break

    # 显示视频帧
    cv2.imshow('USB Camera', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭所有窗口
cap.release()
cv2.destroyAllWindows()