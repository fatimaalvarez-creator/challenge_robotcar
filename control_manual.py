from Rosmaster_Lib import Rosmaster
import sys, tty, termios, threading, time, cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

car = Rosmaster()
car.set_car_type(5)

latest_frame = None
frame_lock = threading.Lock()
running = True

def camera_loop():
    global latest_frame, running
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.05)
    cap.release()

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        while running:
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is None:
                time.sleep(0.1)
                continue
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            try:
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b'\r\n')
                time.sleep(0.1)
            except:
                break

def server_loop():
    server = HTTPServer(('0.0.0.0', 8080), StreamHandler)
    server.serve_forever()

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return key

t_cam = threading.Thread(target=camera_loop, daemon=True)
t_cam.start()
t_srv = threading.Thread(target=server_loop, daemon=True)
t_srv.start()

print('Stream en http://10.43.53.156:8080')
print('W=Adelante S=Atras A=Izquierda D=Derecha Espacio=Stop Q=Salir')

try:
    while True:
        key = get_key().lower()
        if key == 'w':
            print('Adelante')
            car.set_car_motion(0, 0, 0)
            time.sleep(0.1)
            car.set_motor(800, 800, 800, 800)
        elif key == 's':
            print('Atras')
            car.set_car_motion(0, 0, 0)
            time.sleep(0.1)
            car.set_motor(-800, -800, -800, -800)
        elif key == 'a':
            print('Izquierda')
            car.set_car_motion(0.3, 0, 0.5)
        elif key == 'd':
            print('Derecha')
            car.set_car_motion(0.3, 0, -0.5)
        elif key == ' ':
            print('Detenido')
            car.set_motor(0, 0, 0, 0)
            car.set_car_motion(0, 0, 0)
        elif key == 'q':
            running = False
            car.set_motor(0, 0, 0, 0)
            car.set_car_motion(0, 0, 0)
            break
except KeyboardInterrupt:
    running = False
    car.set_motor(0, 0, 0, 0)
    car.set_car_motion(0, 0, 0)
