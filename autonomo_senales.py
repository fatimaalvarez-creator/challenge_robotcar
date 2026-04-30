from Rosmaster_Lib import Rosmaster
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import threading, time, cv2, os
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- Configuracion navegacion ---
DISTANCIA_MIN = 0.5
DISTANCIA_FRENO = 0.8
DISTANCIA_LATERAL = 0.4
VELOCIDAD = 800
VELOCIDAD_PEATONAL = 400

# --- Configuracion Pivot ---
STEERING_SERVO_ID = 1
SERVO_CENTER_DEG = 90
SERVO_SETTLE_SEC = 0.15
USE_PULSED_PIVOT = True
USE_SERVO_COUNTER_STEER = True
SERVO_COUNTER_STEER_DEG = 5
PULSE_ON_SEC = 0.10
PULSE_OFF_SEC = 0.05
PIVOT_SCALE = 600
PIVOT_MIN = 250
PIVOT_MAX = 900
PIVOT_LEFT_SIGN = +1

# --- Configuracion señales ---
REFERENCES_DIR = "/home/jetson/sign_images"
APPLY_CLAHE = True
LOWE_RATIO = 0.75
MIN_GOOD_MATCHES = 12
MIN_INLIERS = 8
RANSAC_REPROJ_THRESH = 5.0
MIN_BBOX_AREA_FRAC = 0.002
MAX_BBOX_AREA_FRAC = 0.40
MIN_ASPECT_RATIO = 0.30
MAX_ASPECT_RATIO = 3.30
PROCESS_EVERY_N = 5

SIGNS = {
    "restricted_area": ("sign0.jpeg", "AREA RESTRINGIDA",  ( 50,  50, 220)),
    "pedestrian_zone": ("sign1.jpeg", "ZONA PEATONAL",     (230, 130,  20)),
    "robots_only":     ("sign2.jpeg", "SOLO ROBOTS",       ( 30, 210, 230)),
    "stop":            ("sign3.jpeg", "STOP",              ( 30,  30, 220)),
    "loading_zone":    ("sign4.jpeg", "ZONA DE CARGA",     ( 30, 150, 255)),
    "parking_zone":    ("sign5.jpeg", "ESTACIONAMIENTO",   ( 40, 200,  40)),
}

# --- Estado global ---
car = Rosmaster()
car.set_car_type(5)
time.sleep(0.2)
car.set_car_motion(0.0, 0.0, 0.0)
time.sleep(0.05)
car.set_pwm_servo(STEERING_SERVO_ID, SERVO_CENTER_DEG)
time.sleep(SERVO_SETTLE_SEC)

latest_frame = None
frame_lock = threading.Lock()
running = True
distancia_frente = 999
distancia_izquierda = 999
distancia_derecha = 999

velocidad_actual = VELOCIDAD
zona_restringida = False
mision_carga_completada = False
esperando_confirmacion = False
senal_detectada = None

# --- Publisher ROS2 para zona de carga ---
pub_carga = None

# =============================================================
# FUNCIONES DE VISION
# =============================================================
def preprocess_gray(bgr, clahe):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if clahe:
        gray = clahe.apply(gray)
    return gray

def load_references(detector, clahe):
    refs = {}
    for label, (filename, text, color) in SIGNS.items():
        path = os.path.join(REFERENCES_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"[AVISO] No se encontro: {path}")
            continue
        gray = preprocess_gray(img, clahe)
        kp, des = detector.detectAndCompute(gray, None)
        if des is None or len(kp) < 10:
            print(f"[AVISO] Pocos keypoints en {label}")
            continue
        refs[label] = {"gray": gray, "kp": kp, "des": des, "text": text, "color": color}
        print(f"  OK {label} -> {len(kp)} keypoints")
    return refs

def detect_sign(ref, scene_kp, scene_des, matcher, frame_shape):
    if scene_des is None or len(scene_des) < 2:
        return None, 0
    try:
        knn = matcher.knnMatch(ref["des"], scene_des, k=2)
    except:
        return None, 0
    good = [m for pair in knn if len(pair)==2 for m,n in [pair] if m.distance < LOWE_RATIO*n.distance]
    if len(good) < MIN_GOOD_MATCHES:
        return None, len(good)
    src_pts = np.float32([ref["kp"][m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([scene_kp[m.trainIdx].pt  for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if H is None or mask is None or int(mask.sum()) < MIN_INLIERS:
        return None, 0
    h_r, w_r = ref["gray"].shape[:2]
    corners = cv2.perspectiveTransform(
        np.float32([[0,0],[0,h_r-1],[w_r-1,h_r-1],[w_r-1,0]]).reshape(-1,1,2), H)
    pts = corners.reshape(4,2).astype(np.int32)
    area = cv2.contourArea(pts)
    h, w = frame_shape[:2]
    if area < MIN_BBOX_AREA_FRAC*h*w or area > MAX_BBOX_AREA_FRAC*h*w:
        return None, 0
    x,y,bw,bh = cv2.boundingRect(pts)
    if bh==0 or not (MIN_ASPECT_RATIO <= bw/bh <= MAX_ASPECT_RATIO):
        return None, 0
    return corners, int(mask.sum())

def detect_red_circle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,80,60]), np.array([10,255,255]))
    m2 = cv2.inRange(hsv, np.array([170,80,60]), np.array([179,255,255]))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.005*h*w or area > 0.40*h*w:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circ = 4*np.pi*area/(peri*peri)
        if circ < 0.55:
            continue
        x,y,bw,bh = cv2.boundingRect(cnt)
        roi = gray[y:y+bh, x:x+bw]
        edges = cv2.Canny(roi, 50, 150)
        min_len = int(0.4*min(bh,bw))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=min_len, maxLineGap=10)
        has_diagonal = False
        if lines is not None:
            for x1,y1,x2,y2 in lines.reshape(-1,4):
                dx,dy = x2-x1, y2-y1
                if dx == 0:
                    continue
                ang = abs(np.degrees(np.arctan2(dy,dx)))
                if 20 <= ang <= 70 or 110 <= ang <= 160:
                    has_diagonal = True
                    break
        if not has_diagonal:
            continue
        corners = np.float32([[x,y],[x,y+bh],[x+bw,y+bh],[x+bw,y]]).reshape(-1,1,2)
        return corners
    return None

def draw_detection(frame, corners, text, color, n_in):
    pts = corners.reshape(-1,2).astype(np.int32)
    cv2.polylines(frame, [pts], True, color, 3)
    x,y,_,_ = cv2.boundingRect(pts)
    label = f"{text} ({n_in})"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, max(0,y-th-10)), (x+tw+10, y), color, -1)
    cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# =============================================================
# COMPORTAMIENTOS POR SEÑAL
# =============================================================
def ejecutar_comportamiento(label):
    global velocidad_actual, zona_restringida, mision_carga_completada
    global esperando_confirmacion, senal_detectada

    if label == "restricted_area":
        print("[SENAL] AREA RESTRINGIDA — evitando y redirigiendo")
        zona_restringida = True
        car.set_motor(0, 0, 0, 0)
        time.sleep(0.3)
        hacer_pivot(-1, duracion=1.5)
        zona_restringida = False

    elif label == "pedestrian_zone":
        print("[SENAL] ZONA PEATONAL — reduciendo velocidad a la mitad")
        velocidad_actual = VELOCIDAD_PEATONAL

    elif label == "robots_only":
        print("[SENAL] SOLO ROBOTS — velocidad normal")
        velocidad_actual = VELOCIDAD

    elif label == "stop":
        print("[SENAL] STOP — deteniendo 5 segundos")
        car.set_motor(0, 0, 0, 0)
        car.set_car_motion(0, 0, 0)
        time.sleep(5)
        print("[SENAL] STOP — continuando")

    elif label == "loading_zone":
        print("[SENAL] ZONA DE CARGA — estacionando y enviando mensaje ROS")
        car.set_motor(0, 0, 0, 0)
        car.set_car_motion(0, 0, 0)
        if pub_carga:
            msg = String()
            msg.data = "LISTO_PARA_CARGA"
            pub_carga.publish(msg)
            print("[ROS] Mensaje publicado: LISTO_PARA_CARGA")
        esperando_confirmacion = True
        print("[SENAL] Esperando confirmacion desde la PC...")
        while esperando_confirmacion and running:
            time.sleep(0.5)
        print("[SENAL] Confirmacion recibida — yendo a zona de estacionamiento")
        mision_carga_completada = True

    elif label == "parking_zone":
        if mision_carga_completada:
            print("[SENAL] ESTACIONAMIENTO — mision de carga completada, estacionando")
            car.set_motor(0, 0, 0, 0)
            car.set_car_motion(0, 0, 0)
            mision_carga_completada = False
        else:
            print("[SENAL] ESTACIONAMIENTO — sin mision de carga, ignorando")

# =============================================================
# FUNCIONES PIVOT
# =============================================================
def set_servo(angle_deg):
    try:
        car.set_pwm_servo(STEERING_SERVO_ID, int(angle_deg))
    except Exception as e:
        print(f'set_pwm_servo error: {e}')

def compute_pivot_motors(w_sign):
    magnitude = int(min(PIVOT_SCALE, PIVOT_MAX))
    if magnitude < PIVOT_MIN:
        magnitude = PIVOT_MIN
    sign = PIVOT_LEFT_SIGN if w_sign > 0 else -PIVOT_LEFT_SIGN
    left = sign * magnitude
    right = -sign * magnitude
    return left, right

def hacer_pivot(w_sign, duracion=1.2):
    print(f"PIVOT {'IZQUIERDA' if w_sign > 0 else 'DERECHA'}")
    if USE_SERVO_COUNTER_STEER:
        angle = SERVO_CENTER_DEG + (SERVO_COUNTER_STEER_DEG * w_sign)
    else:
        angle = SERVO_CENTER_DEG
    set_servo(angle)
    time.sleep(SERVO_SETTLE_SEC)
    left, right = compute_pivot_motors(w_sign)
    inicio = time.time()
    fase = 'on'
    fase_inicio = time.time()
    while time.time() - inicio < duracion:
        ahora = time.time()
        fase_elapsed = ahora - fase_inicio
        if USE_PULSED_PIVOT:
            if fase == 'on':
                car.set_motor(left, left, right, right)
                if fase_elapsed >= PULSE_ON_SEC:
                    car.set_motor(0, 0, 0, 0)
                    fase = 'off'
                    fase_inicio = ahora
            else:
                if fase_elapsed >= PULSE_OFF_SEC:
                    car.set_motor(left, left, right, right)
                    fase = 'on'
                    fase_inicio = ahora
        else:
            car.set_motor(left, left, right, right)
        time.sleep(0.02)
    car.set_motor(0, 0, 0, 0)
    set_servo(SERVO_CENTER_DEG)
    time.sleep(0.3)
    print("Pivot completado")

# =============================================================
# CAMARA Y STREAM
# =============================================================
def camera_loop():
    global latest_frame, running, senal_detectada
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if APPLY_CLAHE else None
    detector = cv2.SIFT_create()
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    print("[INFO] Cargando referencias de señales...")
    refs = load_references(detector, clahe)
    print(f"[INFO] {len(refs)} señales cargadas")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)

    frame_count = 0
    last_detections = {}

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
        if frame_count % PROCESS_EVERY_N == 0:
            gray = preprocess_gray(frame, clahe)
            kp, des = detector.detectAndCompute(gray, None)
            last_detections = {}
            restricted_hit = False

            for label, ref in refs.items():
                corners, n_in = detect_sign(ref, kp, des, matcher, frame.shape)
                if corners is not None:
                    last_detections[label] = (corners, n_in, ref["text"], ref["color"])
                    if label == "restricted_area":
                        restricted_hit = True
                    senal_detectada = label

            if not restricted_hit:
                corners = detect_red_circle(frame)
                if corners is not None:
                    last_detections["restricted_area"] = (corners, 0, "AREA RESTRINGIDA", (50,50,220))
                    senal_detectada = "restricted_area"

        for label, (corners, n_in, text, color) in last_detections.items():
            draw_detection(frame, corners, text, color, n_in)

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

# =============================================================
# LIDAR Y NAVEGACION
# =============================================================
class LidarNode(Node):
    def __init__(self):
        super().__init__('navegacion_autonoma')
        self.sub = self.create_subscription(LaserScan, '/scan', self.callback, 10)
        global pub_carga
        pub_carga = self.create_publisher(String, '/robot_status', 10)

    def callback(self, msg):
        global distancia_frente, distancia_izquierda, distancia_derecha
        ranges = msg.ranges
        n = len(ranges)

        def minimo(inicio, fin):
            vals = [r for r in ranges[inicio:fin] if 0.1 < r < 12.0]
            return min(vals) if vals else 999

        distancia_frente = minimo(0, int(n*0.08))
        frente2 = minimo(int(n*0.92), n)
        distancia_frente = min(distancia_frente, frente2)
        distancia_izquierda = minimo(int(n*0.08), int(n*0.30))
        distancia_derecha = minimo(int(n*0.70), int(n*0.92))

def frenar():
    print("FRENANDO")
    car.set_motor(0, 0, 0, 0)
    car.set_car_motion(0, 0, 0)
    time.sleep(0.5)

def avanzar_recto():
    car.set_car_motion(0, 0, 0)
    time.sleep(0.05)
    car.set_motor(velocidad_actual, velocidad_actual, velocidad_actual, velocidad_actual)

def retroceder():
    print("RETROCEDIENDO")
    car.set_car_motion(0, 0, 0)
    time.sleep(0.1)
    car.set_motor(-400, -400, -400, -400)
    time.sleep(0.6)
    car.set_motor(0, 0, 0, 0)
    time.sleep(0.3)

senal_procesada = None

def navegacion_loop():
    global running, senal_detectada, senal_procesada

    time.sleep(2)
    print("Iniciando navegacion autonoma con señales...")

    while running:
        print(f"F:{distancia_frente:.2f} D:{distancia_derecha:.2f} I:{distancia_izquierda:.2f} | Vel:{velocidad_actual}")

        # Procesar señal si hay una nueva
        if senal_detectada and senal_detectada != senal_procesada:
            senal_procesada = senal_detectada
            ejecutar_comportamiento(senal_detectada)
            senal_detectada = None
            continue

        if distancia_frente >= DISTANCIA_MIN:
            if distancia_frente < DISTANCIA_FRENO:
                print("Cerca — lento")
                car.set_car_motion(0, 0, 0)
                time.sleep(0.05)
                car.set_motor(int(velocidad_actual * 0.5), int(velocidad_actual * 0.5),
                              int(velocidad_actual * 0.5), int(velocidad_actual * 0.5))
            else:
                print("Avanzando")
                avanzar_recto()

        else:
            frenar()
            print(f"Obstaculo — D:{distancia_derecha:.2f} I:{distancia_izquierda:.2f}")

            if distancia_derecha > DISTANCIA_LATERAL:
                print("--- PIVOT DERECHA ---")
                hacer_pivot(-1)
            elif distancia_izquierda > DISTANCIA_LATERAL:
                print("--- PIVOT IZQUIERDA ---")
                hacer_pivot(+1)
            else:
                print("--- RETROCEDER ---")
                retroceder()
                time.sleep(0.3)
                if distancia_derecha > DISTANCIA_LATERAL:
                    hacer_pivot(-1)
                elif distancia_izquierda > DISTANCIA_LATERAL:
                    hacer_pivot(+1)
                else:
                    retroceder()
                    retroceder()

        time.sleep(0.1)

    car.set_motor(0, 0, 0, 0)
    car.set_car_motion(0, 0, 0)

# =============================================================
# CONFIRMACION DE CARGA DESDE PC
# =============================================================
class ConfirmacionNode(Node):
    def __init__(self):
        super().__init__('confirmacion_carga')
        self.sub = self.create_subscription(String, '/confirmar_carga',
                                            self.callback, 10)

    def callback(self, msg):
        global esperando_confirmacion
        if msg.data == "CONTINUAR":
            print("[ROS] Confirmacion recibida desde PC")
            esperando_confirmacion = False

# =============================================================
# MAIN
# =============================================================
def main():
    global running
    rclpy.init()
    node = LidarNode()
    confirmacion_node = ConfirmacionNode()

    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=server_loop, daemon=True).start()
    threading.Thread(target=navegacion_loop, daemon=True).start()

    print("Stream en http://10.43.54.184:8080")
    print("Presiona Ctrl+C para detener")

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(confirmacion_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        running = False
        car.set_motor(0, 0, 0, 0)
        car.set_car_motion(0, 0, 0)
        set_servo(SERVO_CENTER_DEG)
        print("Detenido")
    finally:
        executor.shutdown()
        node.destroy_node()
        confirmacion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
