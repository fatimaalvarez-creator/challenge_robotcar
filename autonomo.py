from Rosmaster_Lib import Rosmaster
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import threading, time, cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- Configuracion navegacion ---
DISTANCIA_MIN  = 0.25
VELOCIDAD      = 30
FORWARD_BONUS  = 0.15

# --- Configuracion Pivot ---
STEERING_SERVO_ID       = 1
SERVO_CENTER_DEG        = 100
SERVO_SETTLE_SEC        = 0.15
USE_PULSED_PIVOT        = True
USE_SERVO_COUNTER_STEER = True
SERVO_COUNTER_STEER_DEG = 5
PULSE_ON_SEC  = 0.10
PULSE_OFF_SEC = 0.05
PIVOT_SCALE   = 600
PIVOT_MIN     = 250
PIVOT_MAX     = 900
PIVOT_LEFT_SIGN = +1

# --- Anti-oscillacion ---
CONSECUTIVO_MAX = 3
REPEAT_PENALTY  = 0.10
REVERSE_PENALTY = 0.20

car = Rosmaster()
car.set_car_type(5)
time.sleep(0.2)
car.set_car_motion(0.0, 0.0, 0.0)
time.sleep(0.05)
car.set_pwm_servo(STEERING_SERVO_ID, SERVO_CENTER_DEG)
time.sleep(SERVO_SETTLE_SEC)

latest_frame  = None
frame_lock    = threading.Lock()
running       = True

distancia_frente    = 999
distancia_derecha   = 999
distancia_trasera   = 999
distancia_izquierda = 999

ultimo_dir        = None
consecutivo_count = 0

def set_servo(angle_deg):
    try:
        car.set_pwm_servo(STEERING_SERVO_ID, int(angle_deg))
    except Exception as e:
        print(f'set_pwm_servo error: {e}')

def compute_pivot_motors(w_sign):
    magnitude = int(min(PIVOT_SCALE, PIVOT_MAX))
    if magnitude < PIVOT_MIN:
        magnitude = PIVOT_MIN
    sign  =  PIVOT_LEFT_SIGN if w_sign > 0 else -PIVOT_LEFT_SIGN
    left  =  sign * magnitude
    right = -sign * magnitude
    return left, right

def hacer_pivot(w_sign, duracion=1.2):
    print(f"PIVOT {'IZQUIERDA' if w_sign > 0 else 'DERECHA'}")
    angle = SERVO_CENTER_DEG + (SERVO_COUNTER_STEER_DEG * w_sign) if USE_SERVO_COUNTER_STEER else SERVO_CENTER_DEG
    set_servo(angle)
    time.sleep(SERVO_SETTLE_SEC)
    left, right = compute_pivot_motors(w_sign)
    inicio      = time.time()
    fase        = 'on'
    fase_inicio = time.time()
    while time.time() - inicio < duracion:
        ahora        = time.time()
        fase_elapsed = ahora - fase_inicio
        if USE_PULSED_PIVOT:
            if fase == 'on':
                car.set_motor(left, left, right, right)
                if fase_elapsed >= PULSE_ON_SEC:
                    car.set_motor(0, 0, 0, 0)
                    fase        = 'off'
                    fase_inicio = ahora
            else:
                if fase_elapsed >= PULSE_OFF_SEC:
                    car.set_motor(left, left, right, right)
                    fase        = 'on'
                    fase_inicio = ahora
        else:
            car.set_motor(left, left, right, right)
        time.sleep(0.02)
    car.set_motor(0, 0, 0, 0)
    set_servo(SERVO_CENTER_DEG)
    time.sleep(0.3)
    print("Pivot completado")

class LidarNode(Node):
    def __init__(self):
        super().__init__('navegacion_autonoma')
        self.sub = self.create_subscription(LaserScan, '/scan', self.callback, 10)

    def callback(self, msg):
        global distancia_frente, distancia_derecha, distancia_trasera, distancia_izquierda
        ranges = msg.ranges
        n      = len(ranges)

        def minimo(inicio, fin):
            if inicio < fin:
                vals = [r for r in ranges[inicio:fin] if 0.1 < r < 12.0]
            else:
                vals = [r for r in (list(ranges[inicio:]) + list(ranges[:fin])) if 0.1 < r < 12.0]
            return min(vals) if vals else 999

        # Front is wider (30% total) to avoid missing slightly off-axis clear paths.
        # Remaining three zones share the back 70% equally (~22% each with small gaps).
        f_end    = int(n * 0.15)
        r_start  = int(n * 0.18)
        r_end    = int(n * 0.40)
        b_start  = int(n * 0.43)
        b_end    = int(n * 0.57)
        l_start  = int(n * 0.60)
        l_end    = int(n * 0.82)
        f2_start = int(n * 0.85)

        distancia_frente    = min(minimo(0, f_end), minimo(f2_start, n))
        distancia_derecha   = minimo(r_start, r_end)
        distancia_trasera   = minimo(b_start, b_end)
        distancia_izquierda = minimo(l_start, l_end)

def mejor_direccion():
    global ultimo_dir, consecutivo_count

    zonas = {
        'frente':    distancia_frente,
        'derecha':   distancia_derecha,
        'trasera':   distancia_trasera,
        'izquierda': distancia_izquierda,
    }

    libres = {d: v for d, v in zonas.items() if v >= DISTANCIA_MIN}
    if not libres:
        return None

    scores = dict(libres)

    # Boost forward — prefer going straight unless something is clearly more open
    if 'frente' in scores:
        scores['frente'] += FORWARD_BONUS

    # Penalise reversing heavily — last resort only
    if 'trasera' in scores:
        scores['trasera'] -= REVERSE_PENALTY

    # Penalise repeating the last direction to discourage flip-flopping
    if ultimo_dir and ultimo_dir in scores:
        scores[ultimo_dir] = max(0, scores[ultimo_dir] - REPEAT_PENALTY)

    # Hard lockout after too many consecutive repeats of any direction
    if consecutivo_count >= CONSECUTIVO_MAX and ultimo_dir in scores:
        del scores[ultimo_dir]
        if not scores:
            return None

    elegida = max(scores, key=scores.get)

    if elegida == ultimo_dir:
        consecutivo_count += 1
    else:
        consecutivo_count = 1
    ultimo_dir = elegida

    return elegida

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

def frenar():
    print("FRENANDO")
    car.set_motor(0, 0, 0, 0)
    car.set_car_motion(0, 0, 0)
    time.sleep(0.3)

def avanzar_recto(duracion=1.0):
    car.set_car_motion(0, 0, 0)
    time.sleep(0.05)
    car.set_motor(VELOCIDAD, VELOCIDAD, VELOCIDAD, VELOCIDAD)
    time.sleep(duracion)
    car.set_motor(0, 0, 0, 0)
    time.sleep(0.1)

def retroceder():
    print("RETROCEDIENDO")
    car.set_car_motion(0, 0, 0)
    time.sleep(0.1)
    car.set_motor(-30, -30, -30, -30)
    time.sleep(1.0)
    car.set_motor(0, 0, 0, 0)
    time.sleep(0.3)

def navegacion_loop():
    global running
    time.sleep(2)

    print("Enderezando llantas...")
    set_servo(SERVO_CENTER_DEG)
    time.sleep(1.0)
    print("Iniciando navegacion autonoma — modo mejor direccion con anti-oscilacion...")

    while running:
        print(f"F:{distancia_frente:.2f}  D:{distancia_derecha:.2f}  T:{distancia_trasera:.2f}  I:{distancia_izquierda:.2f}  | ultimo:{ultimo_dir}  consec:{consecutivo_count}")

        direccion = mejor_direccion()

        if direccion is None:
            print("Todas las direcciones bloqueadas — deteniendo")
            frenar()
            time.sleep(0.5)

        elif direccion == 'frente':
            print("AVANZAR")
            avanzar_recto()

        elif direccion == 'trasera':
            print("RETROCEDER")
            frenar()
            retroceder()

        elif direccion == 'derecha':
            print("PIVOT DERECHA")
            frenar()
            hacer_pivot(-1)

        elif direccion == 'izquierda':
            print("PIVOT IZQUIERDA")
            frenar()
            hacer_pivot(+1)

        time.sleep(0.1)

    car.set_motor(0, 0, 0, 0)
    car.set_car_motion(0, 0, 0)

def main():
    global running
    rclpy.init()
    node = LidarNode()

    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=server_loop, daemon=True).start()
    threading.Thread(target=navegacion_loop, daemon=True).start()

    print("Stream en http://10.43.48.208:8080")
    print("Presiona Ctrl+C para detener")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        running = False
        car.set_motor(0, 0, 0, 0)
        car.set_car_motion(0, 0, 0)
        set_servo(SERVO_CENTER_DEG)
        print("Detenido")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
