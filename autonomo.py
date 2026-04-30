from Rosmaster_Lib import Rosmaster
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import threading, time, cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- Configuracion navegacion ---
DISTANCIA_MIN = 0.5
DISTANCIA_FRENO = 0.8
DISTANCIA_LATERAL = 0.4
VELOCIDAD = 800

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

# --- Funciones Pivot ---
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

    # Centrar servo con counter-steer
    if USE_SERVO_COUNTER_STEER:
        angle = SERVO_CENTER_DEG + (SERVO_COUNTER_STEER_DEG * w_sign)
    else:
        angle = SERVO_CENTER_DEG
    set_servo(angle)
    time.sleep(SERVO_SETTLE_SEC)

    # Calcular motores
    left, right = compute_pivot_motors(w_sign)

    # Ejecutar pivot con pulsos
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

    # Detener y centrar
    car.set_motor(0, 0, 0, 0)
    set_servo(SERVO_CENTER_DEG)
    time.sleep(0.3)
    print("Pivot completado")

# --- Camara ---
class LidarNode(Node):
    def __init__(self):
        super().__init__('navegacion_autonoma')
        self.sub = self.create_subscription(LaserScan, '/scan', self.callback, 10)

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

# --- Navegacion ---
def frenar():
    print("FRENANDO")
    car.set_motor(0, 0, 0, 0)
    car.set_car_motion(0, 0, 0)
    time.sleep(0.5)

def avanzar_recto():
    car.set_car_motion(0, 0, 0)
    time.sleep(0.05)
    car.set_motor(VELOCIDAD, VELOCIDAD, VELOCIDAD, VELOCIDAD)

def retroceder():
    print("RETROCEDIENDO")
    car.set_car_motion(0, 0, 0)
    time.sleep(0.1)
    car.set_motor(-400, -400, -400, -400)
    time.sleep(0.6)
    car.set_motor(0, 0, 0, 0)
    time.sleep(0.3)

def navegacion_loop():
    global running
    time.sleep(2)
    print("Iniciando navegacion autonoma con G-Turn Pivot...")

    while running:
        print(f"F:{distancia_frente:.2f} D:{distancia_derecha:.2f} I:{distancia_izquierda:.2f}")

        if distancia_frente >= DISTANCIA_MIN:
            if distancia_frente < DISTANCIA_FRENO:
                print("Cerca — lento")
                car.set_car_motion(0, 0, 0)
                time.sleep(0.05)
                car.set_motor(400, 400, 400, 400)
            else:
                print("Avanzando")
                avanzar_recto()

        else:
            frenar()
            print(f"Obstaculo — D:{distancia_derecha:.2f} I:{distancia_izquierda:.2f}")

            if distancia_derecha > DISTANCIA_LATERAL:
                print("--- PIVOT DERECHA ---")
                hacer_pivot(-1)  # -1 = derecha

            elif distancia_izquierda > DISTANCIA_LATERAL:
                print("--- PIVOT IZQUIERDA ---")
                hacer_pivot(+1)  # +1 = izquierda

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

def main():
    global running
    rclpy.init()
    node = LidarNode()

    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=server_loop, daemon=True).start()
    threading.Thread(target=navegacion_loop, daemon=True).start()

    print("Stream en http://10.43.54.184:8080")
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
