import cv2, mediapipe as mp, pyautogui, time

hands = mp.solutions.hands.Hands(0.7)
cap = cv2.VideoCapture(0)
prev_positions, max_positions = [], 5
gesture_active, pinch_active = False, False
last_gesture_time, gesture_threshold, gesture_cooldown = 0, 50, 1

def is_fist_except_index(h): return sum(h.landmark[f].y < h.landmark[f-2].y for f in [8,12,16,20]) == 1
def is_fist(h): return all(h.landmark[f].y > h.landmark[f-2].y for f in [8,12,16,20])
def is_palm_facing_camera(h): return h.landmark[0].z > h.landmark[9].z

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_time = time.time()

    if results.multi_hand_landmarks:
        for h in results.multi_hand_landmarks:
            if not is_palm_facing_camera(h) or is_fist(h): continue

            height, width, _ = frame.shape
            ix, iy = int(h.landmark[8].x * width), int(h.landmark[8].y * height)
            tx, ty = int(h.landmark[4].x * width), int(h.landmark[4].y * height)

            cv2.line(frame, (tx, ty), (ix, iy), (0,255,0), 2)
            cv2.circle(frame, (ix, iy), 10, (255,0,0), -1)
            cv2.circle(frame, (tx, ty), 10, (0,0,255), -1)

            if ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5 < 30:
                if not pinch_active and (current_time - last_gesture_time) > 1:
                    pinch_active, last_gesture_time = True, current_time
                    pyautogui.press("playpause")
            else: pinch_active = False

            if is_fist_except_index(h):
                prev_positions.append((ix, iy))
                if len(prev_positions) > max_positions: prev_positions.pop(0)
                for i in range(1, len(prev_positions)): cv2.line(frame, prev_positions[i-1], prev_positions[i], (0,255,0), 2)

                if len(prev_positions) >= max_positions:
                    start_x, end_x = prev_positions[0][0], prev_positions[-1][0]
                    if abs(end_x - start_x) > gesture_threshold and not gesture_active and (current_time - last_gesture_time) > gesture_cooldown:
                        gesture_active, last_gesture_time = True, current_time
                        pyautogui.press("nexttrack" if end_x > start_x else "prevtrack")
                    elif abs(end_x - start_x) <= gesture_threshold or (current_time - last_gesture_time) < gesture_cooldown:
                        gesture_active = False
            else:
                prev_positions.clear()
                gesture_active = False

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()