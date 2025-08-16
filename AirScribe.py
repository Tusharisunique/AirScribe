
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import copy
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
    model_complexity=1
)

WHITE_DRAW = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)
WHITE_CONN = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera. Check permissions and camera device.")
    exit(1)

canvas = None
MARKER_COLOR = (255, 255, 255)
marker_thickness = 10
eraser_thickness = 40

actions = []
current_action = None

calculator_open = False
guide_open = False
marker_slider_open = False
eraser_slider_open = False

finger_smooth = deque(maxlen=1)
PINCH_DOWN_RATIO = 0.018
PINCH_UP_RATIO = 0.026
is_pinched = False
last_ui_click_time = 0.0
UI_DEBOUNCE = 0.16

last_calc_click_time = 0.0
CALC_DEBOUNCE = 0.14
calc_expression = ""
calc_result = ""

clear_hold_start = None
CLEAR_HOLD_SECS = 2.0
pinky_undo_debounce = 0.35
last_pinky_undo_time = 0.0

screenshot_dir = "whiteboard_screens"
os.makedirs(screenshot_dir, exist_ok=True)

def count_open_fingers(landmarks):
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP,  mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP,   mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP,         mp_hands.HandLandmark.PINKY_PIP),
    ]
    count = 0
    for tip, pip in fingers:
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            count += 1
    return count

def is_pinky_up(landmarks):
    return landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

def pinch_distance_px(landmarks, width, height):
    t = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    i = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x1, y1 = int(t.x * width), int(t.y * height)
    x2, y2 = int(i.x * width), int(i.y * height)
    return np.hypot(x2 - x1, y2 - y1)

def within_rect(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_translucent_box(img, rect, alpha=0.35, color=(30,30,30)):
    x1,y1,x2,y2 = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

def rebuild_canvas(canvas_obj, actions_list, toolbar_rect=None):
    canvas_obj[:] = 0
    for act in actions_list:
        if act['type'] == 'draw':
            pts = act['points']
            if toolbar_rect:
                pts = [p for p in pts if not within_rect(p, toolbar_rect)]
            if not pts: continue
            if len(pts) == 1:
                cv2.circle(canvas_obj, pts[0], max(1, act['thickness']//2), MARKER_COLOR, -1, lineType=cv2.LINE_AA)
            else:
                for i in range(1, len(pts)):
                    p1, p2 = pts[i-1], pts[i]
                    cv2.line(canvas_obj, p1, p2, MARKER_COLOR, act['thickness'], cv2.LINE_AA)
        elif act['type'] == 'erase':
            pts = act['points']
            if toolbar_rect:
                pts = [p for p in pts if not within_rect(p, toolbar_rect)]
            for p in pts:
                cv2.circle(canvas_obj, p, act['thickness'], (0,0,0), -1, lineType=cv2.LINE_AA)

def ui_layout(w, h):
    pad = 12
    btn_h = 48
    btn_w = 110
    small_w = 48
    value_w = 72
    gap = 10
    right = w - pad
    top = pad

    guide_btn = (right - btn_w, top, right, top + btn_h)
    calc_btn  = (right - 2*btn_w - gap, top, right - btn_w - gap, top + btn_h)

    y2 = guide_btn[3] + gap
    pen_icon = (right - small_w, y2, right, y2 + btn_h)
    m_plus   = (pen_icon[0] - small_w - 6, y2, pen_icon[0] - 6, y2 + btn_h)
    m_value  = (m_plus[0] - value_w - 6, y2, m_plus[0] - 6, y2 + btn_h)
    m_minus  = (m_value[0] - small_w - 6, y2, m_value[0] - 6, y2 + btn_h)

    y3 = pen_icon[3] + gap
    er_icon = (right - small_w, y3, right, y3 + btn_h)
    e_plus   = (er_icon[0] - small_w - 6, y3, er_icon[0] - 6, y3 + btn_h)
    e_value  = (e_plus[0] - value_w - 6, y3, e_plus[0] - 6, y3 + btn_h)
    e_minus  = (e_value[0] - small_w - 6, y3, e_value[0] - 6, y3 + btn_h)

    slider_panel = (w - 360 - pad, er_icon[3] + gap + 6, w - pad, er_icon[3] + gap + 6 + 110)

    toolbar_left = calc_btn[0]
    toolbar_top = top
    toolbar_right = w - pad
    toolbar_bottom = er_icon[3]
    toolbar_rect = (toolbar_left, toolbar_top, toolbar_right, toolbar_bottom)

    return {
        "guide_btn": guide_btn,
        "calc_btn": calc_btn,
        "pen_icon": pen_icon, "m_minus": m_minus, "m_value": m_value, "m_plus": m_plus,
        "er_icon": er_icon, "e_minus": e_minus, "e_value": e_value, "e_plus": e_plus,
        "slider_panel": slider_panel,
        "toolbar_rect": toolbar_rect
    }

def draw_button(img, rect, label, hovered=False, alpha=0.25):
    x1,y1,x2,y2 = rect
    img = draw_translucent_box(img, rect, alpha=(alpha+0.15 if hovered else alpha), color=(40,40,40))
    cv2.rectangle(img,(x1,y1),(x2,y2),(170,170,170),1)
    if label:
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
        tx = x1 + (x2-x1 - tw)//2
        ty = y1 + (y2-y1 + th)//2 - 4
        cv2.putText(img, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245,245,245), 2)
    return img

def draw_icon_pen(img, rect, hovered=False):
    x1,y1,x2,y2 = rect
    img = draw_translucent_box(img, rect, alpha=(0.2 if not hovered else 0.4), color=(40,40,40))
    cv2.rectangle(img,(x1,y1),(x2,y2),(170,170,170),1)
    cv2.line(img,(x1+8,y2-12),(x2-8,y1+12),(255,255,255),2,cv2.LINE_AA)
    return img

def draw_icon_eraser(img, rect, hovered=False):
    x1,y1,x2,y2 = rect
    img = draw_translucent_box(img, rect, alpha=(0.2 if not hovered else 0.4), color=(40,40,40))
    cv2.rectangle(img,(x1,y1),(x2,y2),(170,170,170),1)
    pts = np.array([[x1+8,y2-12],[x1+22,y1+12],[x2-8,y1+12],[x2-22,y2-12]], np.int32)
    cv2.polylines(img, [pts], True, (255,255,255), 2, cv2.LINE_AA)
    return img

def draw_value(img, rect, val):
    x1,y1,x2,y2 = rect
    img = draw_translucent_box(img, rect, alpha=0.18, color=(20,20,20))
    cv2.rectangle(img,(x1,y1),(x2,y2),(150,150,150),1)
    s = f"{int(val)} px"
    (tw,th),_ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx = x1 + (x2-x1 - tw)//2
    ty = y1 + (y2-y1 + th)//2 - 2
    cv2.putText(img, s, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)
    return img

def draw_slider(img, rect, min_v, max_v, value, label, fingertip=None, pinched=False):
    x1,y1,x2,y2 = rect
    img = draw_translucent_box(img, rect, alpha=0.45, color=(18,18,18))
    cv2.rectangle(img,(x1,y1),(x2,y2),(140,140,140),1)
    cv2.putText(img, label, (x1+12, y1+28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230,230,230), 2)
    track_y = y1 + 54
    left = x1 + 14; right = x2 - 14
    cv2.line(img,(left,track_y),(right,track_y),(200,200,200),2,cv2.LINE_AA)
    t = (value - min_v) / float(max_v - min_v)
    t = max(0.0, min(1.0, t))
    knob_x = int(left + t*(right-left))
    knob_rect = (knob_x-12, track_y-14, knob_x+12, track_y+14)
    hovered = (fingertip is not None and within_rect(fingertip, knob_rect))
    kcol = (255,255,255) if hovered else (220,220,220)
    cv2.rectangle(img,(knob_rect[0],knob_rect[1]),(knob_rect[2],knob_rect[3]),kcol,-1)
    cv2.rectangle(img,(knob_rect[0],knob_rect[1]),(knob_rect[2],knob_rect[3]),(100,100,100),1)
    vs = f"{int(value)} px"
    (tw,th),_ = cv2.getTextSize(vs, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.putText(img, vs, (x2 - tw - 14, y2 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)

    new_value = value
    dragging = False
    if fingertip is not None and pinched:
        if hovered or (track_y-20 <= fingertip[1] <= track_y+20 and left <= fingertip[0] <= right):
            dragging = True
            xclamped = max(left, min(right, fingertip[0]))
            t2 = (xclamped - left) / float(right-left)
            new_value = int(min_v + t2*(max_v - min_v))
    return img, new_value, dragging

calc_buttons = [
    ["7","8","9","/"],
    ["4","5","6","*"],
    ["1","2","3","-"],
    ["0",".","=","+"],
    ["C","(",")","Close"],
    ["<-","","",""]
]

def draw_calculator_center(img, fingertip=None, click_now=False):
    global calc_expression, calc_result, last_calc_click_time
    h, w = img.shape[:2]
    panel_w = min(520, int(w*0.7))
    panel_h = min(420, int(h*0.7))
    x1 = (w - panel_w)//2
    y1 = (h - panel_h)//2
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    img = draw_translucent_box(img, (x1,y1,x2,y2), alpha=0.45, color=(28,28,28))
    pad = 12
    disp_h = 70
    res_h = 48
    cv2.rectangle(img, (x1+pad,y1+pad), (x2-pad,y1+pad+disp_h), (0,0,0), -1)
    cv2.putText(img, calc_expression[-28:], (x1+pad+8, y1+pad+45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.rectangle(img, (x1+pad, y1+2*pad+disp_h), (x2-pad, y1+2*pad+disp_h+res_h), (15,15,15), -1)
    cv2.putText(img, str(calc_result)[-20:], (x1+pad+8, y1+2*pad+disp_h+36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180,255,200), 2)

    grid_top = y1 + 3*pad + disp_h + res_h
    grid_left = x1 + pad
    grid_right = x2 - pad
    grid_bottom = y2 - pad
    rows = len(calc_buttons)
    cols = 4
    cell_w = (grid_right - grid_left) // cols
    cell_h = (grid_bottom - grid_top) // rows
    now = time.time()
    pressed = None

    for r in range(rows):
        for c in range(cols):
            label = calc_buttons[r][c] if c < len(calc_buttons[r]) else ""
            if label == "" and r == rows-1:
                continue
            bx1 = grid_left + c*cell_w
            by1 = grid_top + r*cell_h
            bx2 = bx1 + cell_w - 6
            by2 = by1 + cell_h - 6
            hovered = (fingertip is not None and within_rect(fingertip, (bx1,by1,bx2,by2)))
            img = draw_button(img, (bx1,by1,bx2,by2), label, hovered=hovered, alpha=0.22)
            if click_now and hovered and (now - last_calc_click_time > CALC_DEBOUNCE):
                pressed = label

    if pressed is not None:
        last_calc_click_time = now
        if pressed == "Close":
            return img, "close"
        elif pressed == "C":
            calc_expression = ""
            calc_result = ""
        elif pressed == "<-":
            calc_expression = calc_expression[:-1]
        elif pressed == "=":
            safe = "".join(ch for ch in calc_expression if ch in "0123456789.+-*/() ")
            try:
                calc_result = str(eval(safe, {"__builtins__":None}, {})) if safe.strip() else ""
            except Exception:
                calc_result = "Err"
        else:
            calc_expression += pressed
    return img, None

print("Starting virtual whiteboard (final). Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        actions.clear()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingertip = None
    open_count = 0
    pinky_flag = False
    dragging_slider = False

    tool_active = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                  WHITE_DRAW, WHITE_CONN)

        idx = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ix, iy = int(idx.x * w), int(idx.y * h)
        finger_smooth.append((ix, iy))
        sx = int(np.mean([p[0] for p in finger_smooth]))
        sy = int(np.mean([p[1] for p in finger_smooth]))
        fingertip = (sx, sy)

        open_count = count_open_fingers(hand)
        pinky_flag = is_pinky_up(hand)

        pdist = pinch_distance_px(hand, w, h)
        down_th = PINCH_DOWN_RATIO * w
        up_th = PINCH_UP_RATIO * w
        if not is_pinched and pdist < down_th:
            is_pinched = True
        elif is_pinched and pdist > up_th:
            is_pinched = False

    L = ui_layout(w, h)
    toolbar_rect = L["toolbar_rect"]

    ui_preview = frame.copy()
    def hover(name):
        return fingertip is not None and within_rect(fingertip, L[name])

    ui_preview = draw_button(ui_preview, L["calc_btn"], "Calc", hover("calc_btn"))
    ui_preview = draw_button(ui_preview, L["guide_btn"], "Guide", hover("guide_btn"))
    ui_preview = draw_icon_pen(ui_preview, L["pen_icon"], hover("pen_icon"))
    ui_preview = draw_button(ui_preview, L["m_minus"], "-", hover("m_minus"))
    ui_preview = draw_value(ui_preview, L["m_value"], marker_thickness)
    ui_preview = draw_button(ui_preview, L["m_plus"], "+", hover("m_plus"))
    ui_preview = draw_icon_eraser(ui_preview, L["er_icon"], hover("er_icon"))
    ui_preview = draw_button(ui_preview, L["e_minus"], "-", hover("e_minus"))
    ui_preview = draw_value(ui_preview, L["e_value"], eraser_thickness)
    ui_preview = draw_button(ui_preview, L["e_plus"], "+", hover("e_plus"))

    now = time.time()
    pinch_click = (is_pinched and (now - last_ui_click_time > UI_DEBOUNCE))

    if fingertip is not None and pinch_click:
        clicked = False
        if within_rect(fingertip, L["calc_btn"]):
            calculator_open = not calculator_open; clicked = True
        elif within_rect(fingertip, L["guide_btn"]):
            guide_open = not guide_open; clicked = True
        elif within_rect(fingertip, L["m_minus"]):
            marker_thickness = max(1, marker_thickness - 2); clicked = True
        elif within_rect(fingertip, L["m_plus"]):
            marker_thickness = min(200, marker_thickness + 2); clicked = True
        elif within_rect(fingertip, L["e_minus"]):
            eraser_thickness = max(8, eraser_thickness - 5); clicked = True
        elif within_rect(fingertip, L["e_plus"]):
            eraser_thickness = min(300, eraser_thickness + 5); clicked = True
        elif within_rect(fingertip, L["m_value"]):
            marker_slider_open = not marker_slider_open; eraser_slider_open = False; clicked = True
            if current_action is not None:
                filtered_points = [p for p in current_action['points'] if not within_rect(p, toolbar_rect)]
                if filtered_points:
                    new_act = copy.deepcopy(current_action)
                    new_act['points'] = filtered_points
                    actions.append(new_act)
                current_action = None
        elif within_rect(fingertip, L["e_value"]):
            eraser_slider_open = not eraser_slider_open; marker_slider_open = False; clicked = True
            if current_action is not None:
                filtered_points = [p for p in current_action['points'] if not within_rect(p, toolbar_rect)]
                if filtered_points:
                    new_act = copy.deepcopy(current_action)
                    new_act['points'] = filtered_points
                    actions.append(new_act)
                current_action = None

        if clicked:
            last_ui_click_time = now
            if current_action is not None:
                filtered_points = [p for p in current_action['points'] if not within_rect(p, toolbar_rect)]
                if filtered_points:
                    new_act = copy.deepcopy(current_action)
                    new_act['points'] = filtered_points
                    actions.append(new_act)
                current_action = None
            rebuild_canvas(canvas, actions, toolbar_rect)

    tool_active = calculator_open or marker_slider_open or eraser_slider_open or guide_open

    dragging_slider = False
    if marker_slider_open:
        ui_preview, marker_thickness, dragging_slider = draw_slider(ui_preview, L["slider_panel"], 1, 200, marker_thickness, "Marker Size", fingertip=fingertip, pinched=is_pinched)
        tool_active = True
    elif eraser_slider_open:
        ui_preview, eraser_thickness, dragging_slider = draw_slider(ui_preview, L["slider_panel"], 8, 300, eraser_thickness, "Eraser Size", fingertip=fingertip, pinched=is_pinched)
        tool_active = True

    calc_action = None
    if calculator_open:
        click_now = (fingertip is not None and is_pinched and (time.time() - last_calc_click_time > CALC_DEBOUNCE))
        ui_preview, calc_action = draw_calculator_center(ui_preview, fingertip=fingertip, click_now=click_now)
        if calc_action == "close":
            calculator_open = False
        tool_active = True

    if guide_open:
        gx1, gy1, gx2, gy2 = L["slider_panel"]
        gy2 = gy1 + 220
        ui_preview = draw_translucent_box(ui_preview, (gx1,gy1, gx2, gy2), alpha=0.35, color=(12,12,12))
        tips = [
            "Guide:",
            "• 1 finger = Draw (white).",
            "• 2 fingers = Erase (eraser size top-right).",
            "• Pinch (index+thumb) to click UI or drag slider knob.",
            "• Tap Marker/Eraser value to open slider; pinch-drag knob.",
            "• Calc supports digits, ., + - * / ( ), =, <- (backspace), C (clear).",
            f"• Hold 4 fingers for {CLEAR_HOLD_SECS:.0f}s to Clear screen.",
            "• Show pinky finger to Undo last stroke (immediate).",
            "• Keys: s=save canvas, c=clear, q=quit."
        ]
        y = gy1 + 28
        for i, line in enumerate(tips):
            cv2.putText(ui_preview, line, (gx1+12, y + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)

    ui_regions = [L["calc_btn"], L["guide_btn"],
                  L["pen_icon"], L["m_minus"], L["m_value"], L["m_plus"],
                  L["er_icon"], L["e_minus"], L["e_value"], L["e_plus"]]
    if marker_slider_open or eraser_slider_open:
        ui_regions.append(L["slider_panel"])

    in_ui = False
    if fingertip is not None:
        for R in ui_regions:
            if within_rect(fingertip, R):
                in_ui = True
                break

    if results.multi_hand_landmarks and open_count == 4 and not in_ui and not tool_active:
        if clear_hold_start is None:
            clear_hold_start = time.time()
        else:
            elapsed = time.time() - clear_hold_start
            remaining = CLEAR_HOLD_SECS - elapsed
            cv2.putText(frame, f"Clearing in {max(0,int(remaining)+1)}s", (w//2 - 120, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 3)
            if elapsed >= CLEAR_HOLD_SECS:
                actions.clear()
                rebuild_canvas(canvas, actions, toolbar_rect)
                clear_hold_start = None
                cv2.putText(frame, "Canvas Cleared", (w//2 - 140, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    else:
        clear_hold_start = None

    if results.multi_hand_landmarks and pinky_flag and not in_ui and not tool_active:
        if time.time() - last_pinky_undo_time > pinky_undo_debounce:
            if actions:
                actions.pop()
                rebuild_canvas(canvas, actions, toolbar_rect)
            last_pinky_undo_time = time.time()
        tool_active = True

    if results.multi_hand_landmarks and fingertip is not None and not in_ui and not tool_active and not dragging_slider and not marker_slider_open and not eraser_slider_open:
        if open_count == 1:
            if not within_rect(fingertip, toolbar_rect):
                if current_action is None:
                    current_action = {'type':'draw', 'points':[fingertip], 'thickness':marker_thickness}
                    cv2.circle(canvas, fingertip, max(1, marker_thickness//2), MARKER_COLOR, -1, lineType=cv2.LINE_AA)
                else:
                    current_action['points'].append(fingertip)
                    if len(current_action['points']) >= 2:
                        p1, p2 = current_action['points'][-2], current_action['points'][-1]
                        distance = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        if distance > marker_thickness:
                            divisor = max(1, marker_thickness//2)
                            num_interp = min(int(distance // divisor), 5)
                            for i in range(1, num_interp):
                                t = i / num_interp
                                x = int(p1[0] * (1-t) + p2[0] * t)
                                y = int(p1[1] * (1-t) + p2[1] * t)
                                cv2.circle(canvas, (x, y), max(1, marker_thickness//2), MARKER_COLOR, -1, lineType=cv2.LINE_AA)
                        cv2.line(canvas, p1, p2, MARKER_COLOR, current_action['thickness'], cv2.LINE_AA)
            else:
                if current_action is not None:
                    filtered = [p for p in current_action['points'] if not within_rect(p, toolbar_rect)]
                    if filtered:
                        new_act = copy.deepcopy(current_action)
                        new_act['points'] = filtered
                        actions.append(new_act)
                    current_action = None
        elif open_count == 2:
            if not within_rect(fingertip, toolbar_rect):
                if current_action is None:
                    current_action = {'type':'erase', 'points':[fingertip], 'thickness':eraser_thickness}
                    cv2.circle(canvas, fingertip, eraser_thickness, (0,0,0), -1, lineType=cv2.LINE_AA)
                else:
                    current_action['points'].append(fingertip)
                    if len(current_action['points']) >= 2:
                        p1, p2 = current_action['points'][-2], current_action['points'][-1]
                        distance = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        divisor = max(1, eraser_thickness//3)
                        num_points = max(int(distance // divisor), 1)
                        for i in range(num_points + 1):
                            t = i / num_points
                            x = int(p1[0] * (1-t) + p2[0] * t)
                            y = int(p1[1] * (1-t) + p2[1] * t)
                            cv2.circle(canvas, (x, y), eraser_thickness, (0,0,0), -1, lineType=cv2.LINE_AA)
            else:
                if current_action is not None:
                    filtered = [p for p in current_action['points'] if not within_rect(p, toolbar_rect)]
                    if filtered:
                        new_act = copy.deepcopy(current_action)
                        new_act['points'] = filtered
                        actions.append(new_act)
                    current_action = None
        else:
            if current_action is not None:
                filtered = [p for p in current_action['points'] if not within_rect(p, toolbar_rect)]
                if filtered:
                    new_act = copy.deepcopy(current_action)
                    new_act['points'] = filtered
                    actions.append(new_act)
                current_action = None
    else:
        if current_action is not None:
            filtered = [p for p in current_action['points'] if not within_rect(p, toolbar_rect)]
            if filtered:
                new_act = copy.deepcopy(current_action)
                new_act['points'] = filtered
                actions.append(new_act)
            current_action = None

    if current_action is None and actions:
        rebuild_canvas(canvas, actions, toolbar_rect)

    output = cv2.addWeighted(frame, 1, canvas, 1, 0)
    output = draw_translucent_box(output, toolbar_rect, alpha=0.20, color=(30,30,80))
    output = draw_button(output, L["calc_btn"], "Calc", hover("calc_btn"))
    output = draw_button(output, L["guide_btn"], "Guide", hover("guide_btn"))
    output = draw_icon_pen(output, L["pen_icon"], hover("pen_icon"))
    output = draw_button(output, L["m_minus"], "-", hover("m_minus"))
    output = draw_value(output, L["m_value"], marker_thickness)
    output = draw_button(output, L["m_plus"], "+", hover("m_plus"))
    output = draw_icon_eraser(output, L["er_icon"], hover("er_icon"))
    output = draw_button(output, L["e_minus"], "-", hover("e_minus"))
    output = draw_value(output, L["e_value"], eraser_thickness)
    output = draw_button(output, L["e_plus"], "+", hover("e_plus"))

    if marker_slider_open:
        output, marker_thickness, _ = draw_slider(output, L["slider_panel"], 1, 200, marker_thickness, "Marker Size", fingertip=fingertip, pinched=is_pinched)
    elif eraser_slider_open:
        output, eraser_thickness, _ = draw_slider(output, L["slider_panel"], 8, 300, eraser_thickness, "Eraser Size", fingertip=fingertip, pinched=is_pinched)

    if calculator_open:
        click_now = (fingertip is not None and is_pinched and (time.time() - last_calc_click_time > CALC_DEBOUNCE))
        output, action = draw_calculator_center(output, fingertip=fingertip, click_now=click_now)
        if action == "close":
            calculator_open = False

    if guide_open:
        gx1, gy1, gx2, gy2 = L["slider_panel"]
        gy2 = gy1 + 220
        output = draw_translucent_box(output, (gx1,gy1, gx2, gy2), alpha=0.35, color=(12,12,12))
        tips = [
            "Air Scribe:",
            "• 1 finger = Draw (white).",
            "• 2 fingers = Erase.",
            "• Pinch (index+thumb) to click UI or drag slider knob.",
            "• Tap Marker/Eraser value to open slider; pinch-drag knob.",
            "• Calc supports digits, ., + - * / ( ), =, <- (backspace), C (clear).",
            f"• Hold 4 fingers for {CLEAR_HOLD_SECS:.0f}s to Clear screen.",
            "• Show pinky finger to Undo last stroke (immediate).",
            "• Keys: s=save, c=clear, q=quit."
        ]
        y = gy1 + 28
        for i, line in enumerate(tips):
            cv2.putText(output, line, (gx1+12, y + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)

    cv2.putText(output, f"Marker: {marker_thickness}px   Eraser: {eraser_thickness}px", (12, h-48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,255,0), 2)
    mode_text = "surf"
    if results.multi_hand_landmarks:
        if open_count == 1: mode_text = "draw"
        elif open_count == 2: mode_text = "erase"
    cv2.putText(output, f"Mode: {mode_text}", (12, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,255,0), 2)

    cv2.imshow("Virtual Whiteboard (final)", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"{int(time.time())}_screenshot.png"
        path = os.path.join(screenshot_dir, fname)
        cv2.imwrite(path, canvas)
        print("Saved:", path)
    elif key == ord('c'):
        actions.clear()
        rebuild_canvas(canvas, actions, toolbar_rect)

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
