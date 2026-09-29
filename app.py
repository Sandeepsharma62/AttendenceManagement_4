import os
import csv
import base64
import sqlite3
from datetime import datetime, date
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FACES = DATA / "faces"
DB = DATA / "attendance.db"
MODEL = DATA / "trainer.yml"

DATA.mkdir(exist_ok=True)
FACES.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        roll_no TEXT NOT NULL UNIQUE,
        department TEXT NOT NULL,
        image_path TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Present',
        UNIQUE(student_id, date),
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
    );
    """)
    conn.commit()
    conn.close()

def logged_in():
    return session.get("logged_in") is True

def login_required():
    return logged_in()

def decode_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def detect_face(gray):
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])

def make_recognizer():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face is unavailable. Install opencv-contrib-python."
        )
    return cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )

def train_model():
    image_paths = []
    labels = []

    conn = db()
    students = conn.execute("SELECT id FROM students ORDER BY id").fetchall()
    conn.close()

    for s in students:
        sid = int(s["id"])
        folder = FACES / str(sid)
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.jpg")):
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                image_paths.append(gray)
                labels.append(sid)

    if not image_paths:
        if MODEL.exists():
            MODEL.unlink()
        return False

    recognizer = make_recognizer()
    recognizer.train(image_paths, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL))
    return True

def load_recognizer():
    if not MODEL.exists():
        return None
    try:
        recognizer = make_recognizer()
        recognizer.read(str(MODEL))
        return recognizer
    except Exception:
        return None

def stats():
    conn = db()
    total = conn.execute("SELECT COUNT(*) c FROM students").fetchone()["c"]
    today = date.today().isoformat()
    present = conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE date=? AND status='Present'",
        (today,)
    ).fetchone()["c"]
    absent = max(total - present, 0)
    percentage = round((present / total) * 100) if total else 0
    conn.close()
    return total, present, absent, percentage

@app.route("/")
def index():
    if not logged_in():
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == "admin" and password == "admin":
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))
    total, present, absent, percentage = stats()
    return render_template(
        "dashboard.html",
        total=total, present=present, absent=absent, percentage=percentage
    )

@app.route("/enroll")
def enroll():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("enroll.html")

@app.route("/attendance")
def attendance():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("attendance.html")

@app.route("/students")
def students():
    if not login_required():
        return redirect(url_for("login"))
    conn = db()
    rows = conn.execute("SELECT * FROM students ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("students.html", students=rows)

@app.route("/records")
def records():
    if not login_required():
        return redirect(url_for("login"))
    conn = db()
    rows = conn.execute("""
        SELECT a.id, a.date, a.time, a.status,
               s.roll_no, s.name, s.department
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        ORDER BY a.date DESC, a.time DESC
    """).fetchall()
    conn.close()
    return render_template("records.html", records=rows)

@app.route("/api/dashboard")
def api_dashboard():
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401
    total, present, absent, percentage = stats()
    return jsonify({
        "total": total, "present": present,
        "absent": absent, "percentage": percentage
    })

@app.route("/api/enroll/start", methods=["POST"])
def enroll_start():
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    roll_no = data.get("roll_no", "").strip()
    department = data.get("department", "").strip()

    if not name or not roll_no or not department:
        return jsonify({"error": "All fields are required"}), 400

    conn = db()
    try:
        cur = conn.execute(
            "INSERT INTO students(name,roll_no,department,image_path,created_at) VALUES(?,?,?,?,?)",
            (name, roll_no, department, "", datetime.now().isoformat(timespec="seconds"))
        )
        sid = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Roll number already exists"}), 409
    conn.close()

    folder = FACES / str(sid)
    folder.mkdir(parents=True, exist_ok=True)

    return jsonify({"ok": True, "student_id": sid, "captured": 0, "required": 8})

@app.route("/api/enroll/capture", methods=["POST"])
def enroll_capture():
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    sid = int(data.get("student_id", 0))
    frame_data = data.get("image", "")

    if not sid or not frame_data:
        return jsonify({"error": "Invalid capture data"}), 400

    conn = db()
    student = conn.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()
    conn.close()
    if not student:
        return jsonify({"error": "Student not found"}), 404

    frame = decode_image(frame_data)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect_face(gray)
    if face is None:
        return jsonify({"error": "No face detected", "captured": 0}), 200

    x, y, w, h = face
    crop = gray[y:y+h, x:x+w]
    crop = cv2.resize(crop, (200, 200))

    folder = FACES / str(sid)
    folder.mkdir(parents=True, exist_ok=True)
    current = sorted(folder.glob("*.jpg"))
    if len(current) >= 8:
        return jsonify({"ok": True, "captured": 8, "required": 8, "done": True})

    path = folder / f"{len(current)+1:02d}.jpg"
    cv2.imwrite(str(path), crop)

    captured = len(list(folder.glob("*.jpg")))
    done = captured >= 8

    if done:
        conn = db()
        conn.execute(
            "UPDATE students SET image_path=? WHERE id=?",
            (str(path.relative_to(BASE)).replace("\\", "/"), sid)
        )
        conn.commit()
        conn.close()
        train_model()

    return jsonify({
        "ok": True,
        "captured": captured,
        "required": 8,
        "done": done
    })

@app.route("/api/enroll/cancel", methods=["POST"])
def enroll_cancel():
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    sid = int(data.get("student_id", 0))
    conn = db()
    student = conn.execute("SELECT id FROM students WHERE id=?", (sid,)).fetchone()
    if student:
        conn.execute("DELETE FROM students WHERE id=?", (sid,))
        conn.commit()
    conn.close()
    import shutil
    shutil.rmtree(FACES / str(sid), ignore_errors=True)
    train_model()
    return jsonify({"ok": True})

@app.route("/api/attendance/recognize", methods=["POST"])
def recognize():
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    frame_data = data.get("image", "")
    if not frame_data:
        return jsonify({"recognized": False, "message": "No image"}), 400

    recognizer = load_recognizer()
    if recognizer is None:
        return jsonify({
            "recognized": False,
            "message": "No trained model. Enroll students first."
        })

    frame = decode_image(frame_data)
    if frame is None:
        return jsonify({"recognized": False, "message": "Invalid image"})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect_face(gray)
    if face is None:
        return jsonify({"recognized": False, "message": "No face detected"})

    x, y, w, h = face
    crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

    label, confidence = recognizer.predict(crop)

    # LBPH: lower confidence means a closer match.
    if confidence > 72:
        return jsonify({
            "recognized": False,
            "message": "Face not recognized",
            "confidence": round(float(confidence), 1)
        })

    conn = db()
    student = conn.execute(
        "SELECT * FROM students WHERE id=?", (int(label),)
    ).fetchone()

    if not student:
        conn.close()
        return jsonify({"recognized": False, "message": "Unknown student"})

    today = date.today().isoformat()
    now = datetime.now().strftime("%I:%M %p")

    existing = conn.execute(
        "SELECT * FROM attendance WHERE student_id=? AND date=?",
        (student["id"], today)
    ).fetchone()

    marked = False
    if not existing:
        conn.execute(
            "INSERT INTO attendance(student_id,date,time,status) VALUES(?,?,?,?)",
            (student["id"], today, now, "Present")
        )
        conn.commit()
        marked = True

    conn.close()

    return jsonify({
        "recognized": True,
        "marked": marked,
        "name": student["name"],
        "roll_no": student["roll_no"],
        "department": student["department"],
        "time": now,
        "confidence": round(float(confidence), 1),
        "box": [int(x), int(y), int(w), int(h)]
    })

@app.route("/api/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401
    conn = db()
    conn.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.execute("DELETE FROM students WHERE id=?", (sid,))
    conn.commit()
    conn.close()
    import shutil
    shutil.rmtree(FACES / str(sid), ignore_errors=True)
    train_model()
    return jsonify({"ok": True})

@app.route("/export/csv")
def export_csv():
    if not login_required():
        return redirect(url_for("login"))

    conn = db()
    rows = conn.execute("""
        SELECT a.date, a.time, s.roll_no, s.name, s.department, a.status
        FROM attendance a
        JOIN students s ON s.id=a.student_id
        ORDER BY a.date DESC, a.time DESC
    """).fetchall()
    conn.close()

    def generate():
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Date", "Time", "Roll No", "Name", "Department", "Status"])
        yield output.getvalue()
        output.seek(0); output.truncate(0)

        for r in rows:
            writer.writerow([
                r["date"], r["time"], r["roll_no"],
                r["name"], r["department"], r["status"]
            ])
            yield output.getvalue()
            output.seek(0); output.truncate(0)

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance.csv"}
    )

init_db()

if __name__ == "__main__":
    print("Face Attendance System")
    print("Login: admin / admin")
    print("Open: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
