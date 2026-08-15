import sqlite3


DATABASE = "patients.db"


def create_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        session_id TEXT,

        patient_name TEXT,

        patient_id TEXT,

        age INTEGER,

        gender TEXT,

        doctor_name TEXT,

        hospital TEXT,

        prediction TEXT,

        confidence REAL,

        original_image TEXT,

        heatmap_image TEXT,

        overlay_image TEXT,

        report_path TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)

    # --------------------------------------------------------
    # ADD session_id TO EXISTING DATABASE
    # --------------------------------------------------------

    cursor.execute("PRAGMA table_info(reports)")

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "session_id" not in columns:

        cursor.execute("""
            ALTER TABLE reports
            ADD COLUMN session_id TEXT
        """)

    conn.commit()

    conn.close()


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(

    session_id,

    patient_name,
    patient_id,
    age,
    gender,
    doctor_name,
    hospital,
    prediction,
    confidence,
    original_image,
    heatmap_image,
    overlay_image,
    report_path

):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO reports(

        session_id,

        patient_name,

        patient_id,

        age,

        gender,

        doctor_name,

        hospital,

        prediction,

        confidence,

        original_image,

        heatmap_image,

        overlay_image,

        report_path

    )

    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)

    """, (

        session_id,

        patient_name,

        patient_id,

        age,

        gender,

        doctor_name,

        hospital,

        prediction,

        confidence,

        original_image,

        heatmap_image,

        overlay_image,

        report_path

    ))

    conn.commit()

    conn.close()


# ============================================================
# GET ALL REPORTS FOR CURRENT SESSION
# ============================================================

def get_all_reports(session_id):

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *

        FROM reports

        WHERE session_id = ?

        ORDER BY id DESC

    """, (session_id,))

    reports = cursor.fetchall()

    conn.close()

    return reports


# ============================================================
# SEARCH REPORTS FOR CURRENT SESSION
# ============================================================

def search_reports(session_id, keyword):

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *

        FROM reports

        WHERE session_id = ?

        AND (

            patient_name LIKE ?

            OR patient_id LIKE ?

        )

        ORDER BY id DESC

    """, (

        session_id,

        f"%{keyword}%",

        f"%{keyword}%"

    ))

    reports = cursor.fetchall()

    conn.close()

    return reports


# ============================================================
# GET ONE REPORT
# ============================================================

def get_report(session_id, report_id):

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *

        FROM reports

        WHERE id = ?

        AND session_id = ?

    """, (

        report_id,

        session_id

    ))

    report = cursor.fetchone()

    conn.close()

    return report


# ============================================================
# STATISTICS FOR CURRENT SESSION
# ============================================================

def get_statistics(session_id):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    # Total
    cursor.execute("""
        SELECT COUNT(*)

        FROM reports

        WHERE session_id = ?

    """, (session_id,))

    total = cursor.fetchone()[0]

    # Normal
    cursor.execute("""
        SELECT COUNT(*)

        FROM reports

        WHERE session_id = ?

        AND LOWER(prediction) = 'normal'

    """, (session_id,))

    normal = cursor.fetchone()[0]

    # Pneumonia
    cursor.execute("""
        SELECT COUNT(*)

        FROM reports

        WHERE session_id = ?

        AND LOWER(prediction) != 'normal'

    """, (session_id,))

    pneumonia = cursor.fetchone()[0]

    # Average confidence
    cursor.execute("""
        SELECT AVG(confidence)

        FROM reports

        WHERE session_id = ?

    """, (session_id,))

    avg = cursor.fetchone()[0]

    conn.close()

    return {

        "total": total,

        "normal": normal,

        "pneumonia": pneumonia,

        "average": round(avg, 2) if avg else 0

    }


# ============================================================
# DELETE REPORT
# ============================================================

def delete_report(session_id, report_id):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM reports

        WHERE id = ?

        AND session_id = ?

    """, (

        report_id,

        session_id

    ))

    conn.commit()

    conn.close()




