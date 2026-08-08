import sqlite3


DATABASE = "patients.db"


def create_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS reports(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

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

    conn.commit()

    conn.close()


def save_report(

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

    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)

    """,(

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

def get_all_reports():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""

        SELECT *

        FROM reports

        ORDER BY id DESC

    """)

    reports = cursor.fetchall()

    conn.close()

    return reports

def search_reports(keyword):

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""

        SELECT *

        FROM reports

        WHERE

            patient_name LIKE ?

            OR patient_id LIKE ?

        ORDER BY id DESC

    """, (

        f"%{keyword}%",

        f"%{keyword}%"

    ))

    reports = cursor.fetchall()

    conn.close()

    return reports

def get_report(report_id):

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""

        SELECT *

        FROM reports

        WHERE id=?

    """,(report_id,))

    report = cursor.fetchone()

    conn.close()

    return report

def get_statistics():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM reports")
    total = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM reports WHERE LOWER(prediction)='normal'"
    )
    normal = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM reports WHERE LOWER(prediction)!='normal'"
    )
    pneumonia = cursor.fetchone()[0]

    cursor.execute(
        "SELECT AVG(confidence) FROM reports"
    )

    avg = cursor.fetchone()[0]

    conn.close()

    return {

        "total": total,

        "normal": normal,

        "pneumonia": pneumonia,

        "average": round(avg,2) if avg else 0

    }


def delete_report(report_id):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM reports WHERE id=?",
        (report_id,)
    )

    conn.commit()

    conn.close()