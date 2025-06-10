import smtplib
import os
from dotenv import load_dotenv

load_dotenv()

def send_email_alert(to_email, subject, body):
    sender_email = os.getenv("EMAIL_APP")
    sender_pass = os.getenv("EMAIL_APP_PASSWORD")
    
    if not sender_pass:
        print("Error: App password not found in environment.")
        return
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, to_email, f"Subject: {subject}\n\n{body}")
            print("Email sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)


#send_email_alert("himunagapure114@gmail.com", "Testing notify.py", "Test successful")
