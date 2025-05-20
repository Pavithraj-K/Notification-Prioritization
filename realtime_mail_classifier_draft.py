import imaplib
import email
from email.header import decode_header
import joblib
import time

# Login credentials
IMAP_SERVER = "imap.gmail.com"
EMAIL_ACCOUNT = "1nh22ai115@gmail.com"
PASSWORD = "yrcn gsoy gopj wxhr"  # Replace with your App Password

# Load pre-trained models and vectorizers
spam_vectorizer = joblib.load("spam_vectorizer.pkl")
spam_model = joblib.load("spam_classifier.pkl")

priority_vectorizer = joblib.load("notification_vectorizer.pkl")
priority_model = joblib.load("notification_classifier.pkl")

# Function to clean text for filenames or display
def clean(text):
    """Clean up text for filenames or display."""
    return "".join(c if c.isalnum() else "_" for c in text)

# Function to classify messages
def classify_message(message):
    # Step 1: Spam Classification
    message_tfidf = spam_vectorizer.transform([message])
    spam_prediction = spam_model.predict(message_tfidf)[0]
    
    if spam_prediction == 1:
        return "Spam"
    
    # Step 2: Priority Classification (only if not spam)
    notification_tfidf = priority_vectorizer.transform([message])
    priority_prediction = priority_model.predict(notification_tfidf)[0]
    
    return priority_prediction  # High Priority or Low Priority

# Function to read and classify new emails
def read_and_classify_emails(last_seen_id=None):
    try:
        # Connect to Gmail IMAP server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, PASSWORD)
        
        # Select the mailbox (default is "INBOX")
        mail.select("inbox")
        
        # Search for all emails in the inbox
        status, messages = mail.search(None, "ALL")
        if status != "OK":
            print("No emails found!")
            return last_seen_id

        # Get the list of email IDs
        email_ids = messages[0].split()
        new_emails = []

        # Filter for new emails
        if last_seen_id is not None:
            new_emails = [eid for eid in email_ids if int(eid) > int(last_seen_id)]
        else:
            new_emails = email_ids[-5:]  # Fetch the latest 5 emails for the first run
        
        if not new_emails:
            print("No new emails.")
            return last_seen_id

        for email_id in new_emails:
            # Fetch the email by ID
            res, msg = mail.fetch(email_id, "(RFC822)")
            for response in msg:
                if isinstance(response, tuple):
                    # Parse the raw email
                    msg = email.message_from_bytes(response[1])
                    
                    # Decode the email subject
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8")
                    
                    # Decode sender email
                    from_ = msg.get("From")
                    
                    # Extract the email body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()

                    # Combine subject and body for classification
                    email_content = f"{subject} {body}"
                    classification = classify_message(email_content)
                    
                    # Print results
                    print(f"From: {from_}")
                    print(f"Subject: {subject}")
                    print(f"Classification: {classification}")
                    print("=" * 50)

        # Update the last seen ID
        last_seen_id = email_ids[-1]
        mail.logout()
        return last_seen_id
    except Exception as e:
        print(f"Error: {e}")
        return last_seen_id

# Real-time email monitoring
def monitor_emails(interval=60):
    last_seen_id = None
    print("Starting real-time email monitoring...")
    while True:
        last_seen_id = read_and_classify_emails(last_seen_id)
        time.sleep(interval)  # Wait before checking again

# Start monitoring
monitor_emails(interval=60)  # Check every 60 seconds
