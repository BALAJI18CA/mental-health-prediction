import requests
url = "http://localhost:5000/predict"
payload = {
    "clinical": [30, 1, 0, 2],
    "physiological": [7.5, 8, 70, 8000, 60],
    "social_media": "Feeling okay today, just chilling",
    "chatbot": "I’m doing alright, how about you?",
    "depression_level": 1
}
response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
print(response.json())