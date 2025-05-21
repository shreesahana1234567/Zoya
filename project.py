# Cell 1 — Install dependencies (run once)
!pip install scikit-learn matplotlib seaborn

# Cell 2 — Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import random

sns.set(style='whitegrid')

# Cell 3 — AI Model Performance Visualization
def plot_ai_model_performance():
    accuracies = []
    estimators = [10, 50, 100, 150, 200]

    X, y = make_classification(n_samples=1000, n_features=6, n_informative=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for n in estimators:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(estimators, accuracies, marker='o')
    plt.title("AI Model Accuracy vs Number of Trees")
    plt.xlabel("Number of Trees in RandomForest")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1.0)
    plt.grid(True)
    plt.show()

plot_ai_model_performance()

# Cell 4 — Chatbot Latency Visualization
def plot_chatbot_latency():
    queries = ["inventory", "demand", "status", "unknown"]
    response_times = []

    def chatbot_response(user_input):
        responses = {"inventory": "Inventory is stable.",
                     "demand": "High demand next week.",
                     "status": "System operational."}
        for keyword in responses:
            if keyword in user_input.lower():
                return responses[keyword]
        return "Unknown query."

    for query in queries:
        start = time.time()
        _ = chatbot_response(query)
        end = time.time()
        response_times.append((end - start) * 1000) # in ms

    plt.figure(figsize=(8, 5))
    sns.barplot(x=queries, y=response_times, palette='muted')
    plt.title("Chatbot Response Time per Query")
    plt.xlabel("Query")
    plt.ylabel("Response Time (ms)")
    plt.ylim(0, max(response_times) + 5)
    plt.show()

plot_chatbot_latency()

# Cell 5 — IoT Sensor Data Simulation and Visualization
def plot_iot_sensor_data():
    timestamps = []
    temperatures = []
    humidities = []

    for i in range(20): # Collect 20 samples
        timestamps.append(i)
        temperatures.append(round(random.uniform(20.0, 30.0), 2))
        humidities.append(round(random.uniform(40.0, 60.0), 2))
        time.sleep(0.1) # Simulate near real-time

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(timestamps, temperatures, marker='o', color='red')
    plt.title("Simulated IoT Temperature Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")

    plt.subplot(1, 2, 2)
    plt.plot(timestamps, humidities, marker='o', color='blue')
    plt.title("Simulated IoT Humidity Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Humidity (%)")

    plt.tight_layout()
    plt.show()

plot_iot_sensor_data()
