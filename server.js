import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Directory to save models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

ACTOR_MODEL_PATH = os.path.join(MODEL_DIR, "actor_model.h5")
CRITIC_MODEL_PATH = os.path.join(MODEL_DIR, "critic_model.h5")

# ----------------------------
# Upload Model API
# ----------------------------
@app.route('/upload_model', methods=['POST'])
def upload_model():
    try:
        if 'actor_model' not in request.files or 'critic_model' not in request.files:
            return jsonify({"error": "Missing model files"}), 400

        request.files['actor_model'].save(ACTOR_MODEL_PATH)
        request.files['critic_model'].save(CRITIC_MODEL_PATH)
        
        return jsonify({"message": "Models uploaded successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Test Model API
# ----------------------------
@app.route('/test', methods=['POST'])
def test_model():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No test file uploaded"}), 400

        if not os.path.exists(ACTOR_MODEL_PATH) or not os.path.exists(CRITIC_MODEL_PATH):
            return jsonify({"error": "Models not found. Upload models first!"}), 400

        # Load test dataset
        file = request.files['file']
        test_data = pd.read_csv(file)

        # Load models only once
        actor = tf.keras.models.load_model(ACTOR_MODEL_PATH)
        critic = tf.keras.models.load_model(CRITIC_MODEL_PATH)

        results = []
        total_loss = 0

        for _, row in test_data.iterrows():
            state = np.array([
                row['TransmissionPower'],
                row['CurrentChannelPowerGain'],
                row['CrossChannelPowerGain'],
                row['QoSScore']
            ], dtype=np.float32)

            state = np.expand_dims(state, axis=0)
            action_probs = actor.predict(state, verbose=0)[0]
            chosen_action = np.argmax(action_probs)
            predicted_value = critic.predict(state, verbose=0)[0][0]

            # QoS and Reward are same
            qos_score = row['QoSScore']
            reward = qos_score

            # Critic Loss = MSE between predicted value and actual reward
            loss = (reward - predicted_value) ** 2
            total_loss += loss

            results.append({
                "TransmissionPower": row['TransmissionPower'],
                "ChosenAction": int(chosen_action),
                "QoSScore": float(qos_score),
                "Reward": float(reward),
                "PredictedValue": float(predicted_value),
                "Loss": float(loss)
            })

        avg_loss = total_loss / len(test_data)
        print(f"Test Completed - Average Loss: {avg_loss:.4f}")

        return jsonify({"results": results, "average_loss": avg_loss}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running"}), 200

# ----------------------------
# Run the Flask App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
