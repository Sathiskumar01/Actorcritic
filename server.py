import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.utils import custom_object_scope

app = Flask(__name__)

# Directory to save models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# ✅ Custom Model (ActorNet)
# ----------------------------


class ActorNet(tf.keras.Model):
    def __init__(self, output_dim, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(output_dim, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "trainable": self.trainable  # ✅ Explicitly add trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CriticNet(tf.keras.Model):
    def __init__(self, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def get_config(self):
        config = super().get_config()
        config.update({"trainable": self.trainable})  # ✅ Explicitly add trainable
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ----------------------------
# ✅ Upload Model API
# ----------------------------
@app.route('/upload_model', methods=['POST'])
def upload_model():
    try:
        if 'actor_model' not in request.files or 'critic_model' not in request.files:
            return jsonify({"error": "Missing model files"}), 400
        
        actor_model_path = os.path.join(MODEL_DIR, "actor_model.h5")
        critic_model_path = os.path.join(MODEL_DIR, "critic_model.h5")
        
        request.files['actor_model'].save(actor_model_path)
        request.files['critic_model'].save(critic_model_path)
        
        return jsonify({"message": "Models uploaded successfully!"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# ✅ Test Model API
# ----------------------------
@app.route('/test', methods=['POST'])
def test_model():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No test file uploaded"}), 400

        # Load test dataset
        file = request.files['file']
        test_data = pd.read_csv(file)

        # Load saved models with custom object scope
        with custom_object_scope({'ActorNet': ActorNet}):
            actor = tf.keras.models.load_model(os.path.join(MODEL_DIR, "actor_model.h5"))
            critic = tf.keras.models.load_model(os.path.join(MODEL_DIR, "critic_model.h5"))

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

        return jsonify({"results": results, "average_loss": avg_loss}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# ✅ Run the Flask App
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
