from locust import HttpUser, task, between

class StreamlitUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_sleep_score(self):
        # Simulate form submission
        response = self.client.get("/", json={
            "screen_time_phone": 6.0,
            "screen_time_laptop": 4.0,
            "step_count": 9000,
            "screen_content": "Relaxing / Meditative Content",
            "model_selection": "Linear Regression"
        })

        # Print response for debug purposes (optional)
        print(response.text)

        # Check response time
        assert response.elapsed.total_seconds() < 2, "Response time exceeded 2 seconds"

# Command to run Locust:
# locust -f locustfile.py --host http://localhost:8501
