const classifyBtn = document.getElementById('classifyBtn');
const emailInput = document.getElementById('emailInput');
const resultDiv = document.getElementById('resultDisplay');

classifyBtn.addEventListener('click', async () => {
    const emailText = emailInput.value.trim();
    if (!emailText) {
        alert("Please paste some email text first.");
        return;
    }

    // Reset UI
    classifyBtn.disabled = true;
    resultDiv.style.display = 'block';
    resultDiv.className = 'result pending';
    resultDiv.innerText = "Processing...";

    try {
        // Send email for classification
        const response = await fetch('/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: emailText })
        });

        const initData = await response.json();
        const taskId = initData.task_id;

        // Poll for result
        const pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/result/${taskId}`);
                const data = await res.json();

                if (data.status === "done") {
                    clearInterval(pollInterval);
                    classifyBtn.disabled = false;

                    const prediction = data.prediction;
                    resultDiv.innerText = "Prediction: " + prediction;

                    // Style result
                    const isSpam = prediction.toLowerCase().includes('spam') && !prediction.toLowerCase().includes('not');
                    resultDiv.className = isSpam ? "result spam" : "result not-spam";

                } else if (data.status === "error") {
                    clearInterval(pollInterval);
                    classifyBtn.disabled = false;
                    resultDiv.innerText = "Error: " + data.message;
                    resultDiv.className = 'result error';
                }

            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 1500);

    } catch (err) {
        console.error("Request error:", err);
        resultDiv.innerText = "Failed to connect to server.";
        resultDiv.className = 'result error';
        classifyBtn.disabled = false;
    }
});