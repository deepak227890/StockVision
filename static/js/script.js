document.addEventListener('DOMContentLoaded', () => {
    const predictForm = document.getElementById('predict-form');
    const predictButton = document.getElementById('predict-button');
    const buttonText = document.querySelector('.button-text');
    const spinner = document.querySelector('.spinner');
    const errorMessage = document.getElementById('error-message');
    const resultsSection = document.getElementById('results-section');

    predictForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        buttonText.textContent = 'Analyzing...';
        spinner.classList.remove('hidden');
        predictButton.disabled = true;
        errorMessage.textContent = '';
        resultsSection.classList.add('hidden');

        const formData = new FormData(predictForm);
        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) { throw new Error(data.error || 'An unknown error occurred.'); }
            updateResults(data);
        } catch (error) {
            errorMessage.textContent = `Error: ${error.message}`;
        } finally {
            buttonText.textContent = 'Predict';
            spinner.classList.add('hidden');
            predictButton.disabled = false;
        }
    });

    function updateResults(data) {
        const directionElement = document.getElementById('prediction-direction');
        directionElement.textContent = data.direction;
        directionElement.className = 'large-text';
        directionElement.classList.add(data.direction === 'UP' ? 'up-color' : 'down-color');

        const confidenceValue = data.confidence * 100;
        const confidenceBar = document.getElementById('confidence-bar');
        document.getElementById('confidence-value').textContent = `${confidenceValue.toFixed(1)}%`;
        confidenceBar.style.width = `${confidenceValue}%`;
        confidenceBar.style.backgroundColor = data.direction === 'UP' ? 'var(--up-color)' : 'var(--down-color)';
        
        document.getElementById('prob-up').textContent = `${(data.probabilities.up * 100).toFixed(1)}%`;
        document.getElementById('prob-down').textContent = `${(data.probabilities.down * 100).toFixed(1)}%`;
        
        const plotImage = document.getElementById('plot-image');
        const downloadLink = document.getElementById('download-link');
        const ticker = data.plot.split('/')[2].split('_')[0];
        
        document.getElementById('result-ticker').textContent = ticker;
        plotImage.src = data.plot + '?t=' + new Date().getTime();
        downloadLink.href = data.plot;
        downloadLink.download = `${ticker}_analysis.png`;

        resultsSection.classList.remove('hidden');
    }
});