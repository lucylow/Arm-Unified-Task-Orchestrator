// Example: Voice command integration for accessibility (Web Speech API)
const recognition = new window.webkitSpeechRecognition();
recognition.onresult = (event) => {
    const instruction = event.results[0][0].transcript;
    // POST instruction to backend /api/v1/nl/execute
    fetch('/api/v1/nl/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: instruction })
    });
};
recognition.start();
