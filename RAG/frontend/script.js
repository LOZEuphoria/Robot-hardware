const face = document.getElementById('face');
const statusText = document.getElementById('status-text');
const subtitleText = document.getElementById('subtitle-text');
const visualizer = document.getElementById('visualizer');
const btnReset = document.getElementById('btn-reset');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let isProcessing = false;

const audioPlayer = new Audio();

let idleInterval = null;

function startIdleAnimation() {
    stopIdleAnimation();
    const idleStates = ['idle-normal', 'idle-look-around', 'idle-curious', 'idle-squint', 'idle-happy', 'idle-sad', 'idle-love', 'idle-normal'];
    
    // Set default immediately
    face.className = 'face-container idle-normal';
    
    idleInterval = setInterval(() => {
        if (!isRecording && !isProcessing) {
            const randomState = idleStates[Math.floor(Math.random() * idleStates.length)];
            face.className = 'face-container ' + randomState;
        }
    }, 3500); // Đổi biểu cảm mỗi 3.5 giây
}

function stopIdleAnimation() {
    if (idleInterval) {
        clearInterval(idleInterval);
        idleInterval = null;
    }
}

function setState(state) {
    if (state === 'idle') {
        startIdleAnimation();
        statusText.innerText = "Chạm vào màn hình để nói";
        visualizer.classList.add('hidden');
        subtitleText.classList.add('hidden');
    } else {
        stopIdleAnimation();
        face.className = 'face-container ' + state;
        
        if (state === 'listening') {
            statusText.innerText = "Đang lắng nghe...";
            visualizer.classList.remove('hidden');
            subtitleText.classList.add('hidden');
        } else if (state === 'thinking') {
            statusText.innerText = "Đang suy nghĩ...";
            visualizer.classList.remove('hidden');
        } else if (state.startsWith('speaking')) {
            statusText.innerText = "";
            visualizer.classList.add('hidden');
            subtitleText.classList.remove('hidden');
        }
    }
}

// KHỞI TẠO STATE BAN ĐẦU - Phải gọi ở đây để tránh lỗi TDZ (Temporal Dead Zone)
setState('idle');

document.body.addEventListener('click', async (e) => {
    if (e.target.id === 'btn-reset') return;

    if (isProcessing) return;

    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

btnReset.addEventListener('click', async () => {
    audioPlayer.pause();
    setState('idle');
    isProcessing = false;
    isRecording = false;
    subtitleText.innerText = "";
    
    try {
        await fetch('/api/reset', { method: 'POST' });
        statusText.innerText = "Đã reset trí nhớ. Chạm để nói.";
    } catch (e) {
        console.error(e);
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        const recorderOptions = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 128000 }
            : {};
        mediaRecorder = new MediaRecorder(stream, recorderOptions);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            await sendAudioToAPI(audioBlob);
        };

        mediaRecorder.start();
        isRecording = true;
        setState('listening');
    } catch (err) {
        console.error("Lỗi Microphone:", err);
        alert("Vui lòng cấp quyền Microphone cho trình duyệt.");
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        setState('thinking');
        isProcessing = true;
    }
}

async function sendAudioToAPI(blob) {
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");

    try {
        const response = await fetch('/api/chat_voice', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            statusText.innerText = "Không nghe rõ. Chạm để thử lại.";
            setState('idle');
            isProcessing = false;
            return;
        }

        subtitleText.innerText = "Bạn: " + data.user_text;
        subtitleText.classList.remove('hidden');
        statusText.innerText = "";
        
        setTimeout(() => {
            playResponse(data.ai_text, data.audio_url);
        }, 1500);

    } catch (error) {
        console.error("Lỗi API:", error);
        statusText.innerText = "Lỗi kết nối máy chủ!";
        setState('idle');
        isProcessing = false;
    }
}

function analyzeEmotion(text) {
    const t = text.toLowerCase();
    if (t.includes('không tìm thấy') || t.includes('xin lỗi') || t.includes('rất tiếc') || t.includes('chưa biết') || t.includes('tôi không biết')) {
        return 'sad';
    }
    if (t.includes('tuyệt vời') || t.includes('chào') || t.includes('cảm ơn') || t.includes('vui lòng') || t.includes('hỗ trợ') || t.includes('đây là')) {
        return 'happy';
    }
    if (t.includes('cảnh báo') || t.includes('nghiêm cấm') || t.includes('không được phép') || t.includes('kỷ luật') || t.includes('đình chỉ')) {
        return 'angry';
    }
    if (t.includes('bất ngờ') || t.includes('wow') || t.includes('thật không') || t.includes('đột xuất')) {
        return 'surprised';
    }
    return 'neutral';
}

function playResponse(text, audioUrl) {
    if (!audioUrl) {
        setState('idle');
        isProcessing = false;
        return;
    }

    subtitleText.innerText = text;
    
    // Phân tích cảm xúc dựa trên câu trả lời của AI
    const emotion = analyzeEmotion(text);
    setState('speaking-' + emotion);

    audioPlayer.src = audioUrl + "?t=" + new Date().getTime();
    
    audioPlayer.onended = () => {
        setState('idle');
        isProcessing = false;
    };
    
    audioPlayer.play().catch(e => {
        console.error("Auto-play bị chặn:", e);
        setState('idle');
        isProcessing = false;
    });
}
