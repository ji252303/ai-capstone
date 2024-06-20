// MediaPipe Hands 모듈 로드
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const hands = new Hands({locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 5});
            drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
        }
    }
    canvasCtx.restore();
    processColorDetection();
}

async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({video: true});
    videoElement.srcObject = stream;
    videoElement.addEventListener('loadeddata', () => {
        hands.send({image: videoElement});
    });
}

function processColorDetection() {
    let src = new cv.Mat(videoElement.height, videoElement.width, cv.CV_8UC4);
    let dst = new cv.Mat();
    let low = new cv.Mat(src.rows, src.cols, src.type(), [20, 100, 100, 0]);
    let high = new cv.Mat(src.rows, src.cols, src.type(), [30, 255, 255, 0]);
    cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
    cv.inRange(src, low, high, dst);
    cv.imshow(canvasElement, dst);
    src.delete(); dst.delete(); low.delete(); high.delete();
}

startWebcam();
