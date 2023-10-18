import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
const boundsX = 10;
const boundsY = 7.5;
const initialStateURL = "https://saved-work.desmos.com/calc-states/production/dnm2bpvm6c";
var elt = document.getElementById("calculator");
var calculator = Desmos.GraphingCalculator(elt, {
  expressions: false,
  zoomButtons: false,
  lockViewport: true,
  border: false
});
let initialStateCall = async () => await fetch(initialStateURL).then((response) => response.json()).then((responseJson) => {
  let initialState = responseJson;
  console.log("setting calc state!");
  calculator.setState(initialState);
});
initialStateCall();
calculator.setExpression({id: "0", latex: "P_{0}=\\left(0,0\\right)"});
calculator.setExpression({id: "1", latex: "P_{1}=\\left(0,0\\right)"});
function normalizedToGraphCoordinate(landmark) {
  return {
    x: -1 * (landmark.x * 2 * boundsX - boundsX),
    y: -1 * (landmark.y * 2 * boundsY - boundsY),
    z: landmark.z
  };
}
const demosSection = document.getElementById("demos");
let handLandmarker = void 0;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const createHandLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode,
    numHands: 2
  });
  demosSection.classList.remove("invisible");
};
createHandLandmarker();
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
function enableCam(event) {
  if (!handLandmarker) {
    console.log("Wait! objectDetector not loaded yet.");
    return;
  }
  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  }
  const constraints = {
    video: true
  };
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}
let lastVideoTime = -1;
let results = void 0;
console.log(video);
async function predictWebcam() {
  canvasElement.style.width = video.videoWidth;
  ;
  canvasElement.style.height = video.videoHeight;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({runningMode: "VIDEO"});
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
  }
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (results.landmarks) {
    console.log(results);
    if (results.handednesses.length == 1) {
      let handednesses = results.handednesses[0][0].categoryName;
      if (handednesses === "Right") {
        let graphBoundedLandmarkRight = normalizedToGraphCoordinate(results.landmarks[0][8]);
        drawLandmarks(canvasCtx, [results.landmarks[0][8]], {color: "#FF0000", lineWidth: 2});
        calculator.setExpression({
          id: "0",
          latex: `P_{0}=\\left(${graphBoundedLandmarkRight.x},${graphBoundedLandmarkRight.y}\\right)`
        });
      } else {
        let graphBoundedLandmarkLeft = normalizedToGraphCoordinate(results.landmarks[0][8]);
        drawLandmarks(canvasCtx, [results.landmarks[0][8]], {color: "#00FF00", lineWidth: 2});
        calculator.setExpression({
          id: "1",
          latex: `P_{1}=\\left(${graphBoundedLandmarkLeft.x},${graphBoundedLandmarkLeft.y}\\right)`
        });
      }
    } else if (results.handednesses.length == 2) {
      let handednesses = results.handednesses[0][0].categoryName;
      if (handednesses === "Right") {
        let graphBoundedLandmarkRight = normalizedToGraphCoordinate(results.landmarks[0][8]);
        let graphBoundedLandmarkLeft = normalizedToGraphCoordinate(results.landmarks[1][8]);
        drawLandmarks(canvasCtx, [results.landmarks[0][8]], {color: "#FF0000", lineWidth: 2});
        drawLandmarks(canvasCtx, [results.landmarks[1][8]], {color: "#00FF00", lineWidth: 2});
        calculator.setExpression({
          id: "0",
          latex: `P_{0}=\\left(${graphBoundedLandmarkRight.x},${graphBoundedLandmarkRight.y}\\right)`
        });
        calculator.setExpression({
          id: "1",
          latex: `P_{1}=\\left(${graphBoundedLandmarkLeft.x},${graphBoundedLandmarkLeft.y}\\right)`
        });
      } else {
        let graphBoundedLandmarkRight = normalizedToGraphCoordinate(results.landmarks[1][8]);
        let graphBoundedLandmarkLeft = normalizedToGraphCoordinate(results.landmarks[0][8]);
        drawLandmarks(canvasCtx, [results.landmarks[0][8]], {color: "#00FF00", lineWidth: 2});
        drawLandmarks(canvasCtx, [results.landmarks[1][8]], {color: "#FF0000", lineWidth: 2});
        calculator.setExpression({
          id: "0",
          latex: `P_{0}=\\left(${graphBoundedLandmarkRight.x},${graphBoundedLandmarkRight.y}\\right)`
        });
        calculator.setExpression({
          id: "1",
          latex: `P_{1}=\\left(${graphBoundedLandmarkLeft.x},${graphBoundedLandmarkLeft.y}\\right)`
        });
      }
    }
  }
  canvasCtx.restore();
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
