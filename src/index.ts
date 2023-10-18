// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const boundsX = 10
const boundsY = 7.5 

// Calculator code
var elt = document.getElementById('calculator');
var calculator = Desmos.GraphingCalculator(elt, {
  expressions: false,
  // settingsMenu: false,
  zoomButtons: false,
  lockViewport: true,
  border: false
});

calculator.setExpression({ id: '0', latex: 'P_{0}=\\left(0,0\\right)' });
calculator.setExpression({ id: '1', latex: 'P_{1}=\\left(0,0\\right)' });


function normalizedToGraphCoordinate(landmark) {
  return {
    x: -1 * ((landmark.x * 2 * boundsX) - boundsX),
    y: -1 * ((landmark.y * 2 * boundsY) - boundsY),
    z: landmark.z
  }
}

const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton: HTMLButtonElement;
let webcamRunning: Boolean = false;

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createHandLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numHands: 2
  });
  demosSection.classList.remove("invisible");
};
createHandLandmarker();

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/web_js
********************************************************************/

const video = document.getElementById("webcam") as HTMLVideoElement;
const canvasElement = document.getElementById(
  "output_canvas"
) as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d");

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
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

  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastVideoTime = -1;
let results = undefined;
console.log(video);
async function predictWebcam() {
  canvasElement.style.width = video.videoWidth;;
  canvasElement.style.height = video.videoHeight;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
  }
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (results.landmarks) {
    console.log(results)
    if (results.handednesses.length == 1) {
      let handednesses = results.handednesses[0][0].categoryName
      
      // P0 = Right Hand
      // P1 = Left Hand

      if (handednesses === "Right") {
        // One hand, right hand
        let graphBoundedLandmarkRight = normalizedToGraphCoordinate(results.landmarks[0][8])

        drawLandmarks(canvasCtx, [results.landmarks[0][8]], { color: "#FF0000", lineWidth: 2 });

        calculator.setExpression({
          id: '0',
          latex: `P_{0}=\\left(${graphBoundedLandmarkRight.x},${graphBoundedLandmarkRight.y}\\right)`
        });
      } else {
        // One hand, left hand
        let graphBoundedLandmarkLeft = normalizedToGraphCoordinate(results.landmarks[0][8])

        drawLandmarks(canvasCtx, [results.landmarks[0][8]], { color: "#00FF00", lineWidth: 2 });

        calculator.setExpression({
          id: '1',
          latex: `P_{1}=\\left(${graphBoundedLandmarkLeft.x},${graphBoundedLandmarkLeft.y}\\right)`
        });
      }

    } else if (results.handednesses.length == 2) {
      let handednesses = results.handednesses[0][0].categoryName

      if (handednesses === "Right") {
        // Two hands, 0 is Right hand, 1 is left hand
        let graphBoundedLandmarkRight = normalizedToGraphCoordinate(results.landmarks[0][8])
        let graphBoundedLandmarkLeft = normalizedToGraphCoordinate(results.landmarks[1][8])

        drawLandmarks(canvasCtx, [results.landmarks[0][8]], { color: "#FF0000", lineWidth: 2 });
        drawLandmarks(canvasCtx, [results.landmarks[1][8]], { color: "#00FF00", lineWidth: 2 });

        calculator.setExpression({
          id: '0',
          latex: `P_{0}=\\left(${graphBoundedLandmarkRight.x},${graphBoundedLandmarkRight.y}\\right)`
        });
        calculator.setExpression({
          id: '1',
          latex: `P_{1}=\\left(${graphBoundedLandmarkLeft.x},${graphBoundedLandmarkLeft.y}\\right)`
        });
      } else {
        // Two hands, 1 is Right hand, 0 is left hand
        let graphBoundedLandmarkRight = normalizedToGraphCoordinate(results.landmarks[1][8])
        let graphBoundedLandmarkLeft = normalizedToGraphCoordinate(results.landmarks[0][8])

        drawLandmarks(canvasCtx, [results.landmarks[0][8]], { color: "#00FF00", lineWidth: 2 });
        drawLandmarks(canvasCtx, [results.landmarks[1][8]], { color: "#FF0000", lineWidth: 2 });

        calculator.setExpression({
          id: '0',
          latex: `P_{0}=\\left(${graphBoundedLandmarkRight.x},${graphBoundedLandmarkRight.y}\\right)`
        });
        calculator.setExpression({
          id: '1',
          latex: `P_{1}=\\left(${graphBoundedLandmarkLeft.x},${graphBoundedLandmarkLeft.y}\\right)`
        });
      }

    }
  }
  canvasCtx.restore();

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}