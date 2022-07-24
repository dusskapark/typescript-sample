import React, { useRef, useState, useEffect } from "react";
import cv from "@techstark/opencv-js";
import * as tf from "@tensorflow/tfjs";
import axios from "axios";
tf.setBackend("webgl");

export default function App() {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imgLoaded, setImgLoad] = useState(false);

  // Load the image and crop it based on x,y,w,h value. After that, draw it on the new canvas
  const cropImg = (
    x: number,
    y: number,
    width: number,
    height: number,
    image: HTMLImageElement,
    label: string,
    index: number
  ) => {
    const canvasInput = document.createElement("canvas");
    canvasInput.width = width;
    canvasInput.height = height;
    const canvasOutput = document.createElement("canvas");
    canvasOutput.id = label + `_${index}`;
    const ctx = canvasInput.getContext("2d");
    if (ctx) {
      ctx.drawImage(image, x, y, width, height, 0, 0, width, height);

      /////////////////////////////////////////
      //
      // process image with opencv.js
      // <canvas> elements named canvasInput and canvasOutput have been prepared.
      // In the image data from canvasInput, you can find icons and these icons are normally around the boundaries of icons where foreground and background meet.
      // So, you can use Watershed algorithm from opencv.js to obtain only foreground area from background area.
      //
      /////////////////////////////////////////
      let cvImg = cv.imread(canvasInput);
      let gray = new cv.Mat();

      // gray and threshold image
      cv.cvtColor(cvImg, gray, cv.COLOR_BGR2GRAY, 0);
      cv.threshold(gray, gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
      // save the threshold image in canvasOutput
      cv.imshow(canvasOutput, gray);
    }

    return canvasOutput;
  };

  // Find the contours of the image and draw them on the same canvas
  const findContours = (canvas: HTMLCanvasElement) => {
    let src = cv.imread(canvas);
    let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(src, src, 100, 200, cv.THRESH_BINARY);
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    // find contours and draw them in dst
    cv.findContours(
      src,
      contours,
      hierarchy,
      cv.RETR_CCOMP,
      cv.CHAIN_APPROX_SIMPLE
    );

    // get the convexity Defects of the contours
    let hull = new cv.Mat();
    let defect = new cv.Mat();
    let cnt = contours.get(0);
    let circleColor = new cv.Scalar(255, 255, 255); // white color

    cv.convexHull(cnt, hull, false, false);
    cv.convexityDefects(cnt, hull, defect);

    // draw convexity defects in dst
    for (let i = 0; i < defect.rows; ++i) {
      let far = new cv.Point(
        cnt.data32S[defect.data32S[i * 4 + 2] * 2],
        cnt.data32S[defect.data32S[i * 4 + 2] * 2 + 1]
      );
      cv.circle(dst, far, 3, circleColor, -1);
    }

    // get HoughLines of the contours
    let lines = new cv.Mat();
    let lineColor = new cv.Scalar(255, 0, 0); // red color
    cv.Canny(src, src, 50, 200, 3);
    cv.HoughLinesP(src, lines, 1, Math.PI / 180, 2, 0, 0);

    // draw HoughLinesP on the canvas
    for (let i = 0; i < lines.rows; i++) {
      let startPoint = new cv.Point(
        lines.data32S[i * 4],
        lines.data32S[i * 4 + 1]
      );
      let endPoint = new cv.Point(
        lines.data32S[i * 4 + 2],
        lines.data32S[i * 4 + 3]
      );
      cv.line(dst, startPoint, endPoint, lineColor);
    }

    // console.log canvas id and how many convexity defects and HoughLinesP are found
    console.log(`canvas id: ${canvas.id}`);
    console.log(`convexity defects: ${defect.rows}`);
    console.log(`HoughLinesP: ${lines.rows}`);

    // show on the contours canvas
    cv.imshow(canvas, dst);

    // delete opencv.js objects for memory release
    src.delete();
    dst.delete();
    hierarchy.delete();
    contours.delete();
    hull.delete();
    defect.delete();
    lines.delete();

    //return canvas for further processing
    return canvas;
  };

  const handleImgLoad = () => {
    console.log("image loaded...");
    setImgLoad(true);
  };

  const getLabelByID = (dir: { name: string; id: number }[], i: number) => {
    let label = dir.filter((x) => x.id === i);
    return label[0].name;
  };

  const loadImage = (img: HTMLImageElement | null) => {
    if (!img) return;
    console.log("Pre-processing image...");
    const tfimg = tf.browser.fromPixels(img).toInt();
    const expandedimg = tfimg.expandDims();
    return expandedimg;
  };

  const predict = async (inputs: any, model: any) => {
    console.log("Running predictions...");
    const predictions = await model.executeAsync(inputs);
    return predictions;
  };

  const renderPredictions = (
    predictions: any,

    width: number,
    height: number,
    classesDir: { name: string; id: number }[]
  ) => {
    console.log("Highlighting results...", predictions);
    //Getting predictions
    const boxes = predictions[6].arraySync();
    const scores = predictions[5].arraySync();
    const classes = predictions[7].dataSync();

    let detectionObjects: any = [];

    scores[0].forEach((score: number, i: number) => {
      console.log(classes[i]);
      if (score > 0.25) {
        const bbox = [];
        const minY = boxes[0][i][0] * height;
        const minX = boxes[0][i][1] * width;
        const maxY = boxes[0][i][2] * height;
        const maxX = boxes[0][i][3] * width;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;

        detectionObjects.push({
          class: classes[i],
          label: getLabelByID(classesDir, classes[i]),
          score: score.toFixed(4),
          bbox: bbox,
        });
      }
    });

    return detectionObjects;
  };

  useEffect(() => {
    const run = async (
      image: HTMLImageElement | null,
      c: HTMLCanvasElement | null
    ) => {
      try {
        const context = c?.getContext("2d");
        if (!context || !image) return;
        context.drawImage(image, 0, 0);

        // Font options.
        const font = "16px sans-serif";
        context.font = font;
        context.textBaseline = "top";
        const baseURL =
          "https://raw.githubusercontent.com/dusskapark/design-system-detector/master/icon/rico/models/mobilenetv2-8k/web-model";
        const model = await tf.loadGraphModel(baseURL + "/model.json");
        const classes = await axios.get(baseURL + "/label_map.json");
        const classesDir = classes.data;

        const expandedimg = loadImage(image);
        const predictions = await predict(expandedimg, model);
        const detections: any = renderPredictions(
          predictions,
          image?.width || 0,
          image?.height || 0,
          classesDir
        );
        console.log("interpreted: ", detections);

        detections.forEach((detection: any, i: number) => {
          const x = detection["bbox"][0];
          const y = detection["bbox"][1];
          const width = detection["bbox"][2];
          const height = detection["bbox"][3];
          const label = detection["label"];

          // Crop the image and append it as a new div element
          const canvas = cropImg(x, y, width, height, image, label, i);
          const contours = findContours(canvas);
          document.getElementById("main")?.appendChild(canvas);
          document.getElementById("main")?.appendChild(contours);

          // Draw the bounding box.
          context.strokeStyle = "#00FFFF";
          context.lineWidth = 4;
          context.strokeRect(x, y, width, height);

          // Draw the label background.
          context.fillStyle = "#00FFFF";
          const textWidth = context.measureText(
            label + " " + (100 * detection["score"]).toFixed(2) + "%"
          ).width;
          const textHeight = parseInt(font, 10); // base 10
          context.fillRect(x, y, textWidth + 4, textHeight + 4);
        });

        for (let i = 0; i < detections.length; i++) {
          const detection = detections[i];
          const x = detection["bbox"][0];
          const y = detection["bbox"][1];
          const content =
            detection["label"] +
            " " +
            (100 * detection["score"]).toFixed(2) +
            "%";

          // Draw the text last to ensure it's on top.
          context.fillStyle = "#000000";
          context.fillText(content, x, y);
        }
      } catch (e) {
        console.log(e);
      }
    };

    if (imgLoaded) {
      run(imgRef.current, canvasRef.current);
    }
  }, [imgLoaded]);

  return (
    <div className="App">
      <div className="App-header">
        <h2>Design System Detector</h2>
      </div>
      <div className="App-body">
        <div id="main">
          <img
            style={{ position: "absolute" }}
            ref={imgRef}
            id="image"
            onLoad={handleImgLoad}
            alt=""
            src="Clock.png"
            width="400"
            height="888"
            crossOrigin="anonymous"
          />
          <canvas
            style={{ position: "relative" }}
            ref={canvasRef}
            id="canvas"
            width="400"
            height="888"
          />
        </div>
        <div id="library"></div>
      </div>
    </div>
  );
}
