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
    image: HTMLImageElement
  ) => {
    const canvasInput = document.createElement("canvas");
    const canvasOutput = document.createElement("canvas");
    canvasInput.width = width;
    canvasInput.height = height;
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
      let dst = new cv.Mat();
      let gray = new cv.Mat();

      // gray and thresho ld image
      cv.cvtColor(cvImg, gray, cv.COLOR_BGR2GRAY, 0);
      cv.threshold(gray, gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
      // find contours, hierarchy via opencv.js
      let contours = new cv.MatVector();
      let hierarchy = new cv.Mat();
      cv.findContours(
        cvImg,
        contours,
        hierarchy,
        cv.RETR_CCOMP,
        cv.CHAIN_APPROX_SIMPLE
      );
      let hull = new cv.Mat();
      let defect = new cv.Mat();
      let cnt = contours.get(0);
      let lineColor = new cv.Scalar(255, 0, 0);
      let circleColor = new cv.Scalar(255, 255, 255);
      cv.convexHull(cnt, hull, false, false);
      cv.convexityDefects(cnt, hull, defect);

      for (let i = 0; i < defect.rows; ++i) {
        let start = new cv.Point(
          cnt.data32S[defect.data32S[i * 4] * 2],
          cnt.data32S[defect.data32S[i * 4] * 2 + 1]
        );
        let end = new cv.Point(
          cnt.data32S[defect.data32S[i * 4 + 1] * 2],
          cnt.data32S[defect.data32S[i * 4 + 1] * 2 + 1]
        );
        let far = new cv.Point(
          cnt.data32S[defect.data32S[i * 4 + 2] * 2],
          cnt.data32S[defect.data32S[i * 4 + 2] * 2 + 1]
        );
        cv.line(dst, start, end, lineColor, 2, cv.LINE_AA, 0);
        cv.circle(dst, far, 3, circleColor, -1);
      }
      cv.imshow("canvasOutput", dst);
      cvImg.delete();
      dst.delete();
      hierarchy.delete();
      contours.delete();
      hull.delete();
      defect.delete();
    }

    return canvasOutput;
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

        detections.forEach((item: any) => {
          const x = item["bbox"][0];
          const y = item["bbox"][1];
          const width = item["bbox"][2];
          const height = item["bbox"][3];

          // Crop the image and append it as a new div element
          const canvas = cropImg(x, y, width, height, image);
          document.getElementById("main")?.appendChild(canvas);

          // Draw the bounding box.
          context.strokeStyle = "#00FFFF";
          context.lineWidth = 4;
          context.strokeRect(x, y, width, height);

          // Draw the label background.
          context.fillStyle = "#00FFFF";
          const textWidth = context.measureText(
            item["label"] + " " + (100 * item["score"]).toFixed(2) + "%"
          ).width;
          const textHeight = parseInt(font, 10); // base 10
          context.fillRect(x, y, textWidth + 4, textHeight + 4);
        });

        for (let i = 0; i < detections.length; i++) {
          const item = detections[i];
          const x = item["bbox"][0];
          const y = item["bbox"][1];
          const content =
            item["label"] + " " + (100 * item["score"]).toFixed(2) + "%";

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
  );
}
