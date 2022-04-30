import React, { Component, useRef, useEffect, useState } from "react";
import Button from "react-bootstrap/Button";
import "bootstrap/dist/css/bootstrap.css";
import "./App.css";

function App() {
  const [data, setdata] = useState({
    prediction: "",
  });

  
  let xhr = new XMLHttpRequest();
  let url = "http://localhost:5000/";
  xhr.open("POST", url, true);
  xhr.setRequestHeader("Content-Type", "application/json");
  const [image, setImage] = useState([]);

  

  const videoRef = useRef(null);
  const photoRef = useRef(null);

  const getVideo = () => {
    navigator.mediaDevices
      .getUserMedia({ video: { height: 720, width: 1280 } })
      .then((stream) => {
        let video = videoRef.current;
        video.srcObject = stream;
        video.play();
      })
      .catch((err) => {
        console.error(err);
      });
  };

  const takePhoto = () => {
    const width = 414;
    const height = width / (16 / 9);
    let video = videoRef.current;
    let photo = photoRef.current;

    photo.width = width;
    photo.height = height;

    let ctx = photo.getContext("2d");
    ctx.drawImage(video, 0, 0, width, height);
    const img = ctx.getImageData(0, 0, width, height).data;
    const arr = Array.from(img);
    xhr.send(JSON.stringify(arr));
  };

  useEffect(() => {
    getVideo();
  }, [videoRef]);
  

  // Using useEffect for single rendering
  useEffect(() => {
    // Using fetch to fetch the api from flask server it will be redirected to proxy
    fetch("/data").then((res) =>
      res.json().then((data) => {
        // Setting a data from api
        setdata({
          prediction: data.prediction,
        });
        //console.log(data.prediction);
      })
    );
  });

  return (
    <div className="App">
      <h1>Arabic signs language</h1>
      <div className="camera">
        {/*<video ref={videoRef}></video>*/}
        {<img src="http://localhost:5000/video_feed" alt="Video" />}
      </div>
      <div className="bb">
        <Button onClick={takePhoto}>SNAP</Button>
      </div>
      <div>
        <canvas ref={photoRef}></canvas>
      </div>
      <p className="r">
        <b>{data.prediction}</b>
      </p>
    </div>
  );
}

export default App;
