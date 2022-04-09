import "./App.css";
import React, { Component, useEffect, useState } from "react";
import Form from "react-bootstrap/Form";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Button from "react-bootstrap/Button";
import "bootstrap/dist/css/bootstrap.css";

function App() {
  const [data, setdata] = useState({
    prediction: "",
  });

  // Using useEffect for single rendering
  useEffect(() => {
    // Using fetch to fetch the api from flask server it will be redirected to proxy
    fetch("/data").then((res) =>
      res.json().then((data) => {
        // Setting a data from api
        setdata({
          prediction: data.prediction,
        });
        console.log(data.prediction);
      })
    );
  });

  return (
    <div className="title">
      <header className="result-container">
        <h1>Arabic signs language</h1>
        {/* Calling a data from setdata for showing */}
      </header>
      <body>
        <img src="http://localhost:5000/video_feed" alt="Video" />
        <p className="r">
          <b>{data.prediction}</b>
        </p>
      </body>
    </div>
  );
}

export default App;
