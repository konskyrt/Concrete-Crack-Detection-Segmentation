import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isFilePicked, setIsFilePicked] = useState(false);
  const [predUrl, setPredUrl] = useState();
  const [srcUrl, setSrcUrl] = useState();
  const [side1Url, setSide1Url] = useState();
  const [side2Url, setSide2Url] = useState();
  const [side3Url, setSide3Url] = useState();
  const [side4Url, setSide4Url] = useState();
  const [side5Url, setSide5Url] = useState();
  const [fusedUrl, setFusedUrl] = useState();
  const [width, setImgWidth] = useState(1);
  const [height, setImgHeight] = useState(1);
  const [unit, setImgUnit] = useState('m');

  const changeHandler = (event) => {
      setSelectedFile(event.target.files[0]);
      setIsFilePicked(true);
      if (event.target.files && event.target.files[0]) {
        setSrcUrl(
          URL.createObjectURL(event.target.files[0])
        );
}
  };
  const setWidth = (event) => {
    setImgWidth(event.target.value)
  }
  const setUnit = (event) => {
    setImgUnit(event.target.value)
  }
  const setHeight = (event) => {
    setImgHeight(event.target.value)
  }
  const handleSubmit = event => {
    event.preventDefault();
    const formData = new FormData();
    formData.append(
      "file",
      selectedFile,
      selectedFile.name
    );

    console.log(width, height)
  const requestOptions = {
      method: 'POST',
      //headers: { 'Content-Type': 'multipart/form-data' }, // DO NOT INCLUDE HEADERS
      body: formData,
  };
    fetch('http://127.0.0.1:8000/predict/'+width+'-' + height +'/' + unit, requestOptions)
    .then(response => response.json())//.blob())
    .then(data => {
      console.log(data)
        setPredUrl("data:image/png;base64,"+data[0]);
        setFusedUrl("data:image/png;base64,"+data[1]);
        setSide1Url("data:image/png;base64,"+data[2]);
        setSide2Url("data:image/png;base64,"+data[3]);
        setSide3Url("data:image/png;base64,"+data[4]);
        setSide4Url("data:image/png;base64,"+data[5]);
        setSide5Url("data:image/png;base64,"+data[6]);
    })
    .catch((err) => console.log(err));
  }
  return (  <div className="d-flex">
      <div className="card mx-5">
        <div className="mx-auto mt-5">
          <Form onSubmit={handleSubmit}>
            <input name="image" type="file" onChange={changeHandler} accept=".jpeg, .png, .jpg"/>
            <Form.Group className="mb-3">
            <Form.Label>Real width</Form.Label>
            <Form.Control type="text" placeholder="some number"  onChange={setWidth}/>
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Real height</Form.Label>
            <Form.Control type="text" placeholder="some number" onChange={setHeight}/>
          </Form.Group>
          <Form.Group className="mb-3">
            <Form.Label>Measurement unit</Form.Label>
            <Form.Control type="text" placeholder="some unit" onChange={setUnit}/>
          </Form.Group>
            <Button variant="primary" type="submit">Predict</Button>
          </Form>
        </div>
      </div>
      <div className="card">
        <div className="d-flex mx-auto">
          <h1 className="mt-2">Crack Detection using DeepCrack</h1>
        </div>
        <div className="d-flex mx-auto">
          <div className="image-container m-4">
            <img src={srcUrl} width="512" height="512" alt="" />
            <h3>Selected Image</h3>
          </div>
          <div className="image-container m-4">
            <img src={predUrl} width="512" height="512" alt="" />
            <h3>Prediction Visualization</h3>
          </div>
        </div>
        <div className="d-flex mx-auto">
          <h3 className="mt-2">Intermediate Predictions:</h3>
        </div>
        <div className="d-flex mx-auto">

          <div className="image-container m-3">
            <img src={side1Url} width="256" height="256" alt="" />
            <h3>Side 1 Prediction</h3>
          </div>
          <div className="image-container m-3">
            <img src={side2Url} width="256" height="256" alt="" />
            <h3>Side 2 Prediction</h3>
          </div>
          <div className="image-container m-3">
            <img src={side3Url} width="256" height="256" alt="" />
            <h3>Side 3 Prediction</h3>
          </div>
          <div className="image-container m-3">
            <img src={side4Url} width="256" height="256" alt="" />
            <h3>Side 4 Prediction</h3>
          </div>
          <div className="image-container m-3">
            <img src={side5Url} width="256" height="256" alt="" />
            <h3>Side 5 Prediction</h3>
          </div>
        </div>
        <div className="d-flex mx-auto">
          <h3 className="mt-2">Fused Prediction:</h3>
        </div>
        <div className="mx-auto">
        <div className="image-container m-3">
          <img src={fusedUrl} width="256" height="256" alt="" />
        </div>
        </div>
      </div>
  </div>
);
}

export default App;
