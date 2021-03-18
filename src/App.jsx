import React, { useEffect } from 'react'
import * as faceapi from 'face-api.js'
// import me from './assets/joaquin.jpeg';

export const App = () => {
    const startFaceRegonizer = () => {
        const video = document.getElementById('video');
        navigator.getUserMedia = (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
            navigator.getUserMedia(
                {video: {}},
                stream => video.srcObject = stream,
                err => console.log(err)
            );
        recognizedFaces(video);
    }
    const recognizedFaces = async (video) => {
        const labels = ['Joaquin', 'Kavo', 'Kate'];
        const labelFaceDescriptions =  await Promise.all(
                labels.map(async label => {
                    // const remoteURL = 'http://tecdevsmx.com/face-recognition/imgs/';
                    const imgUrl = `${label}.jpeg`;
                    const img = await faceapi.fetchImage(imgUrl);
    
                    const fullFaceDescripcion = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                    if(!fullFaceDescripcion) throw new Error(`no hay caras para ${label}`);
                    const faceDescriptors = [fullFaceDescripcion.descriptor];
                    return new faceapi.LabeledFaceDescriptors(label, faceDescriptors)
                })
        )
        const faceMatcher = new faceapi.FaceMatcher( labelFaceDescriptions, 0.7);

        video.addEventListener('play', () => {
            const canvas = faceapi.createCanvasFromMedia(video);
            document.getElementById('main').append(canvas);
            const displaysize = {width: video.width, height: video.height};
            faceapi.matchDimensions(canvas, displaysize);

            setInterval(async () => {
                const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors();
                const resizedDetections = faceapi.resizeResults(detections, displaysize);
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
                const results = resizedDetections.map(fd => faceMatcher.findBestMatch(fd.descriptor));
                results.forEach((bestMach, i) => {
                    const box = resizedDetections[i].detection.box;
                    const text = bestMach.toString();
                    const drawBox = new faceapi.draw.DrawBox(box, {label: text});
                    drawBox.draw(canvas);
                });
            }, 100);
        });
    }
    useEffect(() => {
        Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
            faceapi.nets.mtcnn.loadFromUri('/models'),
            faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
            faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
            faceapi.nets.faceLandmark68TinyNet.loadFromUri('/models'),
            faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
            faceapi.nets.faceExpressionNet.loadFromUri('/models')
        ]).then(startFaceRegonizer).catch((err) => console.log(err));
    },[]);

    return (
        <div id="main" style={{
            display: 'flex',
            flexDirection:'column',
            alignItems:'center',
            justifyContent:'center'
        }}>
            <h1 style={{fontWeight: 'bold',fontSize:'40px'}}>Face recognition</h1>
            <video style={{border: '3px dashed black'}} width="700" height="500" autoPlay muted id="video"></video>
        </div>
    )
}
