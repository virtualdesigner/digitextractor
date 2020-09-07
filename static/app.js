var canvas = document.getElementById('myCanvas');
var ctx = canvas.getContext('2d');

canvas.width = 200;
canvas.height = 200;
canvas.style.backgroundColor = "black"
var mouse = {
    x: 0,
    y: 0
};

canvas.addEventListener('mousemove', function (e) {
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
}, false);

ctx.lineWidth = 7;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = '#fff';

canvas.addEventListener('mousedown', function (e) {
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);

    canvas.addEventListener('mousemove', onPaint, false);
}, false);

canvas.addEventListener('mouseup', function () {
    canvas.removeEventListener('mousemove', onPaint, false);
}, false);

var onPaint = function () {
    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();
};

function onPredict() {
    const imageData = ctx.getImageData(0, 0, canvas.height, canvas.width);
    console.log(imageData.data)
    var dataURL = canvas.toDataURL();

    fetch(`${window.location.origin}/predict`, {
            method: "POST",
            body: JSON.stringify({
                "imgData": dataURL
            }),
            headers: {
                "Content-Type": "application/json"
            }
        })
        .then((response) => {
            return response.json();
        })
        .then((myJson) => {
            // console.log("Prediction: " + myJson.prediction);
            document.getElementById('prediction').innerText = myJson.prediction
        });

}