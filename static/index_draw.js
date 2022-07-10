const canvas = document.getElementById('canvas');
const toolbar = document.getElementById('toolbar');
const ctx = canvas.getContext('2d');

const canvasOffsetX = canvas.offsetLeft;
const canvasOffsetY = canvas.offsetTop;

const x = document.getElementById("myImg");

canvas.width = x.width;
canvas.height = x.height;

let isPainting = false;
let lineWidth = 20;
let startX;
let startY;

toolbar.addEventListener('click', e => {
    if (e.target.id === 'clear') {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
});

toolbar.addEventListener('change', e => {
    if(e.target.id === 'stroke') {
        ctx.strokeStyle = e.target.value;
    }

    if(e.target.id === 'lineWidth') {
        lineWidth = e.target.value;
    }
    
});

const draw = (e) => {
    if(!isPainting) {
        return;
    }

    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';

    var rect = canvas.getBoundingClientRect();

    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
}

canvas.addEventListener('mousedown', (e) => {
    isPainting = true;

    var rect = canvas.getBoundingClientRect();

    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
});

canvas.addEventListener('mouseup', e => {
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);

download_img = function(el) {
  // get image URI from canvas object
  var imageURI = canvas.toDataURL("image/jpg");
  el.href = imageURI;
};


function saveMask() {
    var imgURL = canvas.toDataURL("image/jpg");
    console.log(imgURL);
    $.ajax({
      type: "POST",
      url: "/upload_mask", //I have doubt about this url, not sure if something specific must come before "/take_pic"
      data: imgURL,
      contentType: false,
      cache: false,
      processData: false,
      success: function(data) {
        if (data.success) {
          console.log('Your file was successfully uploaded!');
        } else {
          console.log('There was an error uploading your file!');
        }
      }
    }).done(function() {
      console.log("Sent");
    });
  }