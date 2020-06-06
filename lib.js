import * as tf from '@tensorflow/tfjs';
// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import * as cam from './cam';
// import jimp from 'jimp';

const MODEL_PATH = './model/model.json';
// const MODEL_PATH = 'https://firebasestorage.googleapis.com/v0/b/catch-of-the-day-45cb7.appspot.com/o/model%2Fmodel.json?alt=media';

export const IMAGE_SIZE = 512;

let model;
export const loadModel = async () => {
  // await tf.setBackend('wasm');
  console.log('loading model....');
  model = await tf.loadLayersModel(MODEL_PATH, {});

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1])).dispose();

  // Show model layers and summary
  model.summary();
};

export const clearModelMemory = () => {
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1])).dispose();
};

/**
 * Given an image element, makes a prediction through model returning the
 * probabilities of the top K classes.
 */
export async function predict(imgElement) {
  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    // const img = tf.browser.fromPixels(imgElement, 1);

    // Normalize the image from [0, 225] to [INPUT_MIN, INPUT_MAX]
    const normalizationConstant = 1.0 / 255.0;
    // const normalized = img.toFloat().mul(normalizationConstant);

    let img = tf.browser.fromPixels(imgElement, 1)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE], false)
      .expandDims(0)
      .toFloat()
      .mul(normalizationConstant)

    // const image = tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE], false);

    // Reshape to a single-element batch so we can pass it to predict.
    // const batched = image.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1]);

    startTime2 = performance.now();

    // Make a prediction through model.
    return model.predict(img);
  });

  // Convert logits to probabilities and class names.
  const classes = await logits.data();

  console.log('Predictions: ', classes);
  const labelName = classes[0] < 0.5? 'NORMAL' : 'PNEUMONIA';
  const probability = classes[0];

  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  console.log(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  // Return the best probability label
  return `${labelName} (${Math.floor(probability * 100)}%)`;
}

export function resizeImage(imgElement){
  const canvas = document.createElement('canvas');
  canvas.height=IMAGE_SIZE;
  canvas.width=IMAGE_SIZE;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height); // clear canvas
  ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

  const img = document.createElement('img');
  img.src = canvas.toDataURL("image/jpeg");;
  img.width = IMAGE_SIZE;
  img.height = IMAGE_SIZE;

  return img;
}

/*export async function gradCamOverlay(imgElement, canvas){
  // tf.browser.fromPixels() returns a Tensor from an image element.
  const img = tf.browser.fromPixels(imgElement, 1);

  // Normalize the image from [0, 225] to [INPUT_MIN, INPUT_MAX]
  const normalizationConstant = 1.0 / 255.0;
  const normalized = img.toFloat().mul(normalizationConstant);

  const image = tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE], false);
  // Reshape to a single-element batch so we can pass it to predict.
  const batched = image.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1]);

  // Calculate Grad-CAM heatmap.
  const xWithCAMOverlay = cam.gradClassActivationMap(model, 1, batched);

  // const imageH = xWithCAMOverlay.shape[1];
  // const imageW = xWithCAMOverlay.shape[2];
  // let imageData = xWithCAMOverlay.dataSync();

  const ctx = canvas.getContext('2d');
  //get the tensor shape
  const [height, width] = xWithCAMOverlay.shape;
  //create a buffer array
  const buffer = new Uint8Array(width * height * 4)
  //create an Image data var 
  const imageData = new ImageData(width, height);
  //get the tensor values as data
  const data = xWithCAMOverlay.dataSync();

  //map the values to the buffer
  var i = 0;
  for(var y = 0; y < height; y++) {
    for(var x = 0; x < width; x++) {
      var pos = (y * width + x) * 4;      // position in buffer based on x and y
      buffer[pos  ] = data[i]             // some R value [0, 255]
      buffer[pos+1] = data[i+1]           // some G value
      buffer[pos+2] = data[i+2]           // some B value
      buffer[pos+3] = 255;                // set alpha channel
      i+=3
    }
  }

  let index = 0;
  for (let i = 0; i < height; ++i) {
    for (let j = 0; j < width; ++j) {
      const inIndex = 3 * (i * height + j);
      buffer.set([Math.floor(data[inIndex])], index++);
      buffer.set([Math.floor(data[inIndex + 1])], index++);
      buffer.set([Math.floor(data[inIndex + 2])], index++);
      buffer.set([255], index++);
    }
  }

  //set the buffer to the image data
  imageData.data.set(buffer)
  
  //show the image on canvas
  ctx.putImageData(imageData, 0, 0);

  const imageEl = new Image();
  imageEl.onload = function() {
    ctx.drawImage(image, 0, 0);
  };

  new jimp({data: new Buffer(buffer), width: width, height: height}, function (err, image) {
    image.getBase64(jimp.AUTO, function(err, data) {  // Add err
      console.log(data);
      imageEl.src = data;
    });
  });

  // return tf.browser.toPixels(imageData);
}*/

window.loadModel = loadModel;
window.clearModelMemory = clearModelMemory;
window.predict = predict;
window.resizeImage = resizeImage;