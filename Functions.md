# EvaDB Built in Functions

## String Functions
| Function | Description |
|----------|-------------|
| CONCAT | Concatenates multiple strings into one | 
| UPPER | Converts a string to uppercase |
| LOWER | Converts a string to lowercase | 


## Array Functions
| Function | Description |
|----------|-------------|
| Annotate | Annotates boundary boxes of images (draws rectangles on the image frames). Returns just the annotated image frames (the image with the boundary boxes drawn on) |
| ArrayCount | Counts the occurences of a specific element in each row of a given array | 
| Crop | Crops an image frame based on a boundary box | 
| FuzzDistance | Calculates the similarity betweeen two strings using fuzzy matching |
| GaussianBlur | Applies Gaussian blur to a dataframe | 
| HorizontalFlip | Applies a horizontal flip to a dataframe |
| VerticalFlip | Applies a vertical flip to a dataframe |
| Open | Loads image data from a file path | 
| Similarity | Calculates the similarity between a pair of feature vectors | 
| ToGrayscale | Converts a frame from BGR to grayscale | 
| TextFilterKeyword | Filters text based on specified keywords | 


## AI Functions
| Function | Description |
|----------|-------------|
| ASLActionRecognition | Uses a 3D ResNet Model to interpret and translate American Sign Language actions |
| ChatGPT | Allows processing queries with ChatGPT |
| DallE | Generates images using OpenAI's DALL_E model |
| EmotionDetector | Processes images using a modified VGG19 architecture and classifies emotion. There are 7 possible outputs: angry, disgust, fear, happy, sad, surprise, or neutral |
| FaceDetector | Uses MCTNN model to detect faces in images. Returns the bounding boxes and confidence scores for each detected face |
| fastrcnn | Uses a pre-trained Faster R-CNN model for object detection. Process images and uses the model to detect objects, returning predictions that include class labels, bounding boxes, and confidence scores |
| FeatureExtractor | Uses a ResNet model to process images and extract features, which are numerical represenatations of the image and can be thought of as a distilled essence of the image, capturing various aspects like shapes, textures, patterns, and other visual elements. These are typically used for further analysis |
| ForecastModel | Performs time series forecasting using a user specified model |
| GenericLudwigModel | Performs data prediction using a Ludwig model | 
| MnistImageClassifier | Classifies images using a model trained on the MNIST dataset | 
| MVITActionRecognition | Action recognition using a MViT model to recognize actions in video segments. Returns the predicted action label for each video segment |
| SaliencyFeatureExtractor | Extracts saliency maps from images using ResNet-18 model. Processes iamges and computes a saliency map, which identifies the most important parts of the image from the perspective of the model |
| SentenceTransformerFeatureExtractor | Extracts features (numerical representations) from text data using a sentence transformer model (all-MiniLM-L6-v2). |
| SiftFeatureExtractor | Extracts features from images using the Scale-Invariant Feature Transform algorithm | 
| GenericSklearnModel | Makes predictions using a model from Scikit-learn | 
| StableDiffusion | Generates images based on textual prompts using the Stable Diffusion model via the Replicate AI | 
| GenericXGBoostModel | Makes predictions using an XGBoost model | 
| yolo | Uses the YOLO object detection model to perform predictions on input images. Returns the class labels, bounding boxes, and confidence scores for images | 