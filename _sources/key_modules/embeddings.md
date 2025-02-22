# Embeddings

## 1. Concept
Creating embeddings for different types of data (text, images, videos) involves transforming these inputs into a numerical form that machines can understand and process efficiently. Each type of embedding focuses on capturing the essential features of its respective data type.

### 1.1. Text Embeddings
Text embeddings convert textual data into numerical vectors. Each vector represents the semantic meaning of the text, enabling us to process and compare texts based on their meaning rather than just their raw form. Techniques like OpenAI Embedding use large-scale language models to understand context and nuances in language. SentenceTransformerEncoder, on the other hand, is specifically designed to create sentence embeddings, often using BERT-like models.

Consider two sentences:

- "A young boy is playing soccer in a park."

- "A child is kicking a football on a playground."

Text embedding models would transform these sentences into two high-dimensional vector (*e.g.*, 1536 dimension if using `text-embedding-3-small`). Despite different wordings, the vectors will be similar, capturing the shared concept of a child playing a ball game outdoors. This transformation into vectors allows machines to understand and compare the semantic similarities between the context.

### 1.2. Image Embeddings
Image embeddings convert images into numerical vectors, capturing essential features like shapes, colors, textures, and spatial hierarchies. This transformation is typically performed by Convolutional Neural Networks (CNNs) or other advanced neural network architectures designed for image processing. The resulting embeddings can be used for tasks like image classification, similarity comparison, and retrieval.

Suppose we have an image of a cat. An image embedding model would analyze the visual content (e.g., the shape of the ears, the pattern of the fur) and convert it into a vector. This vector encapsulates the essence of the image, allowing the model to recognize it as a cat and differentiate it from other images.


## 2. Types

### 2.1. `OpenAIEmbedding`
Utilizes OpenAI's models for generating text embeddings. This will requires OpenAI API Key.

### 2.2. `MistralEmbedding`
Utilizes Mistral's models for generating text embeddings. This will requires Mistral API Key.

### 2.3. `SentenceTransformerEncoder`
Utilizes open-source models from the Sentence Transformers library for generating text embeddings.

### 2.4. `VisionLanguageEmbedding`
Utilizes OpenAI's models for generating image embeddings. This will requires OpenAI API Key.


## 3. Get Started
To use the embedding functionalities, you need to import the necessary classes.

### 3.1. Using `OpenAIEmbedding`
```python
from camel.embeddings import OpenAIEmbedding
from camel.types import EmbeddingModelType

# Initialize the OpenAI embedding with a specific model
openai_embedding = OpenAIEmbedding(model_type=EmbeddingModelType.TEXT_EMBEDDING_3_SMALL)

# Generate embeddings for a list of texts
embeddings = openai_embedding.embed_list(["Hello, world!", "Another example"])
```

### 3.2. Using `MistralEmbedding`
```python
from camel.embeddings import MistralEmbedding
from camel.types import EmbeddingModelType

# Initialize the OpenAI embedding with a specific model
mistral_embedding = MistralEmbedding(model_type=EmbeddingModelType.MISTRAL_EMBED)

# Generate embeddings for a list of texts
embeddings = mistral_embedding.embed_list(["Hello, world!", "Another example"])
```

### 3.3. Using `SentenceTransformerEncoder`
```python
from camel.embeddings import SentenceTransformerEncoder

# Initialize the Sentence Transformer Encoder with a specific model
sentence_encoder = SentenceTransformerEncoder(model_name='intfloat/e5-large-v2')

# Generate embeddings for a list of texts
embeddings = sentence_encoder.embed_list(["Hello, world!", "Another example"])
```

### 3.4. Using `VisionLanguageEmbedding`
```python
from camel.embeddings import VisionLanguageEmbedding

vlm_embedding = VisionLanguageEmbedding()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
test_images = [image, image]

embeddings = vlm_embedding.embed_list(test_images)
```
