"""Built-in calibration prompt corpora for QNN QDQ quantization.

These are small, varied prompt sets shipped with the tool so users
don't need to supply calibration data for the default flow.
"""
from __future__ import annotations

# Short, diverse prompts covering instruct / QA / reasoning / code / chat patterns.
# Intentionally kept under ~50 tokens each to allow fast calibration.
MIXED_SMALL: list[str] = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "Summarize the key events of World War II in three sentences.",
    "How does photosynthesis work?",
    "Translate the following to Spanish: Hello, how are you?",
    "What are the main differences between TCP and UDP?",
    "Write a haiku about autumn.",
    "Explain the concept of supply and demand in economics.",
    "What is machine learning and how does it differ from traditional programming?",
    "List five tips for effective time management.",
    "Describe the water cycle in detail.",
    "What causes earthquakes?",
    "Write a short story about a robot learning to paint.",
    "Explain the theory of relativity in simple terms.",
    "What are the benefits of regular exercise?",
    "How do neural networks learn?",
    "Compare and contrast democracy and authoritarianism.",
    "What is the significance of the Turing test?",
    "Describe the process of DNA replication.",
    "What are the primary causes of climate change?",
    "Write a SQL query to find the top 10 customers by revenue.",
    "Explain the difference between supervised and unsupervised learning.",
    "What is the Pythagorean theorem?",
    "Describe how a compiler works.",
    "What are the layers of the OSI model?",
    "Explain blockchain technology in simple terms.",
    "What is the difference between a stack and a queue?",
    "How does encryption work?",
    "What are the main components of a CPU?",
    "Explain the concept of recursion with an example.",
    "What is the difference between RAM and ROM?",
    "Describe the structure of an atom.",
    "What is natural language processing?",
    "Explain the concept of object-oriented programming.",
    "What are the SOLID principles in software engineering?",
    "How does a search engine work?",
    "What is the difference between HTTP and HTTPS?",
    "Explain the concept of cloud computing.",
    "What are the advantages of using version control?",
    "Describe the architecture of a transformer model.",
    "What is transfer learning?",
    "Explain gradient descent in machine learning.",
    "What are the main types of database joins?",
    "How does garbage collection work in programming languages?",
    "What is the CAP theorem?",
    "Explain the difference between REST and GraphQL.",
    "What is containerization and how does Docker work?",
    "Describe the MapReduce programming model.",
    "What are attention mechanisms in deep learning?",
    "Explain the concept of eventual consistency.",
    "What is the difference between threads and processes?",
    "How does a hash table work internally?",
    "What is the purpose of a load balancer?",
    "Explain the concept of microservices architecture.",
    "What are design patterns in software engineering?",
    "How does public key cryptography work?",
    "What is the difference between precision and recall?",
    "Explain the concept of data normalization in databases.",
    "What is a knowledge graph?",
    "Describe how a convolutional neural network works.",
    "What are the main challenges of distributed systems?",
    "Explain the concept of idempotency in APIs.",
    "What is the difference between compilation and interpretation?",
]

INSTRUCT_SMALL: list[str] = MIXED_SMALL[:32]
CHAT_SMALL: list[str] = MIXED_SMALL[32:]

BUILTIN_CORPORA: dict[str, list[str]] = {
    "builtin:mixed_small": MIXED_SMALL,
    "builtin:instruct_small": INSTRUCT_SMALL,
    "builtin:chat_small": CHAT_SMALL,
}

DEFAULT_CORPUS_ID = "builtin:mixed_small"
