# Android General Knowledge Chatbot

A simple general knowledge chatbot built with Python and Kivy, designed to run on Android devices.

## Features

- Clean and intuitive user interface
- Real-time chat interaction
- Basic general knowledge responses
- Easy to extend with more knowledge

## Requirements

- Python 3.7+
- Kivy 2.2.1
- NLTK 3.8.1
- Requests 2.31.0

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure you're in the project directory
2. Run the application:
   ```
   python main.py
   ```

## Building for Android

To build for Android, you'll need to use Buildozer. Here are the steps:

1. Install Buildozer:
   ```
   pip install buildozer
   ```

2. Initialize Buildozer in the project directory:
   ```
   buildozer init
   ```

3. Build the Android APK:
   ```
   buildozer android debug
   ```

## Usage

1. Launch the application
2. Type your message in the text input field
3. Press 'Send' or hit Enter to send your message
4. The chatbot will respond with relevant information

## Extending the Chatbot

To add more knowledge to the chatbot, modify the `responses` dictionary in the `ChatBot` class within `main.py`. Add new keywords and their corresponding responses to enhance the chatbot's knowledge base. 