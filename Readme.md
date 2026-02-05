# Note Bot

Note Bot is a web application built with Streamlit that uses the Google Gemini Pro model to convert your notes into a more organized format.

## How it works

You enter your notes into the text area, and when you click the "Convert" button, the app sends your notes to the Gemini Pro model. The model then processes the notes and returns a converted, more structured version.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- A Google Gemini API key. You can obtain one from [Google AI Studio](https://aistudio.google.com/).

### Installation

1.  **Clone the repository or download the code.**

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    You need to set your Gemini API key as an environment variable named `GEMINI_API_KEY`.

    On Windows:

    ```bash
    set GEMINI_API_KEY="YOUR_API_KEY"
    ```

    On macOS/Linux:

    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    ```

### Running the Application

Once you have installed the dependencies and set up your API key, you can run the Streamlit app with the following command:

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## Usage

1.  Open the web application in your browser.
2.  Enter your notes in the text area provided.
3.  Click the "Convert" button.
4.  The converted notes will be displayed below the button.

## License

This project is for academic/demo purposes.
