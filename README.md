# Text-to-Image Generation using Hugging Face and Nvidia CUDA

This project is a high-performance text-to-image generation pipeline that utilizes Hugging Face models for natural language processing and Nvidia CUDA for accelerated GPU-based computations. The solution includes a desktop application built using the CustomTKinter package for an interactive user interface, enabling users to input text and view generated images directly from their desktop.

## Features

### Text-to-Image Generation: 
Converts input text into high-quality images using state-of-the-art Hugging Face models.

### CUDA Acceleration: 
Utilizes Nvidia CUDA for GPU-accelerated processing, reducing image generation time significantly compared to CPU-based methods.

### Custom Desktop UI: 
Built using the CustomTKinter package, providing a modern and user-friendly interface for interaction.

### Efficient Package Management: 
Uses pip-tools for dependency management, ensuring compatible versions of packages are installed automatically.

### Customizable Parameters: 
Adjust the model and generation settings for flexibility and experimentation.

## Prerequisites
Before setting up the project, ensure you have the following dependencies installed:

   - Python 3.8 or higher
   - Nvidia CUDA Toolkit
   - Nvidia GPU with CUDA support
   - PyTorch with CUDA support
   - Hugging Face Transformers library
   - ```pip-tools``` for managing dependencies
   - ```CustomTKinter``` for creating the desktop UI

## Installation
Follow these steps to set up the project locally:
1. Install appropriate Nvidia CUDA toolkit from their official website.
2. Clone the repository
   ```bash
   git clone https://github.com/bhavyom-singh/Text-to-ImageDesktopApp.git
   cd Text-to-ImageDesktopApp
   ```
3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows use `venv\Scripts\activate`
   ```
4. Install ```pip-tools```:
   ```bash
   pip install pip-tools
   ```
5. Install required packages using ```requirements.in```:

   This project uses ```pip-tools``` to maintain and resolve dependencies. List all the required packages in ```requirements.in```. To install compatible versions of all packages, run:
   ```bash
   pip-compile requirements.in
   pip-sync requirements.txt
   ```
   The pip-compile command reads the ```requirements.in``` file, resolves compatible versions, and generates a ```requirements.txt``` file. The ```pip-sync``` command then installs the packages listed in ```requirements.txt``` to your environment.

6. Install PyTorch with CUDA support:

   Don't put ```pytorch``` in the ```requirements.in``` file. This is the only exception for this project. Instead get the required pytorch with cuda from Pytorch's website and install it using ```pip```.
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
## Usage

After installation, you can run the desktop application using the following command:
```bash
python gui.py
```

### Using the Desktop Application

1. Select a directory to store the generated image
2. Input your desired description in the text box provided in the UI. 
3. Click the "Generate Image" button to start the text-to-image conversion process.
4. The generated image will be displayed within the UI and also saved at the location selected earlier.
## Contributing

Contributions are welcome! If you would like to improve this project or add new features, feel free to fork the repository and create a pull request.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

- Hugging Face Transformers for providing pre-trained models and support for text-to-image generation.
- PyTorch for its powerful GPU acceleration capabilities.
- Nvidia CUDA for enabling high-performance computation.
- ```pip-tools``` for simplifying dependency management and version compatibility.
- CustomTKinter for providing a modern, user-friendly interface for the desktop application.