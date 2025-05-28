## Edge Detection using Genetic Algorithm

This project implements an edge detection algorithm based on a Genetic Algorithm (GA) optimization strategy, inspired by the IEEE research paper:
**"Edge Detection Technique Using Genetic Algorithm Based Optimization"**
🔗 [Read the paper on IEEE Xplore](https://ieeexplore.ieee.org/document/5054605)

### Overview

Traditional edge detection methods like Sobel, Canny, and Prewitt apply fixed kernels and thresholds. This project introduces an evolutionary approach, where a population of candidate thresholds evolves over generations using:

* **Fitness Evaluation**
* **Crossover**
* **Mutation**
* **Selection**

The optimal threshold learned through this process is then applied to highlight the edges in an image.

### How to Run

1. Clone the repository.
2. Place the image (`10081.jpg`) in the project directory or change the image filename in `main.py`.
3. Run the program:

   ```bash
   python main.py
   ```

### File Structure

* `main.py` – Main execution script that loads the image, computes gradients using Sobel kernels, and evolves threshold values via a Genetic Algorithm.
* `functions.py` – Contains helper functions:

  * `fitness`
  * `mating`
  * `crossover`
  * `mutation`
* `10081.jpg` – Sample image used for edge detection.
* `README.md` – Project documentation.

### Dependencies

* `numpy`
* `opencv-python`
* `torch`
* `matplotlib`

Install all requirements using:

```bash
pip install -r requirements.txt
```

### Result

The script displays a side-by-side comparison of:

* The edge map obtained using the learned threshold.
* The original grayscale image.