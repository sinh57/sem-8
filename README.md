# 📘 Semester 8 - SPPU Computer Engineering (2019 Pattern)

**Final Year – Semester 2 Lab Assignments**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/your-repo?style=social)](https://github.com/yourusername/your-repo/stargazers)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

⭐ **If this repository helped you, don't forget to give it a star!**

---

## 🧠 Contents

- [HPC (High Performance Computing) Practicals](#-hpc-high-performance-computing-practicals)
  - [OpenMP Programs (Linux/Windows Guide)](#-openmp-programs-linuxwindows-guide)
  - [CUDA Programs (Google Colab Setup)](#-cuda-programs-google-colab-setup)

---

## ⚡ HPC Practicals

### 🪟 Windows Users Note

Make sure MinGW is installed with pthread support:
👉 [MinGW Installation Guide](https://stackoverflow.com/a/39256203)

---

## 🧵 Running OpenMP Programs

#### 🔹 Step 1: Open Terminal

Press: `Ctrl + Alt + T`

#### 🔹 Step 2: Go to your file directory

```bash
cd ~/Desktop
```

#### 🔹 Step 3: Compile with OpenMP

**Format:**
```bash
g++ path/to/file/file_name.cpp -fopenmp -o output
```

**Example:**
```bash
g++ hpc1.cpp -fopenmp -O2 -o hpc1
```

#### 🔹 Step 4: Run the program

**Format:**
```bash
./output
```

**Example:**
```bash
./hpc1
```

---

## 🛠️ Troubleshooting

### If g++ is missing

**Check version:**
```bash
g++ --version
```

**Install compiler:**
```bash
sudo apt update
sudo apt install g++
```

### If OpenMP is not working

**Install build tools:**
```bash
sudo apt install build-essential
```

Then compile again using: `-fopenmp`

---

## ⚡ One-line Compile & Run (Exam Fast Method)

```bash
g++ hpc1.cpp -fopenmp -O2 -o hpc1 && ./hpc1
```

---

## 🔒 Permission Error Fix

```bash
chmod +x hpc1
./hpc1
```

---

## 🧾 Quick Format Summary

**Compile:**
```bash
g++ file.cpp -fopenmp
```

**Run:**
```bash
./a.out   # Linux
./a.exe   # Windows
```

---

## 🚀 CUDA Programs on Google Colab

#### 🔹 Step 1

Go to: 👉 [Google Colab](https://colab.research.google.com)

#### 🔹 Step 2

Create a new notebook (`.ipynb` file)

#### 🔹 Step 3

Enable GPU:
```
Runtime → Change runtime type → GPU
```

#### 🔹 Step 4

Install CUDA support:
```python
!pip install git+https://github.com/afnan47/cuda.git
```

#### 🔹 Step 5

Load CUDA extension:
```python
%load_ext nvcc_plugin
```

#### 🔹 Step 6

Test program:
```cuda
%%cu
#include <iostream>

int main() {
    std::cout << "Hello World\n";
    return 0;
}
```

---

## ⚠️ Important Rule

Always add:
```cuda
%%cu
```
before writing any CUDA program.

---

## 📚 Reference

If errors occur, refer to:
👉 [GeeksforGeeks CUDA Guide](https://www.geeksforgeeks.org/how-to-run-cuda-c-c-on-jupyter-notebook-in-google-colaboratory/)

---

## ⭐ Final Note

If you found this helpful, consider starring ⭐ the repository — it motivates contributors!

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

## 👤 Author

**Your Name** - [Your GitHub Profile](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- SPPU Computer Engineering Curriculum
- OpenMP Community
- CUDA Documentation
