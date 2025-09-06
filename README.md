
````markdown
# Mocktail-Robot

A simple mocktail-making system using **ESP32** and **Raspberry Pi 4**.

---

## Getting Started

### Prerequisites
- **Hardware**: ESP32 board, Raspberry Pi 4  
- **Software**: Python (for Raspberry Pi), Arduino IDE or compatible (for ESP32)

---

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/AwesomeBlossomm/Mocktail-Robot.git
   cd Mocktail-Robot
````

2. **ESP32 Code**

   * Open one of the `.ino` files (`esp_code_output.ino` or `esp_codes_limit_fix.ino`) in Arduino IDE.
   * Make sure ESP32 board support is installed.
   * Upload the sketch to your ESP32.

3. **Raspberry Pi Script**

   * Ensure Python 3.x is installed.
   * Run the script on your Raspberry Pi:

     ```bash
     python3 python_code_inputs.py
     ```
   * This script handles input reception and communicates with the ESP32.

4. **Optional**

   * Run `Raspi_fixes_limit_UI_receipt.py` if you want to test UI or receipt features.

---

## Developers

* Justine Juliana G. Balla
* Angelo Miguel S. Dacumos
* Diana Marie N. Carreon
* Kristine Mae P. Prado
* Kristine Mae A. Verona
* Jury Andrew Nathaniel F. Lebosada
* Marvin S. Gamo

---
