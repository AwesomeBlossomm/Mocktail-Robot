---
# Mocktail-Robot

A simple mocktail-making system using ESP32 and Raspberry Pi 4.

---

##  Getting Started

###  Prerequisites
- **Hardware**: ESP32 board, Raspberry Pi 4
- **Software**: Python (for Pi script), Arduino-IDE or compatible (for ESP32)

###  Setup Steps

1. **Clone the repo**  
   ```bash
   git clone https://github.com/AwesomeBlossomm/Mocktail-Robot.git
   cd Mocktail-Robot
````

2. **ESP32 code**

   * Open one of the `.ino` files (`esp_code_output.ino` or `esp_codes_limit_fix.ino`) in Arduino IDE.
   * Make sure ESP32 board support is installed.
   * Upload the sketch to your ESP32.

3. **Raspberry Pi script**

   * Ensure you’re running Python (version 3.x recommended).
   * Run the script on your Pi:

     ```bash
     python3 python_code_inputs.py
     ```
   * This script handles input reception and likely controls GPIOs or communicates with the ESP32.

4. **(Optional)** Try out `Raspi_fixes_limit_UI_receipt.py` if you have UI or receipt features to test/fix.

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
